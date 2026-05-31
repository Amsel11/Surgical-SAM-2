"""Fully-automatic OCR-anchored instrument masking — no manual clicking.

Pipeline (per OCR segment, so it survives tool swaps + bounds drift):
  1. Read instrument segments from all_segments_v2_burst.csv (one row per
     stable tool-set interval, with src-second bounds + per-arm instruments).
  2. For each segment intersecting the requested window:
       a. extract frames for that interval at --fps
       b. detect instruments on an anchor frame (GDINO ensemble + geometric
          priors + confidence gate), capped to the OCR instrument count
       c. carry obj_id identity across segments by greedy IoU vs the previous
          segment's boxes (boxes stay near their trocar port == their arm)
       d. seed SAM2 with box + corner tissue-negatives, propagate within segment
       e. drift QC: flag any object whose mask balloons past --drift-factor x its
          seed-box area (the leak signature)
  3. Concat per-segment overlays into one video + write summary.json.

Why per-segment instead of one long track: instruments are swapped many times
over a 40-min case and SAM2 memory drifts over thousands of frames — a single
frame-0 seed cannot stay valid. Re-anchoring per OCR segment is what makes it
consistent across the whole case.

Usage:
  python tools/auto_mask.py --video RAW.mp4 --video-id AS_whip_11637098_PJA \
      --segments-csv ~/Desktop/all_segments_v2_burst.csv \
      --start-sec 1600 --end-sec 1740 --out-dir OUT --fps 6
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.config import Stage2Inference, Stage2Outputs
from pipeline.io import davis_color_bgr, encode_mp4_from_jpgs, overlay_mask
from pipeline.models.sam2 import SAM2VideoTracker
from pipeline.prompts._grounding_dino_detector import iou_xyxy

from tools.instrument_detect import active_region, build_detector, detect_instruments

ARM_COLS = ["arm1", "arm2", "arm3", "arm4"]


def load_segments(csv_path, video_id, start_sec, end_sec):
    """Rows for video_id whose [start_sec,end_sec] intersects the window."""
    segs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["video_id"] != video_id:
                continue
            s, e = float(r["start_sec"]), float(r["end_sec"])
            if e < start_sec or s > end_sec:
                continue
            instruments = [r[c].strip() for c in ARM_COLS if r.get(c, "").strip()]
            segs.append({
                "start": max(s, start_sec), "end": min(e, end_sec),
                "instruments": instruments, "n": len(instruments),
            })
    return segs


def extract_frames(video, start_sec, dur, fps, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start_sec), "-i", str(video),
         "-t", str(dur), "-r", str(fps), "-q:v", "2",
         str(out_dir / "%05d.jpg")],
        check=True,
    )
    # ffmpeg numbers from 1; renumber to 0-based to match the loader's expectation
    frames = sorted(out_dir.glob("*.jpg"))
    for i, p in enumerate(frames):
        p.rename(out_dir / f"tmp_{i:05d}.jpg")
    for p in sorted(out_dir.glob("tmp_*.jpg")):
        p.rename(out_dir / p.name.replace("tmp_", ""))
    return len(frames)


def corner_negatives(box, frac=0.12):
    x0, y0, x1, y1 = box
    dw, dh = frac * (x1 - x0), frac * (y1 - y0)
    return [[x0 + dw, y0 + dh], [x1 - dw, y0 + dh],
            [x0 + dw, y1 - dh], [x1 - dw, y1 - dh]]


def match_identity(boxes, last_boxes, next_id, iou_thresh=0.2):
    """Assign obj_ids: greedy-match new boxes to last-known boxes (same trocar
    port/arm stays in a similar screen region across a tool swap)."""
    assigned, used = {}, set()
    pairs = sorted(
        ((iou_xyxy(b["box"], lb), i, oid)
         for i, b in enumerate(boxes) for oid, lb in last_boxes.items()),
        reverse=True,
    )
    for iou, i, oid in pairs:
        if iou < iou_thresh or i in assigned or oid in used:
            continue
        assigned[i], _ = oid, used.add(oid)
    out = {}
    for i, b in enumerate(boxes):
        oid = assigned.get(i)
        if oid is None:
            oid, next_id = next_id, next_id + 1
        out[oid] = b
    return out, next_id


def drift_report(masks_dir, seed_areas, drift_factor):
    """Flag objects whose mask SPIKES well above its own typical (median) size —
    the leak signature. Median, not seed-box area, is the reference: a small
    partial seed box must not make a steadily-larger (correct) mask look drifted."""
    masks = sorted(Path(masks_dir).glob("*.png"))
    if not masks:
        return {}
    areas = {oid: [] for oid in seed_areas}
    for m in masks:
        arr = np.array(Image.open(m))
        for oid in seed_areas:
            areas[oid].append(int((arr == oid).sum()))
    out = {}
    for oid in seed_areas:
        present = sorted(a for a in areas[oid] if a > 0)
        med = present[len(present) // 2] if present else seed_areas[oid]
        peak = max(areas[oid]) if areas[oid] else 0
        out[oid] = {"peak_area": peak, "median_area": med, "seed_area": seed_areas[oid],
                    "ratio": round(peak / max(1, med), 2), "drifted": peak > drift_factor * med}
    return out


def rerender_clean(seg_dir, frames_dir, seed_areas, drift_factor, fps, ui_line=None):
    """Re-render the overlay, suppressing an object only on frames where its mask
    SPIKES above its own typical (median) size — the leak signature. Using the
    median (not the seed-box area) as reference avoids nuking an object whose seed
    box was a small partial detection but whose true mask is steadily larger
    (that bug made the long grasper vanish). Rows at/below ui_line (UI bar) are
    zeroed so no mask renders in the instrument-label strip."""
    masks = sorted((Path(seg_dir) / "masks").glob("*.png"))
    if not masks:
        return None

    def load(m):
        pal = np.array(Image.open(m))
        if ui_line is not None:
            pal[int(ui_line):, :] = 0
        return pal

    # pass 1: per-object area on every frame -> typical (median over present frames)
    areas = {oid: [] for oid in seed_areas}
    for m in masks:
        pal = load(m)
        for oid in seed_areas:
            areas[oid].append(int((pal == oid).sum()))
    ref = {}
    for oid in seed_areas:
        present = sorted(a for a in areas[oid] if a > 0)
        ref[oid] = present[len(present) // 2] if present else seed_areas[oid]

    # pass 2: draw, dropping only spike frames (area > drift_factor * typical)
    clean_dir = Path(seg_dir) / "overlay_clean_jpgs"
    clean_dir.mkdir(exist_ok=True)
    for mi, m in enumerate(masks):
        frame = cv2.imread(str(Path(frames_dir) / f"{m.stem}.jpg"))
        if frame is None:
            continue
        pal = load(m)
        for oid in seed_areas:
            a = areas[oid][mi]
            if a == 0 or a > drift_factor * ref[oid]:
                continue
            frame = overlay_mask(frame, pal == oid, davis_color_bgr(oid))
        cv2.imwrite(str(clean_dir / f"{m.stem}.jpg"), frame)
    out_mp4 = Path(seg_dir) / "overlay_clean.mp4"
    encode_mp4_from_jpgs(clean_dir, out_mp4, fps)
    return out_mp4 if out_mp4.exists() else None


def run_tracker(args, seg_dir, frames_dir, n, objs_by_frame, seg_k):
    """Run SAM2 over a segment given prompts (obj boxes at one or more frames)."""
    pj = {"video": f"seg_{seg_k}", "n_frames": n,
          "prompt_frames": sorted(int(f) for f in objs_by_frame),
          "objects_by_frame": objs_by_frame}
    pj_path = Path(seg_dir) / "prompts.json"
    pj_path.write_text(json.dumps(pj, indent=2))
    cfg = Stage2Inference(
        model="sam2", checkpoint=args.checkpoint, config=args.config,
        device=args.device, bidirectional=True,
        outputs=Stage2Outputs(masks=True, overlay_video=True, overlay_jpgs=False, preview_small=False),
    )
    SAM2VideoTracker(cfg).run(video_id=f"seg_{seg_k}", frames_dir=Path(frames_dir),
                              prompts_json=pj_path, results_dir=Path(seg_dir), src_fps=args.fps)


def recover_prompts(seg_dir, frames_dir, detector, obj_ids, min_score, min_gap, ui_frac=0.08):
    """After a stable first pass, recover sustained dropouts. For each run of >=
    min_gap frames where an object's mask is absent, probe a few frames inside it
    and re-seed that object IFF the assignment is unambiguous:
      - it is the ONLY seeded object absent at the probe frame, and
      - a fresh detection sits in UNCLAIMED space (not covering another present
        object's mask).
    Returns [(frame, oid, box)]. The strict gating is what makes re-prompting
    safe: we never attach a box that belongs to a different instrument."""
    masks = sorted((Path(seg_dir) / "masks").glob("*.png"))
    if not masks:
        return []
    pals = [np.array(Image.open(m)) for m in masks]
    n = len(pals)
    areas = {oid: [int((p == oid).sum()) for p in pals] for oid in obj_ids}
    recoveries = []
    for oid in obj_ids:
        i = 0
        while i < n:
            if areas[oid][i] != 0:
                i += 1
                continue
            j = i
            while j < n and areas[oid][j] == 0:
                j += 1
            if j - i >= min_gap:
                for pf in (i + (j - i) // 2, i + (j - i) // 4, min(j - 1, i + 3 * (j - i) // 4)):
                    if [o for o in obj_ids if areas[o][pf] == 0] != [oid]:
                        continue  # ambiguous — more than just this object is missing
                    kept, _, _ = detect_instruments(detector, frames_dir / f"{pf:05d}.jpg",
                                                    min_score=min_score, ui_frac=ui_frac)
                    pal, best = pals[pf], None
                    for kb in kept:
                        x0, y0, x1, y1 = [int(v) for v in kb["box"]]
                        reg = pal[max(0, y0):y1, max(0, x0):x1]
                        if reg.size == 0:
                            continue
                        claimed = any((reg == o).sum() > 0.25 * reg.size
                                      for o in obj_ids if o != oid and areas[o][pf] > 0)
                        if not claimed and (best is None or kb["score"] > best["score"]):
                            best = kb
                    if best:
                        recoveries.append((pf, oid, best["box"]))
                        break  # one recovery seed per dropout stretch
            i = j
    return recoveries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--segments-csv", required=True)
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--end-sec", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--anchor-frac", type=float, default=0.5, help="where in the segment to detect")
    ap.add_argument("--min-score", type=float, default=0.16)
    ap.add_argument("--drift-factor", type=float, default=4.0)
    ap.add_argument("--recover", action="store_true", default=True,
                    help="second pass: re-seed sustained dropouts under their own obj_id")
    ap.add_argument("--no-recover", dest="recover", action="store_false")
    ap.add_argument("--recover-min-gap", type=int, default=12,
                    help="min consecutive blank frames before recovery kicks in")
    ap.add_argument("--neg-corners", action="store_true", default=True)
    ap.add_argument("--no-neg-corners", dest="neg_corners", action="store_false")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--checkpoint", default="./checkpoints/sam2.1_hiera_s_endo18.pth")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    segs = load_segments(args.segments_csv, args.video_id, args.start_sec, args.end_sec)
    if not segs:
        raise SystemExit(f"No segments for {args.video_id} in [{args.start_sec},{args.end_sec}]")
    print(f"{len(segs)} OCR segment(s) in window:")
    for s in segs:
        print(f"  {s['start']:.0f}-{s['end']:.0f}s  n={s['n']}  {s['instruments']}")

    detector = build_detector(device=args.device)
    last_boxes, next_id = {}, 1
    summary = {"video_id": args.video_id, "window": [args.start_sec, args.end_sec],
               "fps": args.fps, "segments": []}
    overlay_paths = []

    for k, seg in enumerate(segs):
        seg_dir = out / f"seg_{k:02d}"
        frames_dir = seg_dir / "frames"
        dur = seg["end"] - seg["start"]
        if dur < 1.0:
            print(f"  seg {k}: too short ({dur:.1f}s), skipping")
            continue
        n = extract_frames(args.video, seg["start"], dur, args.fps, frames_dir)
        print(f"\n=== seg {k}: {seg['start']:.0f}-{seg['end']:.0f}s, {n} frames ===")

        # Detect at several candidate anchors and seed ALL objects at the SINGLE
        # best frame (most detections, tie-broken by score). One shared
        # conditioning frame keeps obj_id identity stable through propagation.
        # (Seeding objects at different frames, or linking detections across
        # anchors, merged distinct instruments and made identity/colors flicker.)
        best = None
        for fr in (0.25, 0.45, 0.65, 0.85):
            a = min(n - 1, max(0, int(fr * n)))
            kb, _, _ = detect_instruments(detector, frames_dir / f"{a:05d}.jpg",
                                          min_score=args.min_score)
            ssum = sum(b["score"] for b in kb)
            print(f"  anchor@{a}: {len(kb)} det, score-sum {ssum:.2f}")
            if best is None or len(kb) > len(best[1]) or (
                    len(kb) == len(best[1]) and ssum > best[2]):
                best = (a, kb, ssum)
        anchor, kept = best[0], best[1]
        if seg["n"] and len(kept) > seg["n"]:
            kept = kept[:seg["n"]]
        if not kept:
            print("  no detections passed the gate — skipping segment")
            continue

        ided, next_id = match_identity(kept, last_boxes, next_id)
        objs, seed_areas = [], {}
        for oid, b in ided.items():
            x0, y0, x1, y1 = b["box"]
            seed_areas[oid] = (x1 - x0) * (y1 - y0)
            objs.append({"obj_id": oid, "positive": [], "box": b["box"],
                         "negative": corner_negatives(b["box"]) if args.neg_corners else []})
        last_boxes = {oid: b["box"] for oid, b in ided.items()}
        print(f"  seeding {len(objs)} objects @anchor {anchor}: " +
              ", ".join(f"obj{oid}={ided[oid]['query']}({ided[oid]['score']:.2f})" for oid in ided))

        # Pass 1: stable seeding (all objects at one shared anchor).
        objs_by_frame = {str(anchor): objs}
        run_tracker(args, seg_dir, frames_dir, n, objs_by_frame, k)

        # Pass 2: recover sustained dropouts by re-seeding the lost object under
        # its OWN id at a frame where it's unambiguously re-detected.
        if args.recover:
            rec = recover_prompts(seg_dir, frames_dir, detector, list(ided),
                                  args.min_score, args.recover_min_gap)
            if rec:
                print("  recovering dropouts: " +
                      ", ".join(f"obj{o}@{f}" for f, o, _ in rec))
                for pf, oid, box in rec:
                    objs_by_frame.setdefault(str(pf), []).append({
                        "obj_id": oid, "positive": [], "box": box,
                        "negative": corner_negatives(box) if args.neg_corners else []})
                run_tracker(args, seg_dir, frames_dir, n, objs_by_frame, k)

        ui_line = active_region(Image.open(frames_dir / f"{anchor:05d}.jpg"))[3]
        drift = drift_report(seg_dir / "masks", seed_areas, args.drift_factor)
        flagged = [oid for oid, d in drift.items() if d["drifted"]]
        if flagged:
            print(f"  ⚠ drift-flagged objects (mask ballooned): {flagged}")
        clean = rerender_clean(seg_dir, frames_dir, seed_areas, args.drift_factor, args.fps, ui_line)
        summary["segments"].append({
            "idx": k, "start": seg["start"], "end": seg["end"], "n_frames": n,
            "anchor": anchor, "instruments_ocr": seg["instruments"],
            "objects": {oid: {"query": b["query"], "score": b["score"], "box": b["box"]}
                        for oid, b in ided.items()},
            "drift": drift,
        })
        ov = clean if clean else (seg_dir / "overlay.mp4")
        if ov and Path(ov).exists():
            overlay_paths.append(Path(ov))

    # Concat per-segment overlays into one timeline video
    if overlay_paths:
        concat_txt = out / "_concat.txt"
        concat_txt.write_text("".join(f"file '{p.resolve()}'\n" for p in overlay_paths))
        subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", str(concat_txt), "-c", "copy", "-y", str(out / "combined.mp4")],
                       check=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary -> {out/'summary.json'}  Combined overlay -> {out/'combined.mp4'}")


if __name__ == "__main__":
    main()
