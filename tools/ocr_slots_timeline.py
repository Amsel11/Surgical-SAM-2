"""Production instrument-presence timeline via SPATIAL per-slot OCR.

Supersedes the whole-strip reader (tools/ocr_presence_timeline.py), which
cross-confused the adjacent right-side slots (arm 3 vs 4). Here we crop the
strip into fixed fractional slot regions and OCR each in isolation, so the
slot POSITION is the identity (arm#) by construction — no mis-numbering.

Pipeline (cheap: OCR only at stride points, not every frame):
  1. Stride-sample every --stride frames (instruments stay mounted for
     minutes, so this never misses a swap).
  2. At each sample, spatially crop the strip into N slots and OCR each for
     ONE canonical name (or none).
  3. Mode-filter the per-slot sample sequence over a small window to kill
     single-sample blips (slot 3 is ~94% under occlusion).
  4. Form segments = maximal runs of equal {slot->name}; refine each
     segment boundary to the EXACT frame via CPU strip pixel-diff.
  5. Expand to a per-frame timeline.

Outputs per video (--out-dir/<video_id>/): timeline.json, segments.csv,
qc.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_qwen_ocr_closed_vocab import load_vocab  # noqa: E402

# Fractional slot regions (generous overlap so a name is never cut). The
# da Vinci strip layout depends on aspect ratio: 16:9 recordings inset the
# strip with surgical video on the far left/right; 3:2 (720x480) recordings
# spread the slots across the full width. Tuned on RE_whip (1080p, 16:9) and
# DC_whip (480p, 3:2).
SLOT_FRACS_169 = {
    1: (0.13, 0.35),
    2: (0.31, 0.52),
    3: (0.48, 0.68),
    4: (0.64, 0.85),
}
SLOT_FRACS_32 = {
    1: (0.00, 0.30),
    2: (0.25, 0.52),
    3: (0.45, 0.76),
    4: (0.70, 1.00),
}


SLOTS = (1, 2, 3, 4)


def get_slot_fracs(w: int, h: int) -> dict:
    """Pick slot regions by aspect ratio (16:9 vs 3:2/other)."""
    return SLOT_FRACS_169 if (w / max(h, 1)) >= 1.6 else SLOT_FRACS_32


def src_idx_from_name(name: str):
    stem = Path(name).stem
    if stem.startswith("frame_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return None
    return None


def build_single_slot_prompt(canonical_names, modes):
    names = "\n".join(f"  - {c}" for c in canonical_names)
    mode_list = ", ".join(modes) if modes else "(none)"
    return (
        "This is ONE numbered slot cropped from the bottom instrument strip "
        "of a da Vinci surgery video. It shows the contents of a single "
        "robot arm (text possibly abbreviated/truncated).\n\n"
        f"If it is a surgical instrument, choose the best-matching name:\n{names}\n\n"
        "Special cases:\n"
        '  - If it is the CAMERA / endoscope arm (shows a scope angle like '
        '"30°" or "0°", a zoom like "1x", or text such as "UNDOCK '
        'BEFORE MOVING TABLE" / "LASER OFF" instead of an instrument name), '
        'respond "camera".\n'
        '  - If the slot has no instrument mounted / is blank, respond "empty".\n'
        '  - If you genuinely cannot tell, respond "unknown".\n\n'
        f"Ignore mode indicators (NOT instruments): {mode_list}.\n\n"
        'Respond with ONLY a JSON string: a canonical instrument name, or '
        '"camera", "empty", or "unknown". Nothing else.'
    )


SPECIAL_TOKENS = {"camera", "empty", "unknown", "none", "null", ""}


def parse_single(out_text, canonical_set, ui_to_canonical):
    s = out_text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        si = str(json.loads(s)).strip().lower()
    except Exception:
        si = s.strip().strip('"').lower()
    if si in ("", "none", "null"):
        return None
    if si in ("camera", "empty", "unknown"):
        return si
    canon = ui_to_canonical.get(si)
    if canon is None and si in canonical_set:
        canon = si
    if canon is None:
        if "camera" in si or "scope" in si or "endoscop" in si:
            return "camera"
        for c in canonical_set:
            if c in si:
                canon = c
                break
    return canon


def strip_crop(img, strip_px):
    return img[img.shape[0] - strip_px:, :]


def strip_diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 255.0
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def mode_window(seq, W):
    """Per-position mode over a centered window of size W (kills blips)."""
    out = [None] * len(seq)
    half = W // 2
    for i in range(len(seq)):
        win = seq[max(0, i - half): i + half + 1]
        c = collections.Counter(x for x in win if x is not None)
        out[i] = c.most_common(1)[0][0] if c else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--queries-file", type=Path,
                    default=REPO_ROOT / "configs/cardiac_whip_vocabulary.json")
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--strip-px", type=int, default=0,
                    help="Fixed strip height in px; 0 = use --strip-frac.")
    ap.add_argument("--strip-frac", type=float, default=0.17,
                    help="Strip height as fraction of frame height (resolution-aware).")
    ap.add_argument("--stride", type=int, default=30,
                    help="OCR every Nth frame (seconds at 1fps).")
    ap.add_argument("--mode-window", type=int, default=3,
                    help="Sample-level mode filter window (in stride samples).")
    ap.add_argument("--refine-thresh", type=float, default=12.0,
                    help="Strip pixel-diff above which a boundary frame is a swap.")
    ap.add_argument("--source-fps", type=float, default=30.0,
                    help="Source video fps used to map src_frame -> seconds. "
                         "For frames extracted at 1 fps real-time, pass 1.")
    args = ap.parse_args()

    out_dir = args.out_dir / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_names, ui_to_canonical, modes = load_vocab(args.queries_file)
    canonical_set = set(canonical_names)
    prompt = build_single_slot_prompt(canonical_names, modes)

    frames = sorted(
        list(args.frames_dir.glob("frame_*.png")) +
        list(args.frames_dir.glob("frame_*.jpg")),
        key=lambda p: src_idx_from_name(p.name) or 0,
    )
    if not frames:
        print(f"No frames in {args.frames_dir}")
        return 1
    N = len(frames)
    print(f"{args.video_id}: {N} frames, stride={args.stride}")

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    print(f"Loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=64 * 28 * 28, max_pixels=512 * 28 * 28)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("Loaded.\n")

    scratch = Path(tempfile.gettempdir()) / f"slots_{uuid.uuid4().hex[:6]}"
    scratch.mkdir(parents=True, exist_ok=True)

    def ocr_region(region):
        sp = scratch / "r.png"
        cv2.imwrite(str(sp), region)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{sp}"},
            {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            from PIL import Image as PILImage
            image_inputs = [PILImage.open(sp).convert("RGB")]
            video_inputs = None
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return parse_single(raw, canonical_set, ui_to_canonical)

    def strip_px_for(h):
        return args.strip_px if args.strip_px > 0 else max(40, int(h * args.strip_frac))

    # ---- 1+2: stride-sampled spatial OCR ----
    sample_idx = list(range(0, N, args.stride))
    if (N - 1) not in sample_idx:
        sample_idx.append(N - 1)
    raw_reads = {s: [] for s in SLOTS}   # per slot, per sample
    t0 = time.time()
    n_calls = 0
    for k, fi in enumerate(sample_idx):
        img = cv2.imread(str(frames[fi]))
        if img is None:
            for s in SLOTS:
                raw_reads[s].append(None)
            continue
        h, w = img.shape[:2]
        fracs = get_slot_fracs(w, h)
        strip = strip_crop(img, strip_px_for(h))
        for slot in SLOTS:
            a, b = fracs[slot]
            region = strip[:, int(w * a): int(w * b)]
            raw_reads[slot].append(ocr_region(region))
            n_calls += 1
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(sample_idx)} samples  {time.time()-t0:.0f}s")

    # ---- 3: mode-filter per slot over stride samples ----
    sm = {s: mode_window(raw_reads[s], args.mode_window) for s in SLOTS}

    # sample-level config sequence ('empty'/'unknown' are dropped from the
    # arm map but 'camera' is kept — it's a real, identifiable arm state).
    def keep(v):
        return v is not None and v not in ("empty", "unknown")
    sample_cfg = []
    for j in range(len(sample_idx)):
        sample_cfg.append({s: sm[s][j] for s in SLOTS if keep(sm[s][j])})

    # ---- 4: segments at sample resolution, then refine boundaries ----
    seg_bounds = [0]  # indices into sample_idx where config changes
    for j in range(1, len(sample_cfg)):
        if sample_cfg[j] != sample_cfg[j - 1]:
            seg_bounds.append(j)
    seg_bounds.append(len(sample_idx))

    # Refine each transition's exact frame via CPU strip diff between the two
    # bracketing samples; the swap is the first frame whose strip jumps.
    def refine_boundary(lo_fi, hi_fi):
        prev = None
        best_fi, best_d = lo_fi, -1.0
        for fi in range(lo_fi, hi_fi + 1):
            img = cv2.imread(str(frames[fi]))
            if img is None:
                continue
            st = strip_crop(img, strip_px_for(img.shape[0]))
            if prev is not None:
                d = strip_diff(prev, st)
                if d > best_d:
                    best_d, best_fi = d, fi
            prev = st
        return best_fi if best_d >= args.refine_thresh else hi_fi

    segments = []
    for b in range(len(seg_bounds) - 1):
        j0 = seg_bounds[b]
        j1 = seg_bounds[b + 1] - 1
        cfg = sample_cfg[j0]
        start_fi = sample_idx[j0]
        if b > 0:  # refine start against previous sample
            start_fi = refine_boundary(sample_idx[seg_bounds[b] - 1],
                                       sample_idx[j0])
        end_fi = sample_idx[j1]
        segments.append({"start": start_fi, "end": end_fi, "cfg": cfg})
    # fix end frames to be one before next start; last ends at N-1
    for b in range(len(segments) - 1):
        segments[b]["end"] = segments[b + 1]["start"] - 1
    if segments:
        segments[-1]["end"] = N - 1

    # merge adjacent identical configs (collapses any false splits)
    merged = []
    for seg in segments:
        if merged and merged[-1]["cfg"] == seg["cfg"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    segments = merged

    # ---- 5: per-frame timeline ----
    timeline = []
    for seg in segments:
        for fi in range(seg["start"], seg["end"] + 1):
            src = src_idx_from_name(frames[fi].name)
            timeline.append({
                "i": fi,
                "src_frame": src,
                "sec": round(src / args.source_fps, 2) if src is not None else None,
                "arms": {str(k): v for k, v in sorted(seg["cfg"].items())},
                "instruments": sorted(set(seg["cfg"].values())),
            })

    elapsed = time.time() - t0

    # ---- write outputs ----
    (out_dir / "timeline.json").write_text(json.dumps({
        "video_id": args.video_id, "n_frames": N, "stride": args.stride,
        "n_ocr_calls": n_calls, "n_segments": len(segments),
        "elapsed_sec": round(elapsed, 1), "frames": timeline,
    }, indent=2))

    with (out_dir / "segments.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_i", "end_i", "n_frames", "start_src", "end_src",
                    "start_sec", "end_sec", "arm1", "arm2", "arm3", "arm4"])
        for seg in segments:
            s_src = src_idx_from_name(frames[seg["start"]].name)
            e_src = src_idx_from_name(frames[seg["end"]].name)
            cfg = seg["cfg"]
            w.writerow([
                seg["start"], seg["end"], seg["end"] - seg["start"] + 1,
                s_src, e_src,
                round(s_src / args.source_fps, 1) if s_src is not None else "",
                round(e_src / args.source_fps, 1) if e_src is not None else "",
                cfg.get(1, ""), cfg.get(2, ""), cfg.get(3, ""), cfg.get(4, ""),
            ])

    distinct = sorted({i for seg in segments for i in seg["cfg"].values()})
    qc = {
        "video_id": args.video_id, "n_frames": N, "n_ocr_calls": n_calls,
        "n_segments": len(segments), "distinct_instruments": distinct,
        "stride": args.stride, "elapsed_sec": round(elapsed, 1),
        "sec_per_sample": round(elapsed / max(len(sample_idx), 1), 3),
    }
    (out_dir / "qc.json").write_text(json.dumps(qc, indent=2))

    print(f"\n{args.video_id} done: {N} frames, {n_calls} OCR calls, "
          f"{len(segments)} segments, {elapsed:.0f}s")
    print(f"  instruments: {distinct}")
    for seg in segments:
        print(f"  [{seg['start']:5d}-{seg['end']:5d}] {seg['cfg']}")
    print(f"  wrote {out_dir}/timeline.json, segments.csv, qc.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
