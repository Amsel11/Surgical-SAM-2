"""One-video identity demo: GD detect -> greedy-IoU track -> identity from
your anchor box + OCR cross-check. Renders the video with CORRECT instrument
names (not GD's wrong text guess).

Pipeline (the detector+tracker pivot, lean version — no ByteTrack dep yet):
  1. Grounding DINO detects boxes every --stride frames (re-detect, no drift).
  2. Filter: drop boxes covering > --area-frac-max of the frame (the generic
     full-frame blob) and below --score-floor.
  3. Greedy-IoU tracker links detections into tracks with stable ids.
  4. Identity seed: at the anchor frame, match each track to your manual anchor
     box (IoU) -> the track inherits that instrument_id for its whole life.
  5. OCR cross-check: the segments.csv active-tool set at each time is shown as
     a banner; identities not in the mounted set get flagged.

Output: <out-dir>/<video>/f<idx>.jpg labeled frames + _contact_sheet.jpg.

Runs in .sam3_venv (transformers + torch + PIL). GPU recommended.

Usage:
  python -m tools.track_identity_demo --video DC_whip_11609423 --stride 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.db import REPO_ROOT, connect
from pipeline.io import frame_files_ordered
from pipeline.prompts._grounding_dino_detector import (
    GroundingDinoConfig, GroundingDinoDetector, iou_xyxy,
)

DEFAULT_QUERIES = [
    "fenestrated bipolar forceps", "vessel sealer", "grasping retractor",
    "monopolar curved scissors", "prograsp forceps",
]
TRACK_COLORS = [
    (255, 80, 80), (80, 200, 120), (80, 160, 255), (240, 200, 60),
    (200, 120, 255), (60, 220, 220), (255, 150, 60),
]


def load_anchors(video_id, seed, db):
    """Return (anchor_frame_idx -> [(box, instrument_id)], resolution)."""
    conn = connect(db)
    ps = conn.execute(
        "SELECT prompt_set_id, prompts_path FROM prompt_sets "
        "WHERE video_id=? AND seed=? AND prompt_method='manual_box'",
        (video_id, seed)).fetchone()
    if ps is None:
        return {}, None
    obj_inst = {int(r["obj_id"]): r["instrument_id"] for r in conn.execute(
        "SELECT obj_id, instrument_id FROM prompt_objects WHERE prompt_set_id=?",
        (ps["prompt_set_id"],))}
    data = json.loads(Path(ps["prompts_path"]).read_text())
    out = {}
    for fk, objs in data["objects_by_frame"].items():
        out[int(fk)] = [(o["box"], obj_inst.get(int(o["obj_id"]), "unknown"))
                        for o in objs]
    return out, data.get("resolution")


def load_ocr(segments_csv):
    """Return fn(loader_idx) -> set of mounted tool names (excl camera/empty)."""
    rows = []
    if Path(segments_csv).exists():
        with open(segments_csv) as f:
            for r in csv.DictReader(f):
                rows.append(r)

    def active(idx):
        for r in rows:
            if int(r["start_i"]) <= idx <= int(r["end_i"]):
                tools = {r[a].strip() for a in ("arm1", "arm2", "arm3", "arm4")
                         if r.get(a) and r[a].strip() and r[a].strip() != "camera"}
                return tools
        return set()
    return active


class GreedyTracker:
    """Frame-to-frame greedy-IoU association. No motion model — fine for small
    inter-frame displacement at modest stride."""
    def __init__(self, iou_thresh=0.3, max_age=5):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}   # tid -> {"box":, "age":, "instrument":}
        self._next = 1

    def update(self, boxes):
        """boxes: list[[x0,y0,x1,y1,score]]. Returns list of (tid, box)."""
        assigned = {}
        pairs = []
        for tid, t in self.tracks.items():
            for di, b in enumerate(boxes):
                i = iou_xyxy(t["box"], b[:4])
                if i >= self.iou_thresh:
                    pairs.append((i, tid, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _i, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid); used_d.add(di)
            self.tracks[tid]["box"] = boxes[di][:4]
            self.tracks[tid]["age"] = 0
            assigned[di] = tid
        created = set()
        for di, b in enumerate(boxes):
            if di in used_d:
                continue
            tid = self._next; self._next += 1
            self.tracks[tid] = {"box": b[:4], "age": 0, "instrument": None}
            assigned[di] = tid
            created.add(tid)
        for tid in list(self.tracks):
            if tid in used_t or tid in created:
                continue
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]
        return [(assigned[di], boxes[di]) for di in range(len(boxes))]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--frames-root", type=Path,
                   default=Path("/gpfs/data/oermannlab/private_data/whip/frames_attempt2"))
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "track_identity_demo")
    p.add_argument("--ocr-root", type=Path,
                   default=REPO_ROOT / "results" / "ocr_slots_timeline_paddle_full")
    p.add_argument("--queries", default=",".join(DEFAULT_QUERIES))
    p.add_argument("--box-threshold", type=float, default=0.28)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--score-floor", type=float, default=0.28)
    p.add_argument("--area-frac-max", type=float, default=0.5)
    p.add_argument("--iou-track", type=float, default=0.3)
    p.add_argument("--max-age", type=int, default=6)
    p.add_argument("--n-render", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--db")
    args = p.parse_args(argv)

    frames_dir = args.frames_root / args.video
    ordered = frame_files_ordered(frames_dir)
    if not ordered:
        print(f"No frames in {frames_dir}", file=sys.stderr); return 2
    anchors, _res = load_anchors(args.video, args.seed, args.db)
    ocr_active = load_ocr(args.ocr_root / args.video / "segments.csv")
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]

    det = GroundingDinoDetector(GroundingDinoConfig(
        model_id="IDEA-Research/grounding-dino-tiny",
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        device=args.device))
    tracker = GreedyTracker(args.iou_track, args.max_age)

    # Process strided frames + force-include anchor frames so identity can seed.
    idxs = sorted(set(range(0, len(ordered), args.stride)) | set(anchors.keys()))
    frame_dets = {}   # idx -> list[(tid, box)]
    for idx in idxs:
        fpath = frames_dir / ordered[idx]
        by_q = det.detect(fpath, queries)
        W = H = None
        with Image.open(fpath) as im:
            W, H = im.size
        boxes = []
        for q, bs in by_q.items():
            for b in bs:
                x0, y0, x1, y1, sc = b
                if sc < args.score_floor:
                    continue
                if ((x1 - x0) * (y1 - y0)) > args.area_frac_max * W * H:
                    continue  # drop the full-frame blob
                boxes.append([x0, y0, x1, y1, sc])
        tracked = tracker.update(boxes)
        frame_dets[idx] = tracked
        # Identity seed at anchor frames.
        if idx in anchors:
            for abox, inst in anchors[idx]:
                best_tid, best_iou = None, 0.0
                for tid, b in tracked:
                    i = iou_xyxy(abox, b[:4])
                    if i > best_iou:
                        best_iou, best_tid = i, tid
                if best_tid is not None and best_iou >= 0.2:
                    tracker.tracks.setdefault(best_tid, {})["instrument"] = inst
                    tracker._seeded = getattr(tracker, "_seeded", {})
                    tracker._seeded[best_tid] = inst

    seeded = getattr(tracker, "_seeded", {})
    # Render evenly-sampled frames with identity labels.
    out_dir = args.out_dir / args.video
    out_dir.mkdir(parents=True, exist_ok=True)
    render_idxs = idxs if len(idxs) <= args.n_render else [
        idxs[int(i * len(idxs) / args.n_render)] for i in range(args.n_render)]
    thumbs = []
    for idx in render_idxs:
        fpath = frames_dir / ordered[idx]
        img = Image.open(fpath).convert("RGB")
        draw = ImageDraw.Draw(img)
        active = ocr_active(idx)
        for tid, b in frame_dets.get(idx, []):
            inst = seeded.get(tid)
            col = TRACK_COLORS[tid % len(TRACK_COLORS)]
            x0, y0, x1, y1 = [int(v) for v in b[:4]]
            draw.rectangle([x0, y0, x1, y1], outline=col, width=4)
            flag = "" if (inst is None or not active or inst_in(inst, active)) else " [!OCR]"
            lbl = f"T{tid}:{inst or '?'} {b[4]:.2f}{flag}"
            draw.text((x0 + 3, max(0, y0 - 12)), lbl, fill=col)
        draw.text((6, 6), f"f{idx}  OCR mounted: {', '.join(sorted(active)) or '(none)'}",
                  fill=(255, 255, 0))
        op = out_dir / f"f{idx:05d}.jpg"
        img.save(op, quality=88)
        thumbs.append(np.array(img.resize((480, int(480 * img.height / img.width)))))
    if thumbs:
        wmax = max(t.shape[1] for t in thumbs)
        padded = [np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0))) for t in thumbs]
        Image.fromarray(np.concatenate(padded, axis=0)).save(out_dir / "_contact_sheet.jpg", quality=85)
    n_tracks = len(seeded)
    print(f"{args.video}: processed {len(idxs)} frames, {tracker._next-1} tracks, "
          f"{n_tracks} identity-seeded -> {out_dir}", file=sys.stderr)
    for tid, inst in sorted(seeded.items()):
        print(f"    T{tid} = {inst}", file=sys.stderr)
    return 0


def inst_in(instrument_id, ocr_tools):
    """Loose membership: instrument_id (snake) vs OCR tool phrases."""
    key = instrument_id.replace("_", " ")
    toks = set(key.split())
    for t in ocr_tools:
        tt = set(t.lower().split())
        if len(toks & tt) >= 2 or key in t or t in key:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
