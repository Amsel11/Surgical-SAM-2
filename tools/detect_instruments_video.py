"""Stage-2 detection: generic-query Grounding DINO over a video, sampled at a
fixed time interval, with fully-overlapping boxes consolidated (NMS).

Decoupled from OCR + tracking on purpose, so we can ablate the anchor frequency
(`--every-sec`) and measure self-consistency WITHOUT ground truth: run at a few
densities and compare the persistent-instrument count / box agreement.

The detector is swappable: `--gd-checkpoint <path>` overrides the stock HF model
id with finetuned weights (transformers routes the load) — same hook the
prompter uses, no code change. Reuses the package's GroundingDinoDetector and
the class-agnostic NMS/area/strip selection so this is the same detection path,
not a parallel one.

Streams the video (decode is sequential — reliable across codecs); only the
boxes are kept, never full frames. Writes detections.json + a summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.prompts._anchor_select import select_boxes_class_agnostic
from pipeline.prompts._grounding_dino_detector import (
    GroundingDinoConfig,
    GroundingDinoDetector,
    iou_xyxy,
)

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg")


def count_persistent_tracks(per_frame_boxes, match_iou=0.3, max_gap=3):
    """Cross-frame IoU matching to estimate how many DISTINCT instruments the
    detector found over time (a GT-free 'how stable is the set' signal). A track
    persists across short gaps; a box that matches no live track starts a new one.
    Returns (n_tracks, track_lengths)."""
    tracks = []  # each: {"box": last_box, "last_i": idx, "len": n}
    for i, boxes in enumerate(per_frame_boxes):
        claimed = set()
        for box in boxes:
            best, best_iou = None, 0.0
            for ti, tr in enumerate(tracks):
                if ti in claimed or (i - tr["last_i"]) > max_gap:
                    continue
                iou = iou_xyxy(box, tr["box"])
                if iou > best_iou:
                    best, best_iou = ti, iou
            if best is not None and best_iou >= match_iou:
                tracks[best].update(box=box, last_i=i, len=tracks[best]["len"] + 1)
                claimed.add(best)
            else:
                tracks.append({"box": box, "last_i": i, "len": 1})
                claimed.add(len(tracks) - 1)
    return len(tracks), sorted((t["len"] for t in tracks), reverse=True)


def detect_video(video_path: Path, detector, args, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native * args.every_sec))

    scratch = Path(tempfile.mkdtemp(prefix="gddet_"))
    tmp = scratch / "f.png"

    frames_out, per_frame_boxes = [], []
    t0 = time.time()
    idx = n_samples = 0
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if not w:
                h, w = frame.shape[:2]
            cv2.imwrite(str(tmp), frame)
            by_query = detector.detect(tmp, [args.query])
            boxes = select_boxes_class_agnostic(
                by_query, w, h,
                score_floor=args.score_floor,
                min_af=args.min_area_frac, max_af=args.max_area_frac,
                nms_iou=args.nms_iou, exclude_bottom_frac=args.exclude_bottom_frac)
            frames_out.append({
                "src_idx": idx, "sec": round(idx / native, 2),
                "n_boxes": len(boxes), "boxes": [[round(v, 1) for v in b] for b in boxes],
            })
            per_frame_boxes.append(boxes)
            n_samples += 1
            if n_samples % 25 == 0:
                print(f"  {n_samples} samples ({idx/native:.0f}s)  {time.time()-t0:.0f}s")
        idx += 1
    cap.release()

    n_tracks, track_lengths = count_persistent_tracks(per_frame_boxes, args.match_iou)
    box_counts = [f["n_boxes"] for f in frames_out]
    summary = {
        "video": video_path.stem,
        "native_fps": round(native, 2),
        "every_sec": args.every_sec,
        "n_samples": n_samples,
        "duration_sec": round(idx / native, 1),
        "mean_boxes_per_frame": round(sum(box_counts) / max(len(box_counts), 1), 2),
        "max_boxes_per_frame": max(box_counts, default=0),
        "n_persistent_tracks": n_tracks,
        "persistent_track_lengths": track_lengths,
        "detector": args.gd_checkpoint or args.gd_model_id,
        "query": args.query,
        "resolution": [w, h],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "detections.json").write_text(json.dumps({
        "summary": summary, "frames": frames_out,
    }, indent=2))
    print(f"\n{video_path.stem}: {n_samples} samples @ every {args.every_sec}s, "
          f"~{n_tracks} persistent instruments, "
          f"mean {summary['mean_boxes_per_frame']} boxes/frame, "
          f"{summary['elapsed_sec']}s")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--videos-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--every-sec", type=float, default=2.0,
                    help="Sample a frame for detection every N seconds (the "
                         "anchor-frequency ablation knob).")
    ap.add_argument("--query", default="surgical instrument")
    ap.add_argument("--gd-model-id", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--gd-checkpoint", default=None,
                    help="Finetuned GD checkpoint/path; overrides --gd-model-id.")
    ap.add_argument("--box-thr", type=float, default=0.15)
    ap.add_argument("--text-thr", type=float, default=0.10)
    ap.add_argument("--score-floor", type=float, default=0.15)
    ap.add_argument("--min-area-frac", type=float, default=0.02)
    ap.add_argument("--max-area-frac", type=float, default=0.50)
    ap.add_argument("--exclude-bottom-frac", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.30)
    ap.add_argument("--match-iou", type=float, default=0.30,
                    help="Cross-frame IoU for the persistent-track count.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    if (args.video is None) == (args.videos_dir is None):
        ap.error("provide exactly one of --video / --videos-dir")

    detector = GroundingDinoDetector(GroundingDinoConfig(
        model_id=args.gd_checkpoint or args.gd_model_id,
        box_threshold=args.box_thr, text_threshold=args.text_thr,
        device=args.device, cache_dir=args.cache_dir))

    if args.videos_dir:
        vids = sorted(p for p in args.videos_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in _VIDEO_EXTS)
    else:
        vids = [args.video]
    if not vids:
        print("No videos found.", file=sys.stderr)
        return 1

    summaries = []
    for v in vids:
        try:
            summaries.append(detect_video(v, detector, args, args.out_dir / v.stem))
        except Exception as e:
            print(f"FAILED {v.stem}: {e}", file=sys.stderr)

    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\n=== detection summary ({len(summaries)} videos) ===")
    print(f"{'video':40} {'samples':>8} {'instruments':>12} {'mean_box':>9}")
    for s in summaries:
        print(f"{s['video']:40} {s['n_samples']:>8} {s['n_persistent_tracks']:>12} "
              f"{s['mean_boxes_per_frame']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
