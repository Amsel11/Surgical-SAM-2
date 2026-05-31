"""Thin CLI wrapper around SAM2VideoTracker for ad-hoc / single-video runs.

Production / mass runs go through the orchestrator:
    python -m pipeline run +experiment=<name>

This script remains for:
  - One-off smoke tests with a single click/box from the command line
  - Backward compat with the existing slurm/run_inference_array.sh
  - Debugging without setting up Hydra configs

It accepts either:
  --video <mp4_or_frames_dir>  --prompts-json <path>        (clicker output)
  --video <...>                --object 1:x,y --object 2:x,y
  --video <...>                --point x,y [--neg-point x,y]
  --video <...>                --box x1,y1,x2,y2
"""
from __future__ import annotations

import os
# On Apple-silicon (mps) a few SAM2 ops aren't implemented; fall back to CPU
# for those instead of crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json as _json
import shutil
import tempfile
from pathlib import Path

import cv2

from pipeline.config import Stage2Inference, Stage2Outputs
from pipeline.models.sam2 import SAM2VideoTracker


# ---------------------------------------------------------------------------
# CLI parsers
# ---------------------------------------------------------------------------

def parse_point(s):
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"point must be 'x,y', got {s!r}")
    return [float(parts[0]), float(parts[1])]


def parse_box(s):
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"box must be 'x1,y1,x2,y2', got {s!r}")
    return [float(p) for p in parts]


def parse_object(s):
    """Parse '--object OBJID:x,y[,x,y...]'. Prefix '!' for a negative click."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"--object must be 'OBJID:x,y[,x,y...]', got {s!r}")
    oid_str, rest = s.split(":", 1)
    oid = int(oid_str)
    if oid <= 0:
        raise argparse.ArgumentTypeError(f"--object OBJID must be > 0, got {oid}")
    toks = [t.strip() for t in rest.split(",")]
    if len(toks) % 2 != 0:
        raise argparse.ArgumentTypeError(f"--object {oid}: coords must come in x,y pairs")
    pts, labels = [], []
    for i in range(0, len(toks), 2):
        x_tok = toks[i]
        neg = x_tok.startswith("!")
        if neg:
            x_tok = x_tok[1:]
        pts.append([float(x_tok), float(toks[i + 1])])
        labels.append(0 if neg else 1)
    return {"obj_id": oid, "points": pts, "labels": labels}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_frames_to_dir(video_path: str, out_dir: Path) -> float:
    """Decode an mp4 to JPEG frames 00000.jpg, 00001.jpg, ... Returns source fps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1
    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return float(fps)


def _build_ad_hoc_prompts_json(args, frame_count: int, tmp_dir: Path) -> Path:
    """When the user provides --object/--point/--box (not --prompts-json),
    materialize a tiny prompts JSON so the tracker can consume it uniformly."""
    objs_by_frame: dict[str, list[dict]] = {}
    fidx = args.prompt_frame
    objects_to_add: list[dict] = []

    if args.objects:
        for spec in args.objects:
            pos = [p for p, lab in zip(spec["points"], spec["labels"]) if lab == 1]
            neg = [p for p, lab in zip(spec["points"], spec["labels"]) if lab == 0]
            objects_to_add.append({
                "obj_id": spec["obj_id"], "positive": pos, "negative": neg,
            })
    else:
        objects_to_add.append({
            "obj_id": args.obj_id,
            "positive": args.point,
            "negative": args.neg_point,
            **({"box": args.box} if args.box else {}),
        })

    objs_by_frame[str(fidx)] = objects_to_add
    out_path = tmp_dir / "_adhoc_prompts.json"
    out_path.write_text(_json.dumps({
        "video": "adhoc",
        "n_frames": frame_count,
        "prompt_frames": [fidx],
        "objects_by_frame": objs_by_frame,
    }, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help=".mp4 OR a directory of frames")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default="./checkpoints/sam2.1_hiera_s_endo18.pth")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--point", type=parse_point, action="append", default=[])
    ap.add_argument("--neg-point", type=parse_point, action="append", default=[])
    ap.add_argument("--box", type=parse_box, default=None)
    ap.add_argument("--object", dest="objects", type=parse_object, action="append", default=[])
    ap.add_argument("--prompts-json", type=str, default=None)
    ap.add_argument("--prompt-frame", type=int, default=0)
    ap.add_argument("--obj-id", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--no-overlay-video", action="store_true")
    ap.add_argument("--no-overlay-jpgs", action="store_true")
    ap.add_argument("--keep-extracted-frames", action="store_true")
    ap.add_argument("--no-bidirectional", action="store_true",
                    help="Forward-only propagation from min(prompts). Default is bidirectional.")
    args = ap.parse_args()

    if not args.prompts_json and not args.objects and not args.point and not args.box:
        ap.error("Provide --prompts-json, --object, --point, or --box")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare frames dir (extract mp4 if needed) ─────────────────────────
    cleanup_frames = False
    if Path(args.video).is_dir():
        frames_dir = args.video
        src_fps = args.fps or 30.0
    else:
        if args.keep_extracted_frames:
            frames_dir = str(out_dir / "frames")
        else:
            frames_dir = tempfile.mkdtemp(prefix="surgsam2_frames_")
            cleanup_frames = True
        src_fps = extract_frames_to_dir(args.video, Path(frames_dir))
        if args.fps:
            src_fps = args.fps

    # ── Materialize ad-hoc prompts JSON if needed ─────────────────────────
    if args.prompts_json:
        prompts_json = Path(args.prompts_json)
    else:
        n_in_dir = sum(1 for p in Path(frames_dir).iterdir() if p.is_file())
        prompts_json = _build_ad_hoc_prompts_json(args, n_in_dir, out_dir)

    # ── Build tracker config and run ──────────────────────────────────────
    cfg = Stage2Inference(
        model="sam2",                   # tracker class is SAM2VideoTracker regardless
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
        bidirectional=not args.no_bidirectional,
        outputs=Stage2Outputs(
            masks=True,
            overlay_video=not args.no_overlay_video,
            overlay_jpgs=not args.no_overlay_jpgs,
            preview_small=False,        # the slurm wrapper handles preview encoding
        ),
    )
    tracker = SAM2VideoTracker(cfg)

    try:
        log = tracker.run(
            video_id=Path(args.video).stem,
            frames_dir=Path(frames_dir),
            prompts_json=prompts_json,
            results_dir=out_dir,
            src_fps=src_fps,
        )
        print(f"Log     -> {out_dir / 'log.json'}")
        print(f"Masks   -> {out_dir / 'masks'}")
        if cfg.outputs.overlay_video:
            print(f"Overlay video -> {out_dir / 'overlay.mp4'}")
    finally:
        if cleanup_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
