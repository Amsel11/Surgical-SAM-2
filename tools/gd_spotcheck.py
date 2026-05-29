"""Zero-shot Grounding DINO spot-check: draw GD boxes on chosen frames.

De-risks the "detector + tracker instead of SAM mask propagation" pivot: run
ZS GD on the exact frames where SAM propagation failed and see whether GD
localizes the instruments cleanly (no drift — GD re-detects each frame).

Frame indices are the tracker's LOADER indices (the cutout/mask filename stems),
mapped to real source files via pipeline.io.frame_files_ordered so they line up
with the cutouts you reviewed.

Output: <out-dir>/<video>__f<idx>.jpg with one colored box per detection
(label + score burned in). Runs in .gdino_venv (transformers + torch + PIL).

Usage:
  python -m tools.gd_spotcheck --video DC_whip_11609423 --frames 11512,12330 \
      --frames-root /gpfs/data/oermannlab/private_data/whip/frames_attempt2 \
      --out-dir results/gd_spotcheck
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.io import frame_files_ordered
from pipeline.prompts._grounding_dino_detector import (
    GroundingDinoConfig,
    GroundingDinoDetector,
)

DEFAULT_QUERIES = [
    "surgical instrument",
    "forceps",
    "scissors",
    "grasper",
    "needle driver",
]
# Distinct RGB per query slot.
COLORS = [
    (255, 80, 80), (80, 200, 120), (80, 160, 255), (240, 200, 60),
    (200, 120, 255), (255, 255, 255),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True)
    p.add_argument("--frames", required=True,
                   help="Comma-separated LOADER frame indices (mask/cutout stems).")
    p.add_argument("--frames-root", type=Path,
                   default=Path("/gpfs/data/oermannlab/private_data/whip/frames_attempt2"))
    p.add_argument("--out-dir", type=Path, default=Path("results/gd_spotcheck"))
    p.add_argument("--queries", default=",".join(DEFAULT_QUERIES),
                   help="Comma-separated text queries.")
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    frames_dir = args.frames_root / args.video
    ordered = frame_files_ordered(frames_dir)
    if not ordered:
        print(f"No frames under {frames_dir}", file=sys.stderr)
        return 2

    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    color_for = {q.strip().rstrip(".").lower(): COLORS[i % len(COLORS)]
                 for i, q in enumerate(queries)}

    det = GroundingDinoDetector(GroundingDinoConfig(
        model_id=args.model_id, box_threshold=args.box_threshold,
        text_threshold=args.text_threshold, device=args.device))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    loader_idxs = [int(x) for x in args.frames.split(",") if x.strip()]
    for li in loader_idxs:
        if li >= len(ordered):
            print(f"  loader idx {li} >= {len(ordered)} frames; skip", file=sys.stderr)
            continue
        fpath = frames_dir / ordered[li]
        by_query = det.detect(fpath, queries)
        img = Image.open(fpath).convert("RGB")
        draw = ImageDraw.Draw(img)
        n_boxes = 0
        for q, boxes in by_query.items():
            col = color_for.get(q, (255, 255, 255))
            for b in boxes:
                x0, y0, x1, y1, sc = b
                draw.rectangle([x0, y0, x1, y1], outline=col, width=4)
                draw.text((x0 + 3, max(0, y0 - 12)), f"{q} {sc:.2f}", fill=col)
                n_boxes += 1
        out = args.out_dir / f"{args.video}__f{li:05d}.jpg"
        img.save(out, quality=90)
        print(f"  {args.video} f{li}: {n_boxes} boxes -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
