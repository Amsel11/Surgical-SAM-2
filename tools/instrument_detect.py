"""Robust single-frame instrument detector: GDINO query-ensemble + geometric priors.

Stock Grounding DINO is weak on da Vinci endoscopic video (catalog names return
nothing; generic terms fire at low confidence with tissue false-positives). This
recovers usable detections with priors that hold for endoscopy:

  - Instruments enter through trocar ports => their box reaches a frame EDGE.
  - Tissue / wound false-positives are central and large => reject by area cap.
  - The da Vinci UI bar sits in the bottom strip => exclude it.
  - Different query phrasings each catch different instruments => union + NMS.
  - Low-confidence boxes are the ones that leak when fed to SAM2 => --min-score
    gate drops them (precision over recall — a missed instrument beats a leak).

Importable API (used by tools/auto_mask.py):
    det = build_detector(model, device, box_th, text_th)
    kept, dropped, (W, H) = detect_instruments(det, frame_path, min_score=0.16, ...)

CLI writes a self-describing, numbered experiment dir:
    <results-root>/exp_NNN_<name>/{params,boxes}.json + annotated.jpg
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.prompts._grounding_dino_detector import (
    GroundingDinoConfig,
    GroundingDinoDetector,
    iou_xyxy,
)

DEFAULT_QUERIES = [
    "tweezers", "pliers", "scissors", "clamp", "forceps",
    "robotic surgical instrument", "metallic instrument",
    "needle holder", "grasper", "surgical tool", "metal rod",
]


def build_detector(model="IDEA-Research/grounding-dino-base", device="mps",
                   box_th=0.12, text_th=0.12) -> GroundingDinoDetector:
    """Build one GDINO detector. Heavy (~700 MB) — reuse across many frames."""
    return GroundingDinoDetector(GroundingDinoConfig(
        device=device, model_id=model, box_threshold=box_th, text_threshold=text_th))


def content_bbox(img, thresh=14):
    """Bounding box of the non-black surgical content — strips pillar/letterbox
    bars so the edge prior keys off the *image* edge, not the raw frame edge.
    Some clips (e.g. 1920x1080 cases) center a square image between black bars;
    instruments enter at the content edge, which the frame edge misses."""
    arr = np.asarray(img.convert("RGB"))
    bright = arr.max(axis=2) > thresh
    cols = np.where(bright.any(axis=0))[0]
    rows = np.where(bright.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, img.size[0], img.size[1])
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def passes_priors(box, content, edge_frac, area_cap, ui_bottom):
    x0, y0, x1, y1 = box[:4]
    cx0, cy0, cx1, cy1 = content
    cw, ch = max(1, cx1 - cx0), max(1, cy1 - cy0)
    area = ((x1 - x0) * (y1 - y0)) / (cw * ch)
    if area > area_cap:
        return False, f"area {area:.2f}>cap"
    if y0 > cy0 + ui_bottom * ch:
        return False, "in UI bar"
    near_edge = (
        x0 <= cx0 + edge_frac * cw or y0 <= cy0 + edge_frac * ch
        or x1 >= cx1 - edge_frac * cw or y1 >= cy1 - edge_frac * ch
    )
    if not near_edge:
        return False, "not edge-anchored"
    return True, "ok"


def detect_instruments(detector, frame_path, *, queries=None, min_score=0.16,
                       edge_frac=0.06, area_cap=0.18, ui_bottom=0.93, nms_iou=0.55):
    """Return (kept, dropped, (W, H)).

    kept: [{"query", "score", "box":[x0,y0,x1,y1]}] sorted by score desc, after
    confidence gate + geometric priors + greedy NMS.
    """
    queries = queries or DEFAULT_QUERIES
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    content = content_bbox(img)
    res = detector.detect(frame_path, queries)

    cand = sorted(
        ((sc, x0, y0, x1, y1, q)
         for q, boxes in res.items() for (x0, y0, x1, y1, sc) in boxes),
        reverse=True,
    )

    kept, dropped = [], []
    for sc, x0, y0, x1, y1, q in cand:
        if sc < min_score:
            dropped.append({"query": q, "score": round(sc, 3), "box": [x0, y0, x1, y1],
                            "reason": f"score<{min_score}"})
            continue
        ok, why = passes_priors((x0, y0, x1, y1), content, edge_frac, area_cap, ui_bottom)
        if not ok:
            dropped.append({"query": q, "score": round(sc, 3), "box": [x0, y0, x1, y1], "reason": why})
            continue
        if any(iou_xyxy([x0, y0, x1, y1], k["box"]) > nms_iou for k in kept):
            dropped.append({"query": q, "score": round(sc, 3), "box": [x0, y0, x1, y1], "reason": "nms"})
            continue
        kept.append({"query": q, "score": round(sc, 3),
                     "box": [round(x0), round(y0), round(x1), round(y1)]})
    return kept, dropped, (W, H)


def next_exp_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("_")[1]) for p in root.glob("exp_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()]
    d = root / f"exp_{(max(nums) + 1) if nums else 1:03d}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--name", default="run")
    ap.add_argument("--results-root", default="/Users/schula12/whipple/masking_test/experiments")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    ap.add_argument("--box-th", type=float, default=0.12)
    ap.add_argument("--text-th", type=float, default=0.12)
    ap.add_argument("--min-score", type=float, default=0.16, help="drop boxes below this (precision gate)")
    ap.add_argument("--edge-frac", type=float, default=0.06)
    ap.add_argument("--area-cap", type=float, default=0.18)
    ap.add_argument("--ui-bottom", type=float, default=0.93)
    ap.add_argument("--nms-iou", type=float, default=0.55)
    args = ap.parse_args()

    exp = next_exp_dir(Path(args.results_root), args.name)
    det = build_detector(args.model, args.device, args.box_th, args.text_th)
    kept, dropped, (W, H) = detect_instruments(
        det, args.frame, queries=args.queries, min_score=args.min_score,
        edge_frac=args.edge_frac, area_cap=args.area_cap,
        ui_bottom=args.ui_bottom, nms_iou=args.nms_iou)

    img = Image.open(args.frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    for k in kept:
        x0, y0, x1, y1 = k["box"]
        draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=5)
        draw.text((x0 + 3, y0 + 3), f"{k['query'][:12]} {k['score']:.2f}", fill=(0, 255, 0))
    img.save(exp / "annotated.jpg")
    (exp / "boxes.json").write_text(json.dumps({"kept": kept, "dropped": dropped}, indent=2))
    (exp / "params.json").write_text(json.dumps({
        "frame": str(args.frame), "model": args.model, "device": args.device,
        "queries": args.queries, "box_th": args.box_th, "text_th": args.text_th,
        "min_score": args.min_score, "edge_frac": args.edge_frac, "area_cap": args.area_cap,
        "ui_bottom": args.ui_bottom, "nms_iou": args.nms_iou, "n_kept": len(kept),
    }, indent=2))

    print(f"-> {len(kept)} kept (dropped {len(dropped)})")
    for k in kept:
        print(f"  KEEP {k['query']:26s} {k['score']:.2f} {k['box']}")
    print(f"saved -> {exp}")


if __name__ == "__main__":
    main()
