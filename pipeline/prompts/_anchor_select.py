"""Anchor-frame box selection for the automated (`auto`) stage-1 prompter.

Pulled out of the now-archived tools/anchor_and_propagate.py so the logic
lives in the package and is unit-testable. Two responsibilities:

  1. Read an OCR `segments.csv` (the da Vinci arm-slot timeline) into a list
     of segments, each carrying loader-space (`start_i`) AND filename-space
     (`start_src`) indices — the bridge the prompter needs to emit pseudo
     source-frame keys the SAM2 tracker can resolve.
  2. Turn one frame's Grounding DINO detections (the `detect()` output,
     `{query: [[x0,y0,x1,y1,score], ...]}`) into a clean, deduplicated,
     left-to-right-ordered list of candidate boxes — class-agnostic, because
     GD's per-instrument labels are unreliable (the Qwen validator showed 87%
     can't disambiguate) while its localization is decent.

obj_id assignment across anchors is the prompter's job (persistent IoU
matching), not this module's.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pipeline.prompts._grounding_dino_detector import iou_xyxy

# ---------------------------------------------------------------------------
# segments.csv loading
# ---------------------------------------------------------------------------


def _arm_value(raw: str | None) -> str | None:
    """Normalize a segments.csv arm cell. 'camera' / empty -> None."""
    if raw is None:
        return None
    s = raw.strip()
    if not s or s.lower() == "camera":
        return None
    return s.lower()


def load_segments(csv_path: Path) -> list[dict[str, Any]]:
    """Parse an OCR slot timeline into segment dicts.

    Each row carries both index spaces:
      - start_i / end_i   : loader-space (0..N-1, contiguous)
      - start_src/end_src : filename-space (frame_<src>.png, may have gaps)
    """
    rows: list[dict[str, Any]] = []
    with open(csv_path) as f:
        for raw in csv.DictReader(f):
            arms = {a: _arm_value(raw.get(f"arm{a}")) for a in (1, 2, 3, 4)}
            rows.append({
                "start_i":   int(raw["start_i"]),
                "end_i":     int(raw["end_i"]),
                "start_src": int(raw["start_src"]),
                "end_src":   int(raw["end_src"]),
                "n_frames":  int(raw["n_frames"]),
                "arms":      arms,
            })
    return rows


# ---------------------------------------------------------------------------
# Box filters
# ---------------------------------------------------------------------------


def _box_in_range(box, w, h, score_floor, min_af, max_af) -> bool:
    """Score above floor AND area-fraction within [min_af, max_af]."""
    if box[4] < score_floor:
        return False
    af = max(0.0, (box[2] - box[0]) * (box[3] - box[1])) / max(1.0, w * h)
    return min_af <= af <= max_af


def _box_in_bottom_strip(box, h, exclude_bottom_frac) -> bool:
    """True if the box lies ENTIRELY in the bottom `frac` of the frame — used
    to drop the da Vinci OCR strip. Checks the TOP edge (y_min) so a tall
    instrument extending down into the strip zone is NOT dropped."""
    if exclude_bottom_frac <= 0:
        return False
    return box[1] > h * (1.0 - exclude_bottom_frac)


def _shrink_box(box, frac):
    """Pull each edge in by frac/2 so the box ends up `1-frac` of its size in
    each dimension (frac=0.10 -> 90% w+h, 81% area). Tighter prompt -> tighter
    SAM2 mask."""
    if frac <= 0:
        return box
    x0, y0, x1, y1 = box[:4]
    dw = (x1 - x0) * frac / 2.0
    dh = (y1 - y0) * frac / 2.0
    return [x0 + dw, y0 + dh, x1 - dw, y1 - dh]


# ---------------------------------------------------------------------------
# Non-max suppression + class-agnostic selection
# ---------------------------------------------------------------------------


def _nms(boxes, iou_thr=0.5):
    """Greedy NMS on score (box[4]); drops boxes overlapping a kept one."""
    if not boxes:
        return []
    sb = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[list[float]] = []
    for b in sb:
        if all(iou_xyxy(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


def select_boxes_class_agnostic(
    by_query: dict[str, list[list[float]]],
    w: int,
    h: int,
    score_floor: float,
    min_af: float,
    max_af: float,
    nms_iou: float = 0.5,
    exclude_bottom_frac: float = 0.0,
) -> list[list[float]]:
    """Merge GD boxes across all queries, filter, NMS, sort left-to-right.

    `by_query` is the raw `GroundingDinoDetector.detect()` output:
    `{query: [[x0,y0,x1,y1,score], ...]}`. With the generic "surgical
    instrument" query there is only one query, but this stays query-agnostic
    so prompt-ensemble queries also work.

    Returns unassigned boxes `[[x0,y0,x1,y1], ...]`, left-to-right by center-x.
    Falls back to a relaxed area cap (0.70) only if the strict pass is empty.
    """
    def _pool(max_area: float) -> list[list[float]]:
        out: list[list[float]] = []
        for _q, boxes in by_query.items():
            for b in boxes:
                if _box_in_bottom_strip(b, h, exclude_bottom_frac):
                    continue
                if _box_in_range(b, w, h, score_floor, min_af, max_area):
                    out.append(b)
        return out

    pool = _pool(max_af)
    if not pool:
        pool = _pool(0.70)
    if not pool:
        return []
    kept = _nms(pool, nms_iou)
    kept_sorted = sorted(kept, key=lambda b: 0.5 * (b[0] + b[2]))
    return [b[:4] for b in kept_sorted]


__all__ = [
    "load_segments",
    "select_boxes_class_agnostic",
    "_box_in_range",
    "_box_in_bottom_strip",
    "_shrink_box",
    "_nms",
]
