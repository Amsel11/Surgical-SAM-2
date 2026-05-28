"""Lifespan-aware obj_id deduplication on saved mask PNGs.

SAM2's bidirectional propagation can leave two artifacts:
  - ghost tracks: an obj_id that propagates as a few stray pixels.
  - duplicate tracks: two obj_ids that track the same physical instrument and
    "dance" around each other (heavy bbox overlap across many frames).

`dedup_masks` drops ghosts (mean area when present below a floor) and merges
duplicates, but only when two tracks ALSO overlap in time — a late-arriving
instrument that happens to be spatially near an earlier one keeps its own id.
Lifespan is taken from the prompts JSON anchor times when available, because
bidirectional propagation otherwise smears every obj_id across the whole video
and makes mask-presence useless for telling tracks apart.

Survivors are renumbered largest-area = 1. Reads/writes palette PNGs with PIL
(cv2 would decode the palette to RGB and lose the obj_id indices).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_palette_mask(path: Path) -> np.ndarray:
    """Palette mask -> obj_id map (2D uint8). PIL preserves raw palette indices."""
    return np.array(Image.open(path))


def save_palette_mask(combined: np.ndarray, ref_path: Path, out_path: Path) -> None:
    """Re-save with the reference mask's palette."""
    src = Image.open(ref_path)
    img = Image.fromarray(combined.astype(np.uint8), mode="P")
    if src.mode == "P":
        img.putpalette(src.getpalette())
    img.save(out_path)


def dedup_masks(
    in_masks_dir: Path,
    out_masks_dir: Path,
    prompts_json: Path | None = None,
    *,
    iou_thr: float = 0.30,
    min_lifespan_overlap: float = 0.60,
    min_mean_area_frac: float = 0.005,
    min_frames: int = 100,
) -> dict[str, Any]:
    """Drop ghosts, merge duplicate tracks, renumber survivors. Writes cleaned
    palette PNGs to `out_masks_dir` and a `dedup_stats.json` alongside it.
    Returns the stats dict."""
    in_masks_dir = Path(in_masks_dir)
    out_masks_dir = Path(out_masks_dir)
    if out_masks_dir.exists():
        shutil.rmtree(out_masks_dir)
    out_masks_dir.mkdir(parents=True)

    mask_files = sorted(in_masks_dir.glob("*.png"))
    if not mask_files:
        raise RuntimeError(f"No masks in {in_masks_dir}")
    print(f"[dedup] {len(mask_files)} mask files")

    # Pass 1: per-obj presence + area.
    obj_ids: set[int] = set()
    area_sum: dict[int, int] = {}
    presence: dict[int, list[int]] = {}
    for frame_idx, p in enumerate(mask_files):
        m = load_palette_mask(p)
        for v in np.unique(m):
            oid = int(v)
            if oid == 0:
                continue
            obj_ids.add(oid)
            area_sum[oid] = area_sum.get(oid, 0) + int((m == oid).sum())
            presence.setdefault(oid, []).append(frame_idx)

    frame_area = mask_files and (lambda r: r.shape[0] * r.shape[1])(load_palette_mask(mask_files[0]))

    # Lifespan: from prompts-JSON anchor times when given, else mask presence.
    lifespan: dict[int, tuple[int, int]] = {}
    if prompts_json and Path(prompts_json).exists():
        pj = json.load(open(prompts_json))
        anchors_per_obj: dict[int, list[int]] = {}
        for fr_str, objs in pj.get("objects_by_frame", {}).items():
            for o in objs:
                anchors_per_obj.setdefault(int(o["obj_id"]), []).append(int(fr_str))
        for oid in obj_ids:
            anchors = sorted(anchors_per_obj.get(oid, []))
            lifespan[oid] = (anchors[0], anchors[-1]) if anchors else (presence[oid][0], presence[oid][-1])
        print("[dedup] lifespans from prompts JSON (anchor times)")
    else:
        lifespan = {oid: (presence[oid][0], presence[oid][-1]) for oid in obj_ids}
        print("[dedup] lifespans from mask presence (may be inflated by reverse-prop)")

    # Drop ghost tracks (mean area over PRESENT frames below floor).
    ghost_ids: set[int] = set()
    if frame_area and min_mean_area_frac > 0:
        ghost_ids = {
            oid for oid in obj_ids
            if (area_sum[oid] / max(1, len(presence[oid]))) / frame_area < min_mean_area_frac
        }
        if ghost_ids:
            print(f"[dedup] dropping ghosts (mean-when-present < {min_mean_area_frac*100:.2f}%): {sorted(ghost_ids)}")
            obj_ids -= ghost_ids

    # Pass 2: pairwise bbox-IoU over co-present frames.
    bbox_overlap_sum: dict[tuple[int, int], float] = {}
    bbox_overlap_count: dict[tuple[int, int], int] = {}
    for p in mask_files:
        m = load_palette_mask(p)
        boxes = {}
        for oid in obj_ids:
            mm = (m == oid)
            if not mm.any():
                continue
            ys, xs = np.where(mm)
            boxes[oid] = (xs.min(), ys.min(), xs.max(), ys.max())
        oids_here = sorted(boxes.keys())
        for i, a in enumerate(oids_here):
            for b in oids_here[i+1:]:
                ax0, ay0, ax1, ay1 = boxes[a]
                bx0, by0, bx1, by1 = boxes[b]
                ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
                ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
                bb = max(0, bx1 - bx0) * max(0, by1 - by0)
                iou = inter / max(1, aa + bb - inter)
                pair = (a, b)
                bbox_overlap_sum[pair] = bbox_overlap_sum.get(pair, 0.0) + iou
                bbox_overlap_count[pair] = bbox_overlap_count.get(pair, 0) + 1

    # Merge pairs: high mean bbox-IoU AND high lifespan overlap.
    merges: list[tuple[int, int]] = []  # (loser, winner)
    for pair in sorted(bbox_overlap_count):
        n = bbox_overlap_count[pair]
        if n < min_frames:
            continue
        mean_iou = bbox_overlap_sum[pair] / n
        a, b = pair
        a0, a1 = lifespan[a]; b0, b1 = lifespan[b]
        overlap_len = max(0, min(a1, b1) - max(a0, b0) + 1)
        shorter = min(a1 - a0 + 1, b1 - b0 + 1)
        lifespan_overlap_frac = overlap_len / max(1, shorter)
        if mean_iou >= iou_thr and lifespan_overlap_frac >= min_lifespan_overlap:
            loser, winner = (a, b) if area_sum[a] < area_sum[b] else (b, a)
            merges.append((loser, winner))
            print(f"[dedup] MERGE {loser}->{winner} (mean bbox-IoU={mean_iou:.3f}, "
                  f"lifespan overlap={lifespan_overlap_frac:.2f})")

    # Transitively resolve merge chains (a->b, b->c => a->c).
    remap = {l: w for l, w in merges}
    changed = True
    while changed:
        changed = False
        for k in list(remap):
            if remap[k] in remap:
                remap[k] = remap[remap[k]]
                changed = True

    # Renumber survivors by area desc (largest = 1).
    survivors = sorted(set(obj_ids) - set(remap.keys()), key=lambda o: -area_sum[o])
    renumber = {old: new for new, old in enumerate(survivors, start=1)}
    full_map: dict[int, int] = {}
    for oid in obj_ids:
        target = remap.get(oid, oid)
        full_map[oid] = renumber.get(target, target)
    for g in ghost_ids:
        full_map[g] = 0  # erase

    # Pass 3: rewrite masks.
    for p in mask_files:
        m = load_palette_mask(p)
        out = np.zeros_like(m)
        for old_id, new_id in full_map.items():
            if new_id == 0:
                continue
            out[m == old_id] = new_id
        save_palette_mask(out, p, out_masks_dir / p.name)
    print(f"[dedup] wrote {len(mask_files)} masks -> {out_masks_dir}")

    stats = {
        "input_obj_ids": sorted(obj_ids | ghost_ids),
        "ghost_ids": sorted(ghost_ids),
        "merges_applied": [{"loser": l, "winner": w} for l, w in remap.items()],
        "kept_obj_ids": sorted(set(obj_ids) - set(remap.keys())),
        "renumber": renumber,
        "full_id_remap": full_map,
        "iou_thr": iou_thr,
        "min_lifespan_overlap": min_lifespan_overlap,
        "min_mean_area_frac": min_mean_area_frac,
        "min_frames": min_frames,
    }
    (out_masks_dir.parent / "dedup_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


__all__ = ["dedup_masks", "load_palette_mask", "save_palette_mask"]
