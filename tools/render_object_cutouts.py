"""Per-object cutout strips from a tracker's palette masks — visual QC.

For each (video, obj_id) in a run's seed_<k>/masks/, crop the object's masked
region at N evenly-sampled frames across its track and lay them out as a
labeled horizontal strip, so you can eyeball whether each tracked obj_id is the
instrument its prompt_objects label claims AND whether the track stays on the
right thing across the video.

Why this instead of the combined overlay video: the overlay colors every
object at once, which is great for "is anything tracking" but poor for "is
obj 2 actually the vessel sealer the whole time." One strip per object makes
per-instrument identity checkable at a glance.

Output (per video):
    results/<run>/<video>/seed_<seed>/cutouts/obj<NN>_<instrument_id>.png
    results/<run>/<video>/seed_<seed>/cutouts/_contact_sheet.png   (all objects stacked)

Each crop has the mask outline drawn (DAVIS color = obj_id) and its source
frame index burned in; the strip's left band shows "obj N / <instrument_id>".

Reads palette PNGs with PIL (pixel value == obj_id — cv2 would decode the
palette to RGB and lose the ids). Source frames + obj->instrument labels come
from the same manifest the inference used. Runs in .venv (numpy + cv2 + PIL).

Usage:
    python -m tools.render_object_cutouts --run surgsam2_oob_whip_v1 --video DC_whip_11609423
    python -m tools.render_object_cutouts --run surgsam2_oob_whip_v1            # all videos in the run
    python -m tools.render_object_cutouts --run surgsam2_oob_whip_v1 --seed 1 --n-samples 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.db import REPO_ROOT, connect
from pipeline.io import frame_files_ordered

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"

# DAVIS-style palette (BGR) keyed by obj_id — matches the overlay renderers so
# a given obj_id is the same color here as in overlay.mp4.
_DAVIS = [
    (236, 95, 103), (130, 196, 109), (60, 188, 252), (45, 117, 255),
    (180, 130, 70), (137, 79, 247), (61, 220, 255), (180, 180, 180),
]


def davis_bgr(obj_id: int) -> tuple[int, int, int]:
    return _DAVIS[(obj_id * 3) % len(_DAVIS)]


def parse_frame_idx(name: str) -> int:
    return int(Path(name).stem)


def load_labels(conn, video_id: str, seed: int, method: str) -> dict[int, str]:
    """{obj_id: instrument_id} from the manual_box prompt_set (labels source)."""
    row = conn.execute(
        "SELECT prompt_set_id FROM prompt_sets WHERE video_id=? AND seed=? AND prompt_method=?",
        (video_id, seed, method),
    ).fetchone()
    if row is None:
        return {}
    return {
        int(r["obj_id"]): r["instrument_id"]
        for r in conn.execute(
            "SELECT obj_id, instrument_id FROM prompt_objects WHERE prompt_set_id=?",
            (row["prompt_set_id"],),
        )
    }


def frames_dir_for(conn, video_id: str) -> Path | None:
    row = conn.execute(
        "SELECT frames_dir FROM videos WHERE video_id=?", (video_id,)
    ).fetchone()
    return Path(row["frames_dir"]) if row and row["frames_dir"] else None


def find_mask_dirs(results_root: Path, run: str, seed: int) -> list[tuple[str, Path]]:
    run_dir = results_root / run
    if not run_dir.exists():
        return []
    out = []
    for vdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        masks = vdir / f"seed_{seed}" / "masks"
        if not masks.exists():
            alt = list(vdir.glob(f"seed_{seed}*/masks"))
            if not alt:
                continue
            masks = alt[0]
        out.append((vdir.name, masks))
    return out


def obj_bbox(mask: np.ndarray, oid: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask == oid)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def make_crop(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    oid: int,
    bbox: tuple[int, int, int, int],
    pad_frac: float,
    thumb: int,
    frame_idx: int,
) -> np.ndarray:
    """Crop bbox+pad from the frame, draw the obj's mask outline, square-pad to `thumb`."""
    H, W = frame_bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad_frac) + 4, int(bh * pad_frac) + 4
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1 + px), min(H, y1 + py)
    crop = frame_bgr[cy0:cy1, cx0:cx1].copy()

    # Outline this object's mask region within the crop.
    sub = (mask[cy0:cy1, cx0:cx1] == oid).astype(np.uint8)
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, davis_bgr(oid), 2)

    # Square-pad then resize to thumb so strips align regardless of aspect.
    ch, cw = crop.shape[:2]
    side = max(ch, cw, 1)
    sq = np.zeros((side, side, 3), dtype=np.uint8)
    sq[(side - ch) // 2:(side - ch) // 2 + ch, (side - cw) // 2:(side - cw) // 2 + cw] = crop
    out = cv2.resize(sq, (thumb, thumb), interpolation=cv2.INTER_AREA)
    cv2.putText(out, f"f{frame_idx}", (4, thumb - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def label_band(text_lines: list[str], thumb: int, oid: int, width: int = 150) -> np.ndarray:
    band = np.full((thumb, width, 3), 30, dtype=np.uint8)
    cv2.rectangle(band, (0, 0), (8, thumb), davis_bgr(oid), -1)
    y = 28
    for ln in text_lines:
        cv2.putText(band, ln, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return band


def render_video(
    video_id: str, masks_dir: Path, frames_dir: Path, labels: dict[int, str],
    out_dir: Path, n_samples: int, pad_frac: float, thumb: int,
    strip_frac: float = 0.0,
) -> dict:
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: parse_frame_idx(p.name))
    if not mask_files:
        return {"video_id": video_id, "objs": 0, "skipped": "no masks"}
    ordered = frame_files_ordered(frames_dir)

    # First pass: which frames each obj appears in (by loader idx == mask stem).
    present: dict[int, list[int]] = {}
    for mp in mask_files:
        fi = parse_frame_idx(mp.name)
        with Image.open(mp) as im:
            ids = np.unique(np.array(im))
        for oid in ids:
            if oid == 0:
                continue
            present.setdefault(int(oid), []).append(fi)

    out_dir.mkdir(parents=True, exist_ok=True)
    strips = []
    for oid in sorted(present):
        frame_idxs = present[oid]
        # Evenly sample across the object's track.
        if len(frame_idxs) <= n_samples:
            sample = frame_idxs
        else:
            step = len(frame_idxs) / n_samples
            sample = [frame_idxs[int(i * step)] for i in range(n_samples)]

        crops = []
        for fi in sample:
            if fi >= len(ordered):
                continue
            with Image.open(masks_dir / f"{fi:05d}.png") as im:
                mask = np.array(im)
            frame_bgr = cv2.imread(str((frames_dir / ordered[fi]).resolve()))
            if frame_bgr is None:
                continue
            if mask.shape[:2] != frame_bgr.shape[:2]:
                mask = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            # Zero the bottom da Vinci status strip — SAM frequently absorbs its
            # high-contrast text into the mask (the strip is an opaque overlay,
            # so the instrument is occluded there anyway). Same geometry as the
            # OCR tool: bottom max(40, strip_frac*H) rows.
            if strip_frac > 0:
                H = mask.shape[0]
                sp = max(40, int(H * strip_frac))
                mask[H - sp:, :] = 0
            bb = obj_bbox(mask, oid)
            if bb is None:
                continue
            crops.append(make_crop(frame_bgr, mask, oid, bb, pad_frac, thumb, fi))
        if not crops:
            continue
        inst = labels.get(oid, "UNLABELED")
        band = label_band([f"obj {oid}", inst, f"{len(frame_idxs)} frm"], thumb, oid)
        strip = cv2.hconcat([band] + crops)
        cv2.imwrite(str(out_dir / f"obj{oid:02d}_{inst}.png"), strip)
        strips.append(strip)

    if strips:
        width = max(s.shape[1] for s in strips)
        padded = [cv2.copyMakeBorder(s, 0, 0, 0, width - s.shape[1],
                                     cv2.BORDER_CONSTANT, value=(20, 20, 20)) for s in strips]
        cv2.imwrite(str(out_dir / "_contact_sheet.png"), cv2.vconcat(padded))
    return {"video_id": video_id, "objs": len(strips), "out": str(out_dir)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="results/<run>/ dirname.")
    p.add_argument("--video", help="Restrict to one video_id (default: all in run).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--method", default="manual_box", help="prompt_method for labels.")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--n-samples", type=int, default=8, help="Crops per object across its track.")
    p.add_argument("--pad-frac", type=float, default=0.15)
    p.add_argument("--thumb", type=int, default=256)
    p.add_argument("--strip-frac", type=float, default=0.0,
                   help="Zero the bottom da Vinci strip (frac of H, e.g. 0.17) before "
                        "computing each box/outline, so absorbed overlay text is excluded.")
    p.add_argument("--db", help="Manifest path override.")
    args = p.parse_args(argv)

    conn = connect(args.db)
    pairs = find_mask_dirs(args.results_root, args.run, args.seed)
    if args.video:
        pairs = [(v, m) for v, m in pairs if v == args.video]
    if not pairs:
        print(f"No masks under {args.results_root/args.run}/<video>/seed_{args.seed}/masks/",
              file=sys.stderr)
        return 2

    for video_id, masks_dir in pairs:
        fdir = frames_dir_for(conn, video_id)
        if fdir is None or not fdir.exists():
            print(f"  {video_id:28} SKIP (frames_dir missing)", file=sys.stderr)
            continue
        labels = load_labels(conn, video_id, args.seed, args.method)
        out_dir = masks_dir.parent / "cutouts"
        stats = render_video(video_id, masks_dir, fdir, labels, out_dir,
                             args.n_samples, args.pad_frac, args.thumb, args.strip_frac)
        print(f"  {video_id:28} {stats.get('objs', 0)} objects -> {stats.get('out', stats.get('skipped'))}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
