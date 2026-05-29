"""Strip-clip + subsample a tracker's palette masks into a lean clean set.

Two defects motivate this:
  1) SAM absorbs the bottom da Vinci status strip (high-contrast overlay text)
     into the mask — the instrument is occluded behind that opaque bar, so the
     true mask should end at the strip's top edge. We zero the bottom
     max(40, strip_frac*H) rows (same geometry as the OCR tool).
  2) A palette PNG per frame is far more than the FT cohort needs (consecutive
     frames are near-duplicates). We keep every --keep-stride-th frame.

NON-DESTRUCTIVE: writes results/<run>/<video>/seed_<k>/<out-subdir>/ and never
touches the dense originals. Promotion/deletion of the originals is a separate,
explicit step once the clean set is eyeballed.

Palette is preserved (mode 'P', pixel value == obj_id) so downstream readers
(extract_ft_labels, render_object_cutouts) see identical obj-id semantics.

Usage:
    python -m tools.clean_subsample_masks --run surgsam2_oob_whip_v1 --seed 1
    python -m tools.clean_subsample_masks --run surgsam2_oob_whip_v1 --video DC_whip_11609423 \
        --keep-stride 15 --strip-frac 0.17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.db import REPO_ROOT

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def parse_frame_idx(name: str) -> int:
    return int(Path(name).stem)


def find_mask_dirs(results_root: Path, run: str, seed: int, subdir: str) -> list[tuple[str, Path]]:
    run_dir = results_root / run
    if not run_dir.exists():
        return []
    out = []
    for vdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        masks = vdir / f"seed_{seed}" / subdir
        if masks.exists():
            out.append((vdir.name, masks))
    return out


def clean_one(masks_dir: Path, out_dir: Path, keep_stride: int, strip_frac: float) -> dict:
    files = sorted(masks_dir.glob("*.png"), key=lambda p: parse_frame_idx(p.name))
    if not files:
        return {"total": 0, "kept": 0}
    kept_files = files[::max(1, keep_stride)]
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for f in kept_files:
        with Image.open(f) as im:
            palette = im.getpalette()
            arr = np.array(im)  # mode 'P' -> palette indices == obj_ids
        if strip_frac > 0:
            H = arr.shape[0]
            sp = max(40, int(H * strip_frac))
            arr[H - sp:, :] = 0
        out = Image.fromarray(arr, mode="P")
        if palette is not None:
            out.putpalette(palette)
        out.save(out_dir / f.name)
        kept += 1
    return {"total": len(files), "kept": kept}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="results/<run>/ dirname.")
    p.add_argument("--video", help="Restrict to one video_id (default: all in run).")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--in-subdir", default="masks", help="Source masks subdir.")
    p.add_argument("--out-subdir", default="masks_clean", help="Destination subdir.")
    p.add_argument("--keep-stride", type=int, default=15,
                   help="Keep every N-th mask file (1 = keep all).")
    p.add_argument("--strip-frac", type=float, default=0.17,
                   help="Zero bottom max(40, frac*H) rows (da Vinci strip). 0 = off.")
    args = p.parse_args(argv)

    pairs = find_mask_dirs(args.results_root, args.run, args.seed, args.in_subdir)
    if args.video:
        pairs = [(v, m) for v, m in pairs if v == args.video]
    if not pairs:
        print(f"No masks under {args.results_root/args.run}/<video>/seed_{args.seed}/{args.in_subdir}/",
              file=sys.stderr)
        return 2

    tot_in = tot_out = 0
    for video_id, masks_dir in pairs:
        out_dir = masks_dir.parent / args.out_subdir
        st = clean_one(masks_dir, out_dir, args.keep_stride, args.strip_frac)
        tot_in += st["total"]; tot_out += st["kept"]
        print(f"  {video_id:28} {st['total']:>6} -> {st['kept']:>5} kept  ({out_dir})",
              file=sys.stderr)
    print(f"DONE: {tot_in} dense masks -> {tot_out} clean kept "
          f"(stride {args.keep_stride}, strip_frac {args.strip_frac})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
