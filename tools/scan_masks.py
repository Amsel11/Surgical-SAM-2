"""Scan mask PNGs to compute per-(video, obj) empty-frame breakdown.

Output: CSV to stdout. One row per (video, obj_id).

Columns:
  video_id           video name
  obj_id             integer (matches prompt_objects.obj_id)
  n_frames           total frame count for this video (= number of masks/*.png)
  first_anchor       earliest loader-space frame where this obj was clicked
  last_anchor        latest
  n_anchors          number of anchor frames clicked
  empty_total        frames where this obj has 0 pixels
  empty_in_span      empty frames inside [first_anchor, last_anchor]
                     ^ this is the "tracking failure" signal
  empty_pre_span     empty frames before first_anchor
                     ^ ambiguous — could be legitimate (instrument not yet inserted)
  empty_post_span    empty frames after last_anchor
                     ^ ambiguous — could be legitimate (instrument removed)
  span_length        last_anchor - first_anchor + 1
  in_span_rate       empty_in_span / span_length   <- the honest drift metric

Why this is more honest than the raw rate:
  We can prove the instrument was visible at every anchor frame (the user
  clicked it). Empty masks *between* two anchors that both had this object
  almost certainly mean tracking failure. Empty masks *outside* the span
  could be legitimate absence.

Designed to run on bp where masks/ live; output is tiny so ssh-piping the
csv back to olab-1 is fast.

Usage on bp:
  .venv/bin/python tools/scan_masks.py --results-root results/ \
      --prompts-dir prompts/ --seed 1 > /tmp/mask_scan_seed1.csv

Usage over ssh (run on bp, capture locally):
  ssh bp '.venv/bin/python /gpfs/.../tools/scan_masks.py \
      --results-root /gpfs/.../results --prompts-dir /gpfs/.../prompts --seed 1' \
      > local_results/seed_1/qc/mask_scan.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def scan_video(seed_dir: Path, prompts_path: Path | None) -> list[dict]:
    """Return one dict per (video, obj_id) for one seed dir."""
    log_path = seed_dir / "log.json"
    masks_dir = seed_dir / "masks"
    if not log_path.exists() or not masks_dir.is_dir():
        return []
    log = json.loads(log_path.read_text())
    vid = log["video"]
    source_offset = int(log.get("source_offset", 0))

    # Anchor frames per obj (in loader space).
    obj_anchors: dict[int, set[int]] = {}
    if prompts_path and prompts_path.exists():
        pj = json.loads(prompts_path.read_text())
        for f_str, objs in pj.get("objects_by_frame", {}).items():
            fidx = int(f_str) - source_offset
            if fidx < 0:
                continue
            for o in objs:
                obj_anchors.setdefault(int(o["obj_id"]), set()).add(fidx)

    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem))
    if not mask_files:
        return []
    n_frames = len(mask_files)

    # Per-frame set of obj_ids that have non-zero pixels.
    per_frame_objs: dict[int, set[int]] = {}
    for mp in mask_files:
        n = int(mp.stem)
        arr = np.array(Image.open(mp))
        # palette PNG: each pixel value is an obj_id; 0 means background.
        present = {int(x) for x in np.unique(arr) if x != 0}
        per_frame_objs[n] = present

    # Union of objs that ever appeared + objs that were clicked.
    all_objs: set[int] = set()
    for v in per_frame_objs.values():
        all_objs.update(v)
    all_objs.update(obj_anchors.keys())

    rows = []
    for oid in sorted(all_objs):
        anchors = sorted(obj_anchors.get(oid, set()))
        first_a = anchors[0] if anchors else None
        last_a = anchors[-1] if anchors else None

        empty_total = empty_in = empty_pre = empty_post = 0
        for n in range(n_frames):
            is_empty = oid not in per_frame_objs.get(n, set())
            if not is_empty:
                continue
            empty_total += 1
            if first_a is None:
                continue
            if n < first_a:
                empty_pre += 1
            elif n > last_a:
                empty_post += 1
            else:
                empty_in += 1

        span_length = (last_a - first_a + 1) if first_a is not None else 0
        in_span_rate = (empty_in / span_length) if span_length > 0 else None

        rows.append({
            "video_id": vid,
            "obj_id": oid,
            "n_frames": n_frames,
            "first_anchor": first_a if first_a is not None else "",
            "last_anchor": last_a if last_a is not None else "",
            "n_anchors": len(anchors),
            "empty_total": empty_total,
            "empty_in_span": empty_in,
            "empty_pre_span": empty_pre,
            "empty_post_span": empty_post,
            "span_length": span_length,
            "in_span_rate": f"{in_span_rate:.4f}" if in_span_rate is not None else "",
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--prompts-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    seed_dir_name = f"seed_{args.seed}"
    seed_dirs = sorted(args.results_root.glob(f"*/{seed_dir_name}"))
    if not seed_dirs:
        print(f"No */{seed_dir_name} dirs under {args.results_root}", file=sys.stderr)
        return 1

    fields = ["video_id", "obj_id", "n_frames", "first_anchor", "last_anchor",
              "n_anchors", "empty_total", "empty_in_span", "empty_pre_span",
              "empty_post_span", "span_length", "in_span_rate"]
    w = csv.DictWriter(sys.stdout, fieldnames=fields)
    w.writeheader()

    for sd in seed_dirs:
        vid = sd.parent.name
        pj = args.prompts_dir / f"{vid}_seed{args.seed}_manual_box.json"
        try:
            rows = scan_video(sd, pj)
        except Exception as exc:
            print(f"[{vid}] error: {exc}", file=sys.stderr)
            continue
        for r in rows:
            w.writerow(r)
        print(f"[{vid}] {len(rows)} obj rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
