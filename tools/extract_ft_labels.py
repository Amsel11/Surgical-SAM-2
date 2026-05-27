"""Extract Tier-B FT labels from any tracker's palette-mask outputs.

Phase 2 of the GD fine-tune plan. The Stage-1 clicker provides 3 anchor frames
per video (Tier A); a Stage-2 tracker (SurgSAM-2, SAM 2 stock, or SAM 3)
propagates those anchors through every frame in the video, emitting
results/<run>/<video>/seed_<k>/masks/*.png palette PNGs where pixel value =
obj_id. This script walks those masks, extracts a tight bbox per (frame, obj_id),
inherits the obj's instrument_id from prompt_objects, applies a few sanity
filters, and emits a JSONL training corpus.

Generic over tracker: pass --run <name> for whichever tracker produced the
masks. The conversion logic doesn't care if it's SurgSAM-2 or SAM 3.

Filters:
  - area_px < --min-area-px               : drop noise / tiny mask fragments
  - frame outside [first_anchor, last_anchor]
                                          : drop pre-/post-anchor extrapolation
  - area changed >--max-area-jump-ratio   : drop frames where SAM clearly
                                            drifted onto another object
  - obj_id labelled 'unknown_instrument'  : skip entirely (can't train a class
                                            we don't know)

Output JSONL row:
    {"video_id": str, "frame_idx": int, "obj_id": int, "instrument_id": str,
     "box_xyxy": [x0,y0,x1,y1], "area_px": int, "source": <run>}

Usage:
    python -m tools.extract_ft_labels --run surgsam2_oob_whip_v1
    python -m tools.extract_ft_labels --run surgsam2_oob_whip_v1 --seed 1
    python -m tools.extract_ft_labels --run sam3_oob_whip_v1 --cohort whip
    python -m tools.extract_ft_labels --run surgsam2_oob_whip_v1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.db import REPO_ROOT, connect

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_OUT = REPO_ROOT / "data" / "ft_labels_v1.jsonl"
DEFAULT_MIN_AREA_PX = 200
DEFAULT_MAX_AREA_JUMP_RATIO = 5.0


def parse_frame_idx(name: str) -> int:
    """Mask filenames are zero-padded integers (pipeline.io convention)."""
    return int(Path(name).stem)


def tight_bbox(mask: np.ndarray, oid: int) -> tuple[list[int], int] | None:
    """Return ([x0,y0,x1,y1], area_px) for pixels == oid; None if absent."""
    ys, xs = np.where(mask == oid)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], int(ys.size)


def load_obj_labels(
    conn, video_id: str, seed: int, method: str
) -> tuple[dict[int, str], tuple[int, int] | None]:
    """Return ({obj_id: instrument_id}, trusted_span) from manifest + prompts JSON.

    instrument_id == 'unknown_instrument' rows are filtered out — those would
    be label noise. trusted_span is [first_anchor, last_anchor] from the JSON's
    prompt_frames; None when prompts JSON is missing on disk.
    """
    ps_row = conn.execute(
        """SELECT prompt_set_id, prompts_path FROM prompt_sets
           WHERE video_id=? AND seed=? AND prompt_method=? AND status='ready'""",
        (video_id, seed, method),
    ).fetchone()
    if ps_row is None:
        return {}, None
    po_rows = conn.execute(
        "SELECT obj_id, instrument_id FROM prompt_objects WHERE prompt_set_id=?",
        (ps_row["prompt_set_id"],),
    ).fetchall()
    labels = {
        int(r["obj_id"]): r["instrument_id"]
        for r in po_rows
        if r["instrument_id"] and r["instrument_id"] != "unknown_instrument"
    }
    trusted_span: tuple[int, int] | None = None
    if ps_row["prompts_path"] and Path(ps_row["prompts_path"]).exists():
        prompts = json.loads(Path(ps_row["prompts_path"]).read_text())
        frames = prompts.get("prompt_frames") or []
        if frames:
            trusted_span = (int(min(frames)), int(max(frames)))
    return labels, trusted_span


def find_mask_dirs(results_root: Path, run_name: str, seed: int) -> list[tuple[str, Path]]:
    """Return [(video_id, masks_dir), ...] for results/<run>/<video>/seed_<seed>/masks/."""
    run_dir = results_root / run_name
    if not run_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for video_dir in sorted(run_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        masks = video_dir / f"seed_{seed}" / "masks"
        if not masks.exists():
            # Some pipelines use `seed_{k}_{model}/masks/` — fall back.
            alt = list(video_dir.glob(f"seed_{seed}*/masks"))
            if not alt:
                continue
            masks = alt[0]
        out.append((video_dir.name, masks))
    return out


def extract_one(
    video_id: str,
    masks_dir: Path,
    obj_labels: dict[int, str],
    trusted_span: tuple[int, int] | None,
    min_area_px: int,
    max_area_jump_ratio: float,
    source: str,
) -> tuple[list[dict], dict]:
    """Walk masks/*.png for one video. Return (records, stats)."""
    if not obj_labels:
        return [], {"video_id": video_id, "n_frames": 0, "n_records": 0,
                    "skipped_no_labels": True}
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: parse_frame_idx(p.name))
    if not mask_files:
        return [], {"video_id": video_id, "n_frames": 0, "n_records": 0,
                    "skipped_no_masks": True}

    last_area: dict[int, int] = {}
    records: list[dict] = []
    n_dropped_area = 0
    n_dropped_jump = 0
    n_dropped_span = 0

    for mp in mask_files:
        frame_idx = parse_frame_idx(mp.name)
        if trusted_span and not (trusted_span[0] <= frame_idx <= trusted_span[1]):
            n_dropped_span += 1
            continue
        with Image.open(mp) as im:
            mask = np.array(im)
        for oid, instrument_id in obj_labels.items():
            bb = tight_bbox(mask, oid)
            if bb is None:
                # Object absent in this frame — break the last_area chain so
                # the next-appearance jump check starts fresh, not against an
                # arbitrarily-old area value.
                last_area.pop(oid, None)
                continue
            box, area = bb
            if area < min_area_px:
                n_dropped_area += 1
                continue
            prev = last_area.get(oid)
            if prev is not None:
                ratio = max(area, prev) / max(1, min(area, prev))
                if ratio > max_area_jump_ratio:
                    n_dropped_jump += 1
                    last_area[oid] = area
                    continue
            last_area[oid] = area
            records.append({
                "video_id": video_id,
                "frame_idx": frame_idx,
                "obj_id": oid,
                "instrument_id": instrument_id,
                "box_xyxy": box,
                "area_px": area,
                "source": source,
            })

    return records, {
        "video_id": video_id,
        "n_frames": len(mask_files),
        "n_records": len(records),
        "n_dropped_area": n_dropped_area,
        "n_dropped_jump": n_dropped_jump,
        "n_dropped_outside_span": n_dropped_span,
        "trusted_span": list(trusted_span) if trusted_span else None,
        "n_objs_labelled": len(obj_labels),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True,
                   help="Tracker run name (= results/<run>/ dirname).")
    p.add_argument("--seed", type=int, default=1,
                   help="prompt_sets.seed to read labels from (default 1). "
                        "Inference output dir must also be seed_<this>/.")
    p.add_argument("--method", default="manual_box",
                   help="prompt_sets.prompt_method to read labels from (default 'manual_box').")
    p.add_argument("--cohort", default="whip",
                   help="Restrict to a cohort (default 'whip'). Empty string = all.")
    p.add_argument("--video", help="Restrict to one video_id.")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output JSONL (default {DEFAULT_OUT}).")
    p.add_argument("--min-area-px", type=int, default=DEFAULT_MIN_AREA_PX,
                   help=f"Drop boxes with mask area < this (default {DEFAULT_MIN_AREA_PX}).")
    p.add_argument("--max-area-jump-ratio", type=float, default=DEFAULT_MAX_AREA_JUMP_RATIO,
                   help="Drop frames where mask area changes by >ratio between "
                        f"consecutive frames (default {DEFAULT_MAX_AREA_JUMP_RATIO}).")
    p.add_argument("--db", help="Manifest path override.")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk masks + print stats but don't write the JSONL.")
    args = p.parse_args(argv)

    conn = connect(args.db)

    # Resolve scope: every video in results/<run>/ filtered by cohort + optional --video
    pairs = find_mask_dirs(args.results_root, args.run, args.seed)
    if not pairs:
        print(f"No mask dirs under {args.results_root / args.run}/<video>/seed_{args.seed}/masks/",
              file=sys.stderr)
        return 2
    if args.cohort:
        in_cohort = {
            r["video_id"] for r in conn.execute(
                "SELECT video_id FROM videos WHERE cohort = ?", (args.cohort,)
            )
        }
        pairs = [(v, p) for v, p in pairs if v in in_cohort]
    if args.video:
        pairs = [(v, p) for v, p in pairs if v == args.video]
    if not pairs:
        print("No videos in scope after cohort/video filtering.", file=sys.stderr)
        return 2

    all_records: list[dict] = []
    summaries: list[dict] = []
    for video_id, masks_dir in pairs:
        labels, trusted_span = load_obj_labels(conn, video_id, args.seed, args.method)
        records, stats = extract_one(
            video_id=video_id,
            masks_dir=masks_dir,
            obj_labels=labels,
            trusted_span=trusted_span,
            min_area_px=args.min_area_px,
            max_area_jump_ratio=args.max_area_jump_ratio,
            source=args.run,
        )
        all_records.extend(records)
        summaries.append(stats)
        flag = ""
        if stats.get("skipped_no_labels"):
            flag = "  (no labels — relabel needed)"
        elif stats.get("skipped_no_masks"):
            flag = "  (no masks)"
        print(f"  {video_id:30} {stats['n_records']:6} records  "
              f"(frames={stats['n_frames']}, objs={stats.get('n_objs_labelled', 0)}){flag}",
              file=sys.stderr)

    # Per-class totals
    per_class: dict[str, int] = {}
    for r in all_records:
        per_class[r["instrument_id"]] = per_class.get(r["instrument_id"], 0) + 1
    print(f"\n--- {args.run} totals", file=sys.stderr)
    print(f"  videos:  {len(summaries)}", file=sys.stderr)
    print(f"  records: {len(all_records)}", file=sys.stderr)
    print(f"  per-class:", file=sys.stderr)
    for inst, n in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {inst:30}  {n}", file=sys.stderr)

    if args.dry_run:
        print("\n[dry-run] no JSONL written", file=sys.stderr)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(all_records)} records → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
