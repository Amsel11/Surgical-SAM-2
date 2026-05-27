"""Deterministic stratified train/val/test split over a labelled cohort.

Phase 3 of the GD fine-tune plan. Reads videos in --cohort that have a
labelled prompt_set (no remaining 'unknown_instrument' rows in the seed/
method scope), computes each video's class membership, then runs iterative
stratification — assigns the rarest class first to the split that needs it
most, to give every class non-zero coverage in train/val/test even when N
is small.

The output YAML is checked into git so split assignment is reproducible
and inspectable. Re-running with the same seed + cohort yields the same
split.

Usage:
    python -m tools.make_split --out configs/splits/whip_ft_v1.yaml
    python -m tools.make_split --cohort whip --train 0.65 --val 0.20 --test 0.15
    python -m tools.make_split --dry-run                # print, don't write

The default 65/20/15 split is sized for N≈38 (gives 24/8/6). For the
larger 114-cohort run later, defaults still hold — they're percentages.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.db import REPO_ROOT, connect

DEFAULT_OUT = REPO_ROOT / "configs" / "splits" / "whip_ft_v1.yaml"
DEFAULT_TRAIN = 0.65
DEFAULT_VAL = 0.20
DEFAULT_TEST = 0.15

LABELS_SQL = """
SELECT v.video_id, po.instrument_id
FROM videos v
JOIN prompt_sets ps      ON ps.video_id = v.video_id
JOIN prompt_objects po   ON po.prompt_set_id = ps.prompt_set_id
WHERE v.cohort        = ?
  AND ps.seed         = ?
  AND ps.prompt_method = ?
  AND ps.status        = 'ready'
  AND po.instrument_id != 'unknown_instrument'
"""


def collect_video_classes(
    conn, cohort: str, seed: int, method: str, min_class_videos: int
) -> tuple[dict[str, set[str]], list[str]]:
    """Return ({video_id: set(instrument_ids)}, ordered_classes_rare_first).

    Classes appearing in fewer than `min_class_videos` videos are dropped
    (and the videos they belong to may be dropped if no other class survives).
    """
    pairs = list(conn.execute(LABELS_SQL, (cohort, seed, method)))
    raw: dict[str, set[str]] = defaultdict(set)
    for r in pairs:
        raw[r["video_id"]].add(r["instrument_id"])

    class_video_count: Counter[str] = Counter()
    for vid, cls_set in raw.items():
        for c in cls_set:
            class_video_count[c] += 1

    surviving = {c for c, n in class_video_count.items() if n >= min_class_videos}
    video_classes: dict[str, set[str]] = {}
    for vid, cls_set in raw.items():
        intersect = cls_set & surviving
        if intersect:
            video_classes[vid] = intersect

    ordered_classes = sorted(surviving, key=lambda c: (class_video_count[c], c))
    return video_classes, ordered_classes


def iterative_stratified_split(
    video_classes: dict[str, set[str]],
    ordered_classes: list[str],
    train: float, val: float, test: float,
) -> dict[str, list[str]]:
    """Iterative stratification (Sechidis et al. 2011, simplified for 3 splits).

    Assigns each video to one of train/val/test such that every class is
    represented in every split when possible. Works rare-first so small
    classes get their guaranteed slots before the larger ones bid.
    """
    splits = {"train": [], "val": [], "test": []}
    targets = {"train": train, "val": val, "test": test}
    # Per-split desired count per class.
    desired: dict[str, dict[str, float]] = {
        s: {c: targets[s] * sum(1 for cs in video_classes.values() if c in cs)
            for c in ordered_classes}
        for s in splits
    }
    unassigned = set(video_classes.keys())

    for cls in ordered_classes:
        # Videos containing this class, not yet assigned, sorted by id for determinism.
        candidates = sorted(v for v in unassigned if cls in video_classes[v])
        for vid in candidates:
            # Pick split with greatest remaining desired count for THIS class.
            # Ties → split with smallest total assigned count (balance overall size).
            best_split = max(
                splits,
                key=lambda s: (desired[s][cls], -len(splits[s])),
            )
            splits[best_split].append(vid)
            unassigned.discard(vid)
            for c2 in video_classes[vid]:
                if c2 in desired[best_split]:
                    desired[best_split][c2] -= 1

    # Anything left (videos with no surviving class) → round-robin to balance.
    leftovers = sorted(unassigned)
    for vid in leftovers:
        smallest = min(splits, key=lambda s: len(splits[s]))
        splits[smallest].append(vid)

    for s in splits:
        splits[s].sort()
    return splits


def verify_split(
    splits: dict[str, list[str]],
    video_classes: dict[str, set[str]],
    ordered_classes: list[str],
) -> list[str]:
    """Return a list of human-readable issues (empty list = clean split)."""
    issues: list[str] = []
    for cls in ordered_classes:
        coverage = {
            s: sum(1 for v in splits[s] if cls in video_classes.get(v, set()))
            for s in splits
        }
        zero_splits = [s for s, n in coverage.items() if n == 0]
        if zero_splits:
            issues.append(f"class {cls!r} absent from {zero_splits} (coverage: {coverage})")
    return issues


def write_yaml(splits: dict[str, list[str]], path: Path, meta: dict) -> None:
    """Hand-rolled YAML so we don't pull pyyaml in just for this."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Generated by tools/make_split.py — re-run to regenerate.")
    lines.append(f"# meta: {json.dumps(meta)}")
    lines.append("")
    for split_name in ("train", "val", "test"):
        lines.append(f"{split_name}:")
        for vid in splits[split_name]:
            lines.append(f"  - {vid}")
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cohort", default="whip")
    p.add_argument("--seed", type=int, default=1,
                   help="prompt_sets.seed (label source) (default 1).")
    p.add_argument("--method", default="manual_box")
    p.add_argument("--train", type=float, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=float, default=DEFAULT_VAL)
    p.add_argument("--test", type=float, default=DEFAULT_TEST)
    p.add_argument("--min-class-videos", type=int, default=2,
                   help="Drop classes appearing in fewer than this many videos (default 2).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--db", help="Manifest path override.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the split + issues; don't write YAML.")
    args = p.parse_args(argv)

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print(f"FATAL: train+val+test must sum to 1.0 "
              f"(got {args.train + args.val + args.test:.3f})", file=sys.stderr)
        return 2

    conn = connect(args.db)
    video_classes, ordered_classes = collect_video_classes(
        conn, args.cohort, args.seed, args.method, args.min_class_videos
    )
    if not video_classes:
        print(f"No labelled videos in cohort={args.cohort!r} "
              f"seed={args.seed} method={args.method!r}.", file=sys.stderr)
        return 2

    splits = iterative_stratified_split(
        video_classes, ordered_classes,
        train=args.train, val=args.val, test=args.test,
    )
    issues = verify_split(splits, video_classes, ordered_classes)

    print(f"--- make_split: cohort={args.cohort} "
          f"({args.train:.2f}/{args.val:.2f}/{args.test:.2f})", file=sys.stderr)
    print(f"  videos: {sum(len(s) for s in splits.values())} "
          f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})",
          file=sys.stderr)
    print(f"  classes: {len(ordered_classes)}  {ordered_classes}", file=sys.stderr)
    for cls in ordered_classes:
        cov = {s: sum(1 for v in splits[s] if cls in video_classes.get(v, set())) for s in splits}
        print(f"    {cls:30}  train={cov['train']:2}  val={cov['val']:2}  test={cov['test']:2}",
              file=sys.stderr)
    if issues:
        print("\nISSUES:", file=sys.stderr)
        for i in issues:
            print(f"  ⚠ {i}", file=sys.stderr)

    meta = {
        "cohort": args.cohort,
        "seed": args.seed,
        "method": args.method,
        "train_pct": args.train,
        "val_pct": args.val,
        "test_pct": args.test,
        "n_videos": sum(len(s) for s in splits.values()),
        "n_classes": len(ordered_classes),
        "classes": ordered_classes,
    }
    if args.dry_run:
        print("\n[dry-run]", file=sys.stderr)
        for s in ("train", "val", "test"):
            print(f"  {s}: {splits[s]}")
        return 0

    write_yaml(splits, args.out, meta)
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
