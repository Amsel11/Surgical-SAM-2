"""Audit instrument labels on prompt_objects, per video × prompt_method.

Phase 0a of the GD fine-tune plan. Surfaces, for a cohort:
  - which videos have any prompt_sets at all
  - which prompt_objects rows are still 'unknown_instrument' (legacy ingest)
  - how many distinct obj_ids per video (a low count flags videos that
    likely need extra boxes drawn during the relabel pass)
  - which instrument_ids have been used so far

After the relabel pass (Phase 0b), re-running this should report zero
'unknown_instrument' rows. The `instruments_seen` aggregate then feeds
tools/build_vocab.py to derive the empirical vocabulary.

Usage:
    python -m tools.audit_labels                            # whip cohort, stdout
    python -m tools.audit_labels --out local_results/label_audit.csv
    python -m tools.audit_labels --cohort endovis18

On bp:
    .venv/bin/python -m tools.audit_labels \\
        --out local_results/label_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from pipeline.db import connect, db_path

AUDIT_SQL = """
SELECT v.video_id,
       ps.prompt_method,
       COUNT(DISTINCT po.obj_id) AS n_objects,
       SUM(CASE WHEN po.instrument_id  = 'unknown_instrument' THEN 1 ELSE 0 END) AS n_unknown,
       SUM(CASE WHEN po.instrument_id != 'unknown_instrument' THEN 1 ELSE 0 END) AS n_labelled,
       GROUP_CONCAT(DISTINCT
           CASE WHEN po.instrument_id != 'unknown_instrument'
                THEN po.instrument_id END
       ) AS instruments_seen
FROM videos v
LEFT JOIN prompt_sets ps    ON ps.video_id = v.video_id
LEFT JOIN prompt_objects po ON po.prompt_set_id = ps.prompt_set_id
WHERE v.cohort = ?
GROUP BY v.video_id, ps.prompt_method
ORDER BY v.video_id, ps.prompt_method
"""

COLUMNS = ["video_id", "prompt_method", "n_objects", "n_unknown", "n_labelled", "instruments_seen"]


def run_audit(conn, cohort: str) -> list[dict]:
    rows = []
    for r in conn.execute(AUDIT_SQL, (cohort,)):
        rows.append({c: r[c] for c in COLUMNS})
    return rows


def write_csv(rows: list[dict], out: Path | None) -> None:
    f = open(out, "w", newline="") if out else sys.stdout
    try:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    finally:
        if out:
            f.close()


def print_summary(rows: list[dict], cohort: str, out_path: Path | None) -> None:
    """One-paragraph console summary aimed at the relabel-pass operator."""
    videos = sorted({r["video_id"] for r in rows})

    videos_with_prompts = {r["video_id"] for r in rows if r["prompt_method"] is not None}
    videos_without_prompts = [v for v in videos if v not in videos_with_prompts]

    per_video_unknown: dict[str, int] = {}
    per_video_objs: dict[str, int] = {}
    for r in rows:
        if r["prompt_method"] is None:
            continue
        per_video_unknown[r["video_id"]] = per_video_unknown.get(r["video_id"], 0) + (r["n_unknown"] or 0)
        per_video_objs[r["video_id"]]    = per_video_objs.get(r["video_id"], 0)    + (r["n_objects"] or 0)

    videos_needing_relabel = sorted(v for v, n in per_video_unknown.items() if n > 0)
    videos_fully_labelled  = sorted(v for v in videos_with_prompts if per_video_unknown.get(v, 0) == 0)

    instruments_seen: set[str] = set()
    for r in rows:
        s = r["instruments_seen"]
        if s:
            instruments_seen.update(p for p in s.split(",") if p)

    low_obj_videos = sorted(v for v in videos_with_prompts if per_video_objs.get(v, 0) < 3)

    print("---", file=sys.stderr)
    print(f"label audit: cohort={cohort}", file=sys.stderr)
    print(f"  videos in cohort           : {len(videos)}", file=sys.stderr)
    print(f"  videos with prompt_sets    : {len(videos_with_prompts)}", file=sys.stderr)
    print(f"  videos missing prompt_sets : {len(videos_without_prompts)}", file=sys.stderr)
    print(f"  videos needing relabel     : {len(videos_needing_relabel)}", file=sys.stderr)
    print(f"  videos fully labelled      : {len(videos_fully_labelled)}", file=sys.stderr)
    print(f"  distinct instruments seen  : {len(instruments_seen)}", file=sys.stderr)
    if instruments_seen:
        print(f"    {sorted(instruments_seen)}", file=sys.stderr)
    if videos_without_prompts:
        print(f"  no-prompt-set videos: {videos_without_prompts}", file=sys.stderr)
    if low_obj_videos:
        print(f"  videos with <3 objects (probable missing arms): {low_obj_videos}", file=sys.stderr)
    if out_path:
        print(f"  csv: {out_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", help="Manifest path (default: $SURGSAM_MANIFEST or repo/manifest.db)")
    p.add_argument("--cohort", default="whip", help="videos.cohort filter (default: whip)")
    p.add_argument("--out", type=Path, help="CSV output path (default: stdout)")
    args = p.parse_args(argv)

    conn = connect(args.db)
    rows = run_audit(conn, args.cohort)
    write_csv(rows, args.out)
    print_summary(rows, args.cohort, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
