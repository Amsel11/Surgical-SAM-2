"""Rescale an OCR segments.csv from its native frame indexing to a target
tracking fps, so the AutoPrompter (which treats `start_i`/`end_i` as loader
indices) lands anchors on the frames we actually extract.

The PJA OCR ran at 1 fps, so its `start_i`/`end_i` are second-indexed while
`start_sec`/`end_sec` hold the true wall-clock seconds. We re-derive:

    start_i' = round(start_sec * fps)
    end_i'   = round(end_sec   * fps)

against frames extracted at `--fps`. All other columns are preserved (the arm
labels, the _src/_sec provenance). Segments that collapse to <1 frame after
rounding are dropped; end_i' is clamped >= start_i'.

Usage:
    python -m tools.rescale_segments --in seg.csv --out seg_rescaled.csv --fps 6
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, required=True, help="Tracking-frame fps.")
    args = p.parse_args(argv)

    src = Path(args.inp)
    if not src.exists():
        print(f"missing segments.csv: {src}", file=sys.stderr)
        return 2
    with open(src) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"empty segments.csv: {src}", file=sys.stderr)
        return 2

    fieldnames = list(rows[0].keys())
    out_rows = []
    last_end = -1
    for r in rows:
        s_i = round(float(r["start_sec"]) * args.fps)
        e_i = round(float(r["end_sec"]) * args.fps)
        if e_i < s_i:
            e_i = s_i
        # Keep anchors monotonic: nudge a duplicate start past the previous end.
        if s_i <= last_end:
            s_i = last_end + 1
            if e_i < s_i:
                e_i = s_i
        last_end = e_i
        r = dict(r)
        r["start_i"] = s_i
        r["end_i"] = e_i
        if "n_frames" in r:
            r["n_frames"] = e_i - s_i + 1
        out_rows.append(r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"rescaled {len(out_rows)} segments @ {args.fps} fps -> {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
