"""Split long OCR segments into smaller chunks so the AutoPrompter places
extra anchors *inside* long stable segments. The re-prompt density ablation
knob: AutoPrompter runs GD once per segment, so a 5,000-frame stable segment
gets exactly one anchor and SAM has to memory-track 5k frames between
re-prompts (drift). Splitting that segment into 5 × 1,000-frame chunks puts
4 extra anchors in there, persistent obj_ids carried via the prompter's IoU
matching across anchors.

Arm labels and _src/_sec are inherited / linearly interpolated per chunk;
start_i, end_i, n_frames are recomputed.

Usage:
  python -m tools.densify_segments --in seg.csv --out seg_dense.csv --gap 1000
  --gap 0 → no-op copy.
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
    p.add_argument("--gap", type=int, required=True,
                   help="Max anchor gap (frames). 0 = no densify.")
    args = p.parse_args(argv)

    src = Path(args.inp)
    if not src.exists():
        print(f"missing segments.csv: {src}", file=sys.stderr); return 2
    with open(src) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"empty segments.csv: {src}", file=sys.stderr); return 2

    fieldnames = list(rows[0].keys())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.gap <= 0:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
        print(f"copied {len(rows)} segments (gap=0, no densify)", file=sys.stderr)
        return 0

    out_rows: list[dict] = []
    for r in rows:
        si = int(r["start_i"]); ei = int(r["end_i"])
        length = ei - si + 1
        if length <= args.gap:
            out_rows.append(r); continue
        n_chunks = (length + args.gap - 1) // args.gap
        chunk = length / n_chunks
        # Provenance values to linearly interpolate (best-effort; missing cols ok).
        def _flt(k):
            try: return float(r[k])
            except (KeyError, ValueError, TypeError): return None
        ss, es = _flt("start_sec"), _flt("end_sec")
        src_s, src_e = _flt("start_src"), _flt("end_src")

        for c in range(n_chunks):
            s = si + int(round(c * chunk))
            e = (si + int(round((c + 1) * chunk)) - 1) if c < n_chunks - 1 else ei
            rr = dict(r)
            rr["start_i"] = s
            rr["end_i"] = e
            if "n_frames" in rr:
                rr["n_frames"] = e - s + 1
            denom = max(1, length - 1)
            fr_s = (s - si) / denom
            fr_e = (e - si) / denom
            if ss is not None and es is not None:
                rr["start_sec"] = round(ss + (es - ss) * fr_s, 1)
                rr["end_sec"]   = round(ss + (es - ss) * fr_e, 1)
            if src_s is not None and src_e is not None:
                rr["start_src"] = int(round(src_s + (src_e - src_s) * fr_s))
                rr["end_src"]   = int(round(src_s + (src_e - src_s) * fr_e))
            out_rows.append(rr)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(out_rows)
    print(f"densified {len(rows)} segments -> {len(out_rows)} (gap={args.gap})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
