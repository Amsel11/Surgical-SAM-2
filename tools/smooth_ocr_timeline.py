"""Smooth an OCR arm-slot timeline (segments.csv) by killing flicker.

Per-frame OCR (esp. PaddleOCR) produces spurious segment boundaries between
*identical* arm configurations, plus single-frame misreads. Raw, that inflates
the segment count (e.g. 192 segments for one stable video), which would make the
auto prompter anchor at far too many frames. This collapses the timeline so it
reflects real instrument-config changes.

Two operations, applied at the SEGMENT level (so boundaries stay on original
segment edges and the start_src/end_src -> filename mapping remains valid):
  1. merge adjacent segments with an identical arm tuple, and
  2. iteratively absorb any segment shorter than `min_seconds` into its longer
     neighbor (adopting that neighbor's arms) — debouncing transient misreads
     and boundary flicker — then re-collapse.

min duration is given in SECONDS and converted to frames via the per-video fps
inferred from the timeline's own sec<->frame span, so it works across the
cohort's varying frame rates.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ARM_COLS = ("arm1", "arm2", "arm3", "arm4")
FIELDS = ["start_i", "end_i", "n_frames", "start_src", "end_src",
          "start_sec", "end_sec", "arm1", "arm2", "arm3", "arm4"]


def _norm(s: str | None) -> str:
    s = (s or "").strip().lower()
    return "" if s == "camera" else s


def load_segments(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "start_i":   int(r["start_i"]),
                "end_i":     int(r["end_i"]),
                "start_src": int(r["start_src"]),
                "end_src":   int(r["end_src"]),
                "start_sec": float(r["start_sec"]),
                "end_sec":   float(r["end_sec"]),
                "arms":      tuple(_norm(r.get(c)) for c in ARM_COLS),
            })
    return rows


def _dur(seg: dict) -> int:
    return seg["end_i"] - seg["start_i"] + 1


def _collapse_adjacent(rows: list[dict]) -> list[dict]:
    """Merge consecutive segments that share the same arm tuple."""
    out: list[dict] = []
    for r in rows:
        if out and out[-1]["arms"] == r["arms"]:
            p = out[-1]
            p["end_i"], p["end_src"], p["end_sec"] = r["end_i"], r["end_src"], r["end_sec"]
        else:
            out.append(dict(r))
    return out


def fill_blanks_per_slot(rows: list[dict]) -> list[dict]:
    """Forward-fill each arm slot across blank reads.

    The da Vinci strip shows MOUNTED instruments; a slot reading empty is
    usually OCR transiently failing to read a still-mounted tool (occlusion /
    glare), not a real unmount. So per slot, an empty reading inherits the last
    non-empty instrument seen. Slots that are genuinely never populated (e.g.
    the camera arm) stay empty. Assumes "once mounted, stays mounted" — fine for
    this cohort's stable configs; a true mid-video unmount would be masked, but
    that only leaves a stale track the segmenter/dedup can shed.
    """
    rows = [dict(r) for r in rows]
    last = ["", "", "", ""]
    for r in rows:
        arms = list(r["arms"])
        for s in range(4):
            if arms[s] == "":
                arms[s] = last[s]
            else:
                last[s] = arms[s]
        r["arms"] = tuple(arms)
    return rows


def _merge_span(a: dict, b: dict, keep_arms: tuple) -> dict:
    """Span two adjacent segments into one carrying `keep_arms`."""
    lo, hi = (a, b) if a["start_i"] <= b["start_i"] else (b, a)
    return {
        "start_i": lo["start_i"], "end_i": hi["end_i"],
        "start_src": lo["start_src"], "end_src": hi["end_src"],
        "start_sec": lo["start_sec"], "end_sec": hi["end_sec"],
        "arms": keep_arms,
    }


def infer_fps(rows: list[dict]) -> float:
    span_f = rows[-1]["end_i"] - rows[0]["start_i"]
    span_s = rows[-1]["end_sec"] - rows[0]["start_sec"]
    return span_f / span_s if span_s > 0 else 30.0


def smooth(rows: list[dict], min_frames: int, fill_blanks: bool = True) -> list[dict]:
    if fill_blanks:
        rows = fill_blanks_per_slot(rows)
    rows = _collapse_adjacent([dict(r) for r in rows])
    while len(rows) > 1:
        i = min(range(len(rows)), key=lambda k: _dur(rows[k]))
        if _dur(rows[i]) >= min_frames:
            break
        left = rows[i - 1] if i > 0 else None
        right = rows[i + 1] if i < len(rows) - 1 else None
        if left and right:
            keeper = left if _dur(left) >= _dur(right) else right
        else:
            keeper = left or right
        ki = i - 1 if keeper is left else i + 1
        merged = _merge_span(rows[i], keeper, keeper["arms"])
        lo, hi = min(i, ki), max(i, ki)
        rows = rows[:lo] + [merged] + rows[hi + 1:]
        rows = _collapse_adjacent(rows)
    return rows


def write_segments(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in rows:
            a = r["arms"]
            w.writerow([
                r["start_i"], r["end_i"], _dur(r),
                r["start_src"], r["end_src"],
                f"{r['start_sec']:.1f}", f"{r['end_sec']:.1f}",
                a[0], a[1], a[2], a[3],
            ])


def smooth_file(in_csv: Path, out_csv: Path, min_seconds: float,
                fill_blanks: bool = True) -> tuple[int, int]:
    rows = load_segments(in_csv)
    if not rows:
        write_segments(rows, out_csv)
        return 0, 0
    min_frames = max(1, round(min_seconds * infer_fps(rows)))
    smoothed = smooth(rows, min_frames, fill_blanks=fill_blanks)
    write_segments(smoothed, out_csv)
    return len(rows), len(smoothed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-csv", type=Path, help="Single segments.csv to smooth")
    ap.add_argument("--out-csv", type=Path)
    ap.add_argument("--in-dir", type=Path, help="Root of <video>/segments.csv to batch-smooth")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--min-seconds", type=float, default=5.0,
                    help="Segments shorter than this (converted to frames via "
                         "per-video fps) are absorbed as flicker. Default 5s.")
    ap.add_argument("--no-fill-blanks", action="store_true",
                    help="Disable per-slot forward-fill of blank reads "
                         "(default: fill, treating blanks as OCR misses).")
    args = ap.parse_args()
    fill = not args.no_fill_blanks

    if args.in_csv:
        n0, n1 = smooth_file(args.in_csv, args.out_csv, args.min_seconds, fill)
        print(f"{args.in_csv}: {n0} -> {n1} segments")
        return 0
    if args.in_dir:
        for d in sorted(args.in_dir.iterdir()):
            seg = d / "segments.csv"
            if not seg.exists():
                continue
            out = args.out_dir / d.name / "segments.csv"
            n0, n1 = smooth_file(seg, out, args.min_seconds, fill)
            print(f"{d.name}: {n0} -> {n1}")
        return 0
    ap.error("provide --in-csv/--out-csv or --in-dir/--out-dir")


if __name__ == "__main__":
    raise SystemExit(main())
