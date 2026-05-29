"""Shared OCR-timeline parser: instrument first-appearance events.

Reads a per-video `segments.csv` (produced by tools/ocr_slots_timeline_paddle.py)
and returns the frames where each instrument *first appears* on a da Vinci arm.
Used by the OCR-driven clicker (smart anchor frames) and the automatic
detect→track pipeline, so both read appearance events from one source of truth.

segments.csv columns:
    start_i,end_i,n_frames,start_src,end_src,start_sec,end_sec,arm1,arm2,arm3,arm4
Each row is a time segment; arm{1..4} is the instrument mounted on that arm
(or empty / 'camera'). start_i/end_i are LOADER indices (positions in
pipeline.io.frame_files_ordered), the same index the SAM mask stems use.

An "appearance event" fires the first time a real instrument shows up on an arm,
or when it changes to a different instrument. Empty / 'camera' values are ignored
and do NOT count as the tool leaving — so tool→camera→same-tool yields one event,
while tool A→camera→tool B yields A then B.

The appearance frame is offset a few frames PAST the OCR change (default 15), so
the tool is fully in view rather than just entering — OCR 'mounted' precedes
'fully visible'. Clamped to stay within the segment and the video.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ARM_COLS = ("arm1", "arm2", "arm3", "arm4")
DEFAULT_IGNORE = ("", "camera", "none", "empty")


def _clean(v: str | None) -> str:
    return (v or "").strip()


def appearance_events(
    segments_csv: str | Path,
    offset: int = 15,
    n_frames: int | None = None,
    ignore_values: tuple[str, ...] = DEFAULT_IGNORE,
    dedupe: bool = True,
) -> list[dict]:
    """Parse segments.csv → sorted list of instrument first-appearance events.

    Each event: {arm, arm_idx, instrument, ocr_change_frame, appearance_frame}.
    `offset` shifts the appearance frame past the OCR change (clamped to the
    segment's end_i and n_frames-1). `dedupe` keeps only the first appearance of
    each (arm, instrument) pair.
    """
    rows = []
    with open(segments_csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["_start"] = int(r["start_i"])
                r["_end"] = int(r["end_i"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["_start"])

    ignore = {v.lower() for v in ignore_values}
    prev_real: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    events: list[dict] = []
    for r in rows:
        for ai, arm in enumerate(ARM_COLS, start=1):
            val = _clean(r.get(arm))
            if val.lower() in ignore:
                continue  # not a real instrument; don't reset prev_real
            if prev_real.get(arm) == val:
                continue  # same instrument still mounted — no new appearance
            prev_real[arm] = val
            if dedupe and (arm, val) in seen:
                continue
            seen.add((arm, val))
            start_i = r["_start"]
            app = start_i + offset
            app = min(app, r["_end"])
            if n_frames is not None:
                app = min(app, n_frames - 1)
            app = max(app, start_i)
            events.append({
                "arm": arm,
                "arm_idx": ai,
                "instrument": val,
                "ocr_change_frame": start_i,
                "appearance_frame": app,
            })
    events.sort(key=lambda e: (e["appearance_frame"], e["arm_idx"]))
    return events


def appearance_events_for_video(
    ocr_root: str | Path, video_id: str, **kwargs
) -> list[dict]:
    """Convenience: locate <ocr_root>/<video_id>/segments.csv and parse it.

    Returns [] if the segments.csv is missing.
    """
    seg = Path(ocr_root) / video_id / "segments.csv"
    if not seg.exists():
        return []
    return appearance_events(seg, **kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--segments", type=Path, help="Path to a segments.csv.")
    p.add_argument("--ocr-root", type=Path, help="OCR root; use with --video-id.")
    p.add_argument("--video-id")
    p.add_argument("--offset", type=int, default=15)
    p.add_argument("--n-frames", type=int, default=None)
    p.add_argument("--no-dedupe", action="store_true")
    args = p.parse_args(argv)

    if args.segments:
        events = appearance_events(args.segments, offset=args.offset,
                                   n_frames=args.n_frames, dedupe=not args.no_dedupe)
    elif args.ocr_root and args.video_id:
        events = appearance_events_for_video(
            args.ocr_root, args.video_id, offset=args.offset,
            n_frames=args.n_frames, dedupe=not args.no_dedupe)
    else:
        print("Provide --segments OR (--ocr-root and --video-id).", file=sys.stderr)
        return 2

    print(f"{len(events)} appearance event(s):")
    for e in events:
        print(f"  frame {e['appearance_frame']:>7}  (ocr change @ {e['ocr_change_frame']:>7})  "
              f"arm{e['arm_idx']}  {e['instrument']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
