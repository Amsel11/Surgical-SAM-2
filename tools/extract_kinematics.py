"""Extract per-frame, per-instrument kinematic features from SAM mask outputs.

Phase 6 of the GD fine-tune plan. The headline metric of the ablation
('eval on kinematic signal not DICE' per memory) — turns palette-PNG mask
sequences into per-obj centroid trajectories + three aggregate metrics:

  identity_switches : frames where (cx,cy) jumps more than --jump-threshold-px
                      between consecutive frames for the SAME obj_id. Most
                      sensitive to upstream identity errors (the bottleneck
                      the FT-GD ablation is targeting).

  track_continuity  : fraction of frames in the trusted span where the obj_id
                      has a non-empty mask. Trusted span is currently the
                      full mask range; a follow-up will join scan_masks.py
                      output (first_anchor / last_anchor / in_span_rate).

  kinematic_stability : median absolute deviation of frame-to-frame speed.
                      Lower = smoother trajectory.

Outputs:
  local_results/kinematics/<run>/<video>.csv : frame-level rows
      (frame_idx, obj_id, cx, cy, area, dx, dy, speed)
  local_results/kinematics/<run>/summary.csv : one row per video with the
      three aggregate metrics — the headline plot is paired
      identity_switches[FT] vs identity_switches[ZS] across the test split.

Usage:
    python -m tools.extract_kinematics --run sam3_oob_whip_v1
    python -m tools.extract_kinematics --run dino_zs_whip_v1 --video DC_whip_11609423
    python -m tools.extract_kinematics --run dino_ft_whip_v1 --jump-threshold-px 120
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.db import REPO_ROOT, connect

DEFAULT_KINEMATICS_OUT = REPO_ROOT / "local_results" / "kinematics"
DEFAULT_JUMP_THRESHOLD_PX = 150.0  # tuned for 1080p; pass --jump-threshold-px to override
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def parse_frame_idx(name: str) -> int:
    """Mask filenames are zero-padded ints (per pipeline.io.list_frames convention)."""
    return int(Path(name).stem)


def load_palette_mask(path: Path) -> np.ndarray:
    """Read a palette-mode PNG → 2D label array (background=0, obj_ids > 0)."""
    with Image.open(path) as im:
        return np.array(im)


def per_frame_centroids(
    mask: np.ndarray, min_area_px: int = 1
) -> dict[int, tuple[float, float, int]]:
    """Return {obj_id: (cx, cy, area_px)} for every present obj_id.

    Centroid is the arithmetic mean of (x, y) over mask pixels — fast and
    robust for a single connected region per obj_id. Multiple disconnected
    components for the same obj_id collapse to a single average centroid
    (acceptable for kinematics; would be wrong for instance counting).
    """
    out: dict[int, tuple[float, float, int]] = {}
    for oid in np.unique(mask):
        if oid == 0:
            continue
        ys, xs = np.where(mask == oid)
        if ys.size < min_area_px:
            continue
        out[int(oid)] = (float(xs.mean()), float(ys.mean()), int(ys.size))
    return out


def find_runs(results_root: Path, run_name: str) -> list[tuple[str, Path]]:
    """Return [(video_id, masks_dir), ...] for every video under results/<run>/."""
    run_dir = results_root / run_name
    if not run_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for video_dir in sorted(run_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        masks_candidates = (
            sorted(video_dir.glob("seed_*/masks")) + sorted(video_dir.glob("masks"))
        )
        if not masks_candidates:
            continue
        out.append((video_dir.name, masks_candidates[0]))
    return out


def load_trusted_span(conn, video_id: str) -> tuple[int, int] | None:
    """Hook for scan_masks.py output (first_anchor, last_anchor) lookup.

    TODO: persist scan_masks_seed.sh output to the manifest and query here.
    Returns None today, which makes extract_one fall back to the full
    mask range — fine for v1 but inflates continuity for runs where SAM3
    propagates outside the labelled span.
    """
    return None


def extract_one(
    video_id: str,
    masks_dir: Path,
    jump_threshold_px: float,
    out_dir: Path,
    trusted_span: tuple[int, int] | None,
) -> dict:
    """Walk masks/*.png for one video; write per-frame CSV; return summary."""
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: parse_frame_idx(p.name))
    if not mask_files:
        return {
            "video_id": video_id,
            "n_frames": 0,
            "n_objs": 0,
            "identity_switches": 0,
            "track_continuity_mean": 0.0,
            "kinematic_stability_mad_speed": 0.0,
            "trusted_span_start": -1,
            "trusted_span_end": -1,
        }

    obj_records: dict[int, list[tuple[int, float, float, int]]] = defaultdict(list)
    for mp in mask_files:
        frame_idx = parse_frame_idx(mp.name)
        for oid, (cx, cy, area) in per_frame_centroids(load_palette_mask(mp)).items():
            obj_records[oid].append((frame_idx, cx, cy, area))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{video_id}.csv"
    n_switches = 0
    speed_samples: list[float] = []
    continuity_per_obj: list[float] = []

    span_start = trusted_span[0] if trusted_span else parse_frame_idx(mask_files[0].name)
    span_end = trusted_span[1] if trusted_span else parse_frame_idx(mask_files[-1].name)
    span_length = max(1, span_end - span_start + 1)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "obj_id", "cx", "cy", "area", "dx", "dy", "speed"])
        for oid, recs in sorted(obj_records.items()):
            recs.sort(key=lambda r: r[0])
            in_span_frames = sum(1 for r in recs if span_start <= r[0] <= span_end)
            continuity_per_obj.append(in_span_frames / span_length)
            for i, (fi, cx, cy, area) in enumerate(recs):
                if i == 0:
                    dx = dy = 0.0
                    speed = 0.0
                else:
                    pfi, pcx, pcy, _ = recs[i - 1]
                    dx = cx - pcx
                    dy = cy - pcy
                    speed = float(np.hypot(dx, dy))
                    if speed > jump_threshold_px:
                        n_switches += 1
                    speed_samples.append(speed)
                w.writerow([fi, oid, f"{cx:.2f}", f"{cy:.2f}", area,
                            f"{dx:.2f}", f"{dy:.2f}", f"{speed:.2f}"])

    if speed_samples:
        arr = np.array(speed_samples)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
    else:
        mad = 0.0

    return {
        "video_id": video_id,
        "n_frames": len(mask_files),
        "n_objs": len(obj_records),
        "identity_switches": n_switches,
        "track_continuity_mean": (
            sum(continuity_per_obj) / len(continuity_per_obj)
            if continuity_per_obj else 0.0
        ),
        "kinematic_stability_mad_speed": mad,
        "trusted_span_start": span_start,
        "trusted_span_end": span_end,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True,
                   help="Run name (= experiment 'name' field = results/<run>/ dirname).")
    p.add_argument("--video", help="Restrict to one video_id (default: all in the run).")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                   help=f"Root of results dirs (default {DEFAULT_RESULTS_ROOT}).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Per-video CSV output dir "
                        "(default: local_results/kinematics/<run>/).")
    p.add_argument("--jump-threshold-px", type=float, default=DEFAULT_JUMP_THRESHOLD_PX,
                   help=f"Centroid jump > this px between consecutive frames "
                        f"counts as an identity switch "
                        f"(default {DEFAULT_JUMP_THRESHOLD_PX:.0f}, tuned for 1080p).")
    p.add_argument("--db", help="Manifest path override (for trusted-span lookup).")
    args = p.parse_args(argv)

    run_pairs = find_runs(args.results_root, args.run)
    if not run_pairs:
        print(f"No mask dirs under {args.results_root / args.run}/", file=sys.stderr)
        return 2
    if args.video:
        run_pairs = [(v, p) for v, p in run_pairs if v == args.video]
        if not run_pairs:
            print(f"No mask dir for {args.video!r} in {args.run!r}", file=sys.stderr)
            return 2

    out_dir = args.out_dir or (DEFAULT_KINEMATICS_OUT / args.run)
    conn = connect(args.db)

    print(f"--- extract_kinematics: run={args.run}", file=sys.stderr)
    print(f"  videos: {len(run_pairs)}", file=sys.stderr)
    print(f"  jump threshold: {args.jump_threshold_px:.0f} px\n", file=sys.stderr)

    summaries: list[dict] = []
    for video_id, masks_dir in run_pairs:
        trusted_span = load_trusted_span(conn, video_id)
        summary = extract_one(
            video_id=video_id,
            masks_dir=masks_dir,
            jump_threshold_px=args.jump_threshold_px,
            out_dir=out_dir,
            trusted_span=trusted_span,
        )
        summaries.append(summary)
        print(
            f"  {video_id}: {summary['n_frames']} frames, "
            f"{summary['n_objs']} objs, "
            f"switches={summary['identity_switches']}, "
            f"cont={summary['track_continuity_mean']:.3f}, "
            f"MAD speed={summary['kinematic_stability_mad_speed']:.2f}",
            file=sys.stderr,
        )

    summary_path = out_dir / "summary.csv"
    if summaries:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            for s in summaries:
                w.writerow(s)
    print(f"\nWrote {len(summaries)} summaries to {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
