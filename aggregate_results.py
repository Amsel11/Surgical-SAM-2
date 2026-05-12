#!/usr/bin/env python
"""Aggregate per-video log.json files into a single CSV + a few QA plots.

Usage:
    python aggregate_results.py --results-dir results --out results/_summary.csv

Reads every results/<video>/seed_<k>/log.json. For each (video, seed) emits
one CSV row with: n_frames, resolution, wall_seconds, fps, per-object mean
mask area, per-object frames_with_empty_mask, etc.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def iter_logs(results_dir: Path):
    for log_path in sorted(results_dir.glob('*/seed_*/log.json')):
        try:
            with open(log_path) as fp:
                obj = json.load(fp)
        except Exception as exc:
            print(f'skip {log_path}: {exc}')
            continue
        video = log_path.parent.parent.name
        seed = int(log_path.parent.name.split('_')[-1])
        yield video, seed, log_path, obj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default='results/_summary.csv')
    args = ap.parse_args()

    rows = []
    for video, seed, _, log in iter_logs(Path(args.results_dir)):
        W, H = log.get('resolution', [None, None])
        n = log.get('n_frames', 0)
        wall = log.get('wall_seconds')
        mean_area = log.get('mean_mask_area_px', {})
        empty = log.get('frames_with_empty_mask', {})
        obj_ids = sorted({*mean_area.keys(), *empty.keys()})

        # Per-object summary: dump the union as JSON strings; also flatten the
        # first two objects into dedicated columns for easy plotting.
        row = {
            'video': video,
            'seed': seed,
            'n_frames': n,
            'width': W, 'height': H,
            'wall_seconds': wall,
            'fps': (n / wall) if (wall and n) else None,
            'n_prompt_frames': len(log.get('prompts_frames', [])),
            'n_prompt_calls': log.get('n_prompt_calls', 0),
            'n_objects': len(obj_ids),
            'mean_mask_area_px_json': json.dumps(mean_area, sort_keys=True),
            'frames_with_empty_mask_json': json.dumps(empty, sort_keys=True),
        }
        for i, oid in enumerate(obj_ids[:4], start=1):
            row[f'obj{i}_id'] = int(oid)
            row[f'obj{i}_mean_area'] = float(mean_area.get(oid, 0))
            row[f'obj{i}_empty_frames'] = int(empty.get(oid, 0))
            row[f'obj{i}_pct_lost'] = 100.0 * empty.get(oid, 0) / n if n else None
        rows.append(row)

    if not rows:
        print(f'No log.json under {args.results_dir}/*/seed_*/. Nothing to aggregate.')
        return

    df = pd.DataFrame(rows).sort_values(['video', 'seed'])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote {len(df)} rows to {out_path}')
    print()
    print('Per-video summary (first 20):')
    summary = df.groupby('video').agg(
        seeds=('seed', 'count'),
        n_frames=('n_frames', 'first'),
        mean_wall=('wall_seconds', 'mean'),
        mean_fps=('fps', 'mean'),
    ).head(20)
    print(summary.to_string())


if __name__ == '__main__':
    main()
