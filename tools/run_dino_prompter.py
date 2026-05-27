"""Run the Grounding DINO Stage-1 prompter on one or more videos.

Invokes pipeline.prompts.GroundingDinoPrompter directly so the prompter can
be called from .gdino_venv (transformers + torch) without dragging the SAM
trackers in. Inserts/UPSERTs a prompt_sets row + prompt_objects rows for
each (video, seed, prompt_method='dino'); the SAM 3 orchestrator (running
in .sam3_venv) then picks them up via _resolve_dino's "existing prompt_set"
path with no further work.

Usage:
    python -m tools.run_dino_prompter --video DC_whip_11609423
    python -m tools.run_dino_prompter --video DC_whip_11609423 --seed 2
    python -m tools.run_dino_prompter --all-cohort whip
    python -m tools.run_dino_prompter --video <id> --box-threshold 0.20 --text-threshold 0.15
    python -m tools.run_dino_prompter --video <id> --checkpoint checkpoints/gdino_whip_ft_v1.pth

On bp:
    ssh bp 'cd /gpfs/data/oermannlab/users/schula12/Surgical-SAM-2 \\
            && source .gdino_venv/bin/activate \\
            && python -m tools.run_dino_prompter --video DC_whip_11609423'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.db import connect
from pipeline.prompts.dino import GroundingDinoPrompter


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--video", help="Single video_id to prompt.")
    sel.add_argument("--all-cohort", metavar="COHORT",
                     help="Run on every video in this cohort (e.g. 'whip').")
    p.add_argument("--seed", type=int, default=1,
                   help="prompt_sets.seed to write under (default 1).")
    p.add_argument("--checkpoint", default=None,
                   help="Path to FT'd weights (overrides HF zero-shot default).")
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny",
                   help="HF model id when --checkpoint is not set "
                        "(default 'IDEA-Research/grounding-dino-tiny').")
    p.add_argument("--box-threshold", type=float, default=0.35,
                   help="GD box score threshold (default 0.35).")
    p.add_argument("--text-threshold", type=float, default=0.25,
                   help="GD text score threshold (default 0.25).")
    p.add_argument("--vocab", type=Path, default=None,
                   help="Override path to vocab JSON "
                        "(default: configs/cardiac_whip_vocab.json).")
    p.add_argument("--db", help="Manifest path override.")
    p.add_argument("--device", default="cuda",
                   help="torch device (default cuda; falls back to cpu if "
                        "CUDA unavailable).")
    p.add_argument("--prompts-dir", type=Path, default=None,
                   help="Override the prompts JSON output dir.")
    args = p.parse_args(argv)

    conn = connect(args.db)
    if args.video:
        video_ids = [args.video]
    else:
        rows = conn.execute(
            "SELECT video_id FROM videos WHERE cohort = ? ORDER BY video_id",
            (args.all_cohort,),
        ).fetchall()
        video_ids = [r["video_id"] for r in rows]
        if not video_ids:
            print(f"No videos in cohort {args.all_cohort!r}", file=sys.stderr)
            return 2

    prompter = GroundingDinoPrompter(
        model_id=args.model_id,
        checkpoint=args.checkpoint,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        vocab_path=args.vocab,
    )

    failures: list[tuple[str, str]] = []
    for vid in video_ids:
        row = conn.execute(
            "SELECT frames_dir FROM videos WHERE video_id = ?", (vid,)
        ).fetchone()
        if row is None:
            print(f"✗ {vid}: not in manifest", file=sys.stderr)
            failures.append((vid, "not in manifest"))
            continue
        try:
            out_path = prompter.run(
                video_id=vid,
                frames_dir=Path(row["frames_dir"]),
                seed=args.seed,
                prompts_dir=args.prompts_dir,
            )
            print(f"✓ {vid} -> {out_path}", file=sys.stderr)
        except Exception as exc:
            print(f"✗ {vid}: {exc}", file=sys.stderr)
            failures.append((vid, str(exc)))

    n_ok = len(video_ids) - len(failures)
    print(f"\nDone: {n_ok}/{len(video_ids)} succeeded", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
