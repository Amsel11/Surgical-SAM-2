"""Pipeline orchestrator. Entry point for `python -m pipeline run ...`.

Flow per invocation:
  1. Hydra resolves the YAML config (group defaults + CLI overrides +
     ${oc.env:...} interpolations from .env).
  2. Pydantic validates the resolved config.
  3. Resolve which videos this run covers (scope.videos against the manifest).
  4. For each video:
       a) resolve prompts (stage 1 — only manual_box wired for now)
       b) instantiate the model from MODEL_REGISTRY (stage 2)
       c) run inference -> writes masks/, overlay.mp4, log.json
       d) snapshot the config next to the outputs
       e) record an inference_runs row in the manifest
  5. Stage 3 (eval) lands later — gated behind cfg.stage3_eval.enabled.

Single-video mode for slurm arrays: pass `video_index=<i>` to process only
the i-th video in the resolved scope, so an `sbatch --array=0-N` works.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from pipeline.config import PipelineConfig
from pipeline.db import connect

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------


def dump_config_snapshot(
    cfg_dict: dict[str, Any],
    results_dir: Path,
    log_path: Path | None = None,
) -> Path:
    """Write the resolved config alongside a run's outputs for reproducibility.

    Drops `<results_dir>/_config.yaml` and `<results_dir>/_config.json`.
    If `log_path` points at an existing `log.json`, the config is also
    embedded under its `config` key so the snapshot survives even if
    `_config.yaml` is later separated from the result dir.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = results_dir / "_config.yaml"
    yaml_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))

    json_path = results_dir / "_config.json"
    json_path.write_text(json.dumps(cfg_dict, indent=2, default=str))

    if log_path is not None and Path(log_path).exists():
        log = json.loads(Path(log_path).read_text())
        log["config"] = cfg_dict
        log["config_snapshot_at"] = datetime.now(timezone.utc).isoformat()
        Path(log_path).write_text(json.dumps(log, indent=2, default=str))

    return yaml_path


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def resolve_video_scope(conn, scope) -> list[str]:
    """Return the list of video_ids this run should process."""
    if isinstance(scope.videos, list):
        return list(scope.videos)
    if scope.videos == "all":
        rows = conn.execute(
            "SELECT video_id FROM videos WHERE cohort = ? ORDER BY video_id",
            (scope.cohort,),
        ).fetchall()
        return [r["video_id"] for r in rows]
    if scope.videos == "qc_pass":
        # qc_results table lands in Phase E; until then, fail loudly.
        raise RuntimeError(
            "scope.videos='qc_pass' requires the qc_results table (Phase E)."
        )
    if scope.videos == "failed_inference":
        rows = conn.execute("""
            SELECT DISTINCT v.video_id FROM videos v
            JOIN inference_runs r ON r.prompt_set_id IN (
                SELECT prompt_set_id FROM prompt_sets WHERE video_id = v.video_id
            )
            WHERE v.cohort = ? AND r.status = 'failed'
            ORDER BY v.video_id
        """, (scope.cohort,)).fetchall()
        return [r["video_id"] for r in rows]
    raise ValueError(f"Unknown scope.videos: {scope.videos!r}")


# ---------------------------------------------------------------------------
# Stage 1 — prompts resolution (manual_box only for now; Phase D adds others)
# ---------------------------------------------------------------------------


def resolve_prompts_path(conn, video_id: str, seed: int, stage1) -> Path:
    """Look up (or generate, for on-demand methods) the prompts JSON.

    manual_box / manual_click: read pre-existing prompt_sets row. Fails loud
    if absent — the user is expected to have clicked first.

    dino: read pre-existing row if one is 'ready'; otherwise instantiate the
    GroundingDinoPrompter and run it on the fly. Inserts a prompt_sets row as
    a side effect.
    """
    if stage1.method == "manual_box":
        return _resolve_manual_box(conn, video_id, seed)
    if stage1.method == "dino":
        return _resolve_dino(conn, video_id, seed, stage1)
    raise NotImplementedError(
        f"Stage1 method {stage1.method!r} not yet wired. yolo/gt_box "
        "prompt strategies remain future work."
    )


def _resolve_manual_box(conn, video_id: str, seed: int) -> Path:
    row = conn.execute("""
        SELECT prompts_path, status FROM prompt_sets
        WHERE video_id = ? AND seed = ? AND prompt_method = ?
    """, (video_id, seed, "manual_box")).fetchone()
    if row is None:
        raise RuntimeError(f"No prompt_set for ({video_id}, seed={seed}, manual_box).")
    if row["status"] != "ready":
        raise RuntimeError(
            f"Prompt_set for {video_id} has status={row['status']!r}; expected 'ready'."
        )
    if not row["prompts_path"]:
        raise RuntimeError(f"prompts_path is empty for {video_id} in the manifest.")
    return Path(row["prompts_path"])


def _resolve_dino(conn, video_id: str, seed: int, stage1) -> Path:
    """Return an existing dino prompt_set's JSON, or generate one on demand."""
    row = conn.execute("""
        SELECT prompts_path, status FROM prompt_sets
        WHERE video_id = ? AND seed = ? AND prompt_method = ?
    """, (video_id, seed, "dino")).fetchone()
    if row is not None and row["status"] == "ready" and row["prompts_path"]:
        existing_path = Path(row["prompts_path"])
        if existing_path.exists():
            return existing_path
        # JSON went missing on disk; fall through and regenerate.

    from pipeline.prompts import build_prompter

    frames_row = conn.execute(
        "SELECT frames_dir FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    if frames_row is None:
        raise RuntimeError(f"Video {video_id} not in manifest.")
    prompter = build_prompter(stage1)
    return prompter.run(
        video_id=video_id,
        frames_dir=Path(frames_row["frames_dir"]),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Per-video runner
# ---------------------------------------------------------------------------


def run_one_video(
    *,
    video_id: str,
    config: PipelineConfig,
    cfg_dict: dict[str, Any],
    conn,
) -> dict[str, Any]:
    """Inference + bookkeeping for one video. Returns the log dict."""
    from pipeline.models import build_tracker  # lazy; pulls torch

    row = conn.execute(
        "SELECT frames_dir FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Video {video_id} not in manifest. Run pipeline cli scan-videos first.")
    frames_dir = Path(row["frames_dir"])
    if not frames_dir.exists():
        raise RuntimeError(f"frames_dir does not exist on disk: {frames_dir}")

    prompts_path = resolve_prompts_path(conn, video_id, config.scope.seed, config.stage1_prompting)

    results_dir = (
        Path(config.infrastructure.results_root) / config.name / video_id
        / f"seed_{config.scope.seed}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    tracker = build_tracker(config.stage2_inference)
    print(f"\n=== {video_id}: {tracker.REGISTRY_KEY} -> {results_dir} ===")
    log = tracker.run(
        video_id=video_id,
        frames_dir=frames_dir,
        prompts_json=prompts_path,
        results_dir=results_dir,
        src_fps=1.0,    # TODO: pull from videos.fps once that column is populated
    )

    dump_config_snapshot(cfg_dict, results_dir, log_path=results_dir / "log.json")
    print(f"Snapshotted config -> {results_dir / '_config.yaml'}")
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv(REPO_ROOT / ".env")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # video_index is a runtime selector for slurm-array single-video mode,
    # not part of the pipeline config schema. Pop before pydantic validates.
    video_index = cfg_dict.pop("video_index", None)
    config = PipelineConfig.model_validate(cfg_dict)

    print(f"\n=== Pipeline run: {config.name} ===")
    if config.description:
        print(config.description)

    conn = connect(config.infrastructure.manifest_db)
    videos = resolve_video_scope(conn, config.scope)
    if not videos:
        print("No videos in scope. Nothing to do.")
        return

    if video_index is not None:
        vi = int(video_index)
        if vi < 0 or vi >= len(videos):
            print(f"video_index={vi} out of range [0, {len(videos)})", file=sys.stderr)
            sys.exit(2)
        videos = [videos[vi]]
        print(f"Single-video mode: index {vi} -> {videos[0]}")
    else:
        print(f"Scope: {len(videos)} videos")

    failed = []
    for vid in videos:
        try:
            run_one_video(video_id=vid, config=config, cfg_dict=cfg_dict, conn=conn)
        except Exception as exc:
            import traceback
            print(f"FAILED {vid}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            failed.append((vid, str(exc)))
            if video_index is not None:
                # In slurm-array mode, surface the failure as a non-zero exit
                # so the array task is marked FAILED, not silent.
                raise

    if failed:
        print(f"\n{len(failed)} videos failed:")
        for v, e in failed:
            print(f"  {v}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
