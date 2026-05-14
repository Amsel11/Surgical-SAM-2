"""Pipeline orchestrator. Entry point for `python -m pipeline run ...`.

Flow:
  1. Hydra resolves the YAML config (composes group defaults, applies CLI
     overrides, resolves ${oc.env:...} interpolations).
  2. Pydantic validates the resolved config — typed errors point at the
     exact bad field.
  3. Dispatch through registries (Phase C+) to the actual work.
  4. After each run, snapshot the resolved config next to outputs AND
     embed it in log.json — so any result is fully reproducible from its
     artifacts alone.

Phase B scope: steps 1-2 + the snapshot helper. Step 3 is stubbed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def dump_config_snapshot(
    cfg_dict: dict[str, Any],
    results_dir: Path,
    log_path: Path | None = None,
) -> Path:
    """Write the resolved config alongside a run's outputs for reproducibility.

    Drops `<results_dir>/_config.yaml` and `<results_dir>/_config.json`.
    Caller is responsible for `results_dir` existing. If `log_path` points at
    an existing `log.json`, the config is also embedded under its `config`
    key (so the snapshot survives even if `_config.yaml` is later separated
    from the result dir).
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


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    # .env populates os.environ; ${oc.env:VAR} in YAML resolves against it.
    load_dotenv(REPO_ROOT / ".env")

    # OmegaConf -> plain dict -> pydantic validation.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = PipelineConfig.model_validate(cfg_dict)

    print(f"\n=== Pipeline run: {config.name} ===")
    if config.description:
        print(config.description)
    print()
    print(config.model_dump_json(indent=2))
    print()
    print("[Phase B] orchestrator stub — model + prompt registries land in Phase C+.")
    print(f"           dump_config_snapshot() is wired and ready for results_dir output.")


if __name__ == "__main__":
    main()
