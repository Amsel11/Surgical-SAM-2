"""Pipeline orchestrator. Entry point for `python -m pipeline run ...`.

Flow:
  1. Hydra resolves the YAML config (composes group defaults, applies CLI
     overrides, resolves ${oc.env:...} interpolations).
  2. Pydantic validates the resolved config — typed errors point at the
     exact bad field.
  3. Dispatch through registries (Phase C+) to the actual work.

Phase B scope: steps 1-2 only. Step 3 is stubbed — we print the resolved
config and exit. Real orchestration lands when the model + prompt
registries are wired up.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    # .env populates os.environ; ${oc.env:VAR_NAME} in YAML resolves against it.
    load_dotenv(REPO_ROOT / ".env")

    # OmegaConf -> plain dict -> pydantic validation
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    config = PipelineConfig.model_validate(cfg_dict)

    print(f"\n=== Pipeline run: {config.name} ===")
    if config.description:
        print(config.description)
    print()
    print(config.model_dump_json(indent=2))
    print()
    print("[Phase B] orchestrator stub — registries land in Phase C+.")


if __name__ == "__main__":
    main()
