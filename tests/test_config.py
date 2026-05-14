"""Smoke tests for Hydra + pydantic config plumbing.

These tests don't load SAM2 or hit GPFS; they just verify the config layer is
correctly wired so a broken YAML or schema change fails fast in CI rather
than surfacing at job-launch time on bp.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pipeline.config import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def _compose(overrides: list[str]):
    """Helper: compose the default config with given CLI-style overrides."""
    # initialize_config_dir wants an absolute path. version_base=None silences
    # the deprecation warning about Hydra 1.1 vs 1.2 defaults handling.
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def test_default_config_validates(monkeypatch):
    """Hydra resolves config.yaml + pydantic accepts the result with a name."""
    monkeypatch.setenv("BP_FRAMES_ROOT", "/tmp/frames")
    monkeypatch.setenv("BP_RESULTS_ROOT", "/tmp/results")
    monkeypatch.setenv("SURGSAM_MANIFEST", "/tmp/manifest.db")
    monkeypatch.setenv("BP_REPO", "/tmp/repo")

    cfg_dict = _compose(["name=test_run"])
    config = PipelineConfig.model_validate(cfg_dict)

    assert config.name == "test_run"
    assert config.scope.cohort == "whip"
    assert config.stage1_prompting.method == "manual_box"
    assert config.stage2_inference.model == "surgsam2"
    assert config.stage2_inference.bidirectional is True
    assert config.infrastructure.frames_root == "/tmp/frames"


def test_surgsam2_oob_whip_experiment(monkeypatch):
    """The named experiment composes and validates."""
    monkeypatch.setenv("BP_FRAMES_ROOT", "/tmp/frames")
    monkeypatch.setenv("BP_RESULTS_ROOT", "/tmp/results")
    monkeypatch.setenv("SURGSAM_MANIFEST", "/tmp/manifest.db")
    monkeypatch.setenv("BP_REPO", "/tmp/repo")

    cfg_dict = _compose(["+experiment=surgsam2_oob_whip"])
    config = PipelineConfig.model_validate(cfg_dict)

    assert config.name == "surgsam2_oob_whip_v1"
    assert config.stage2_inference.model == "surgsam2"


def test_sam2_oob_whip_experiment(monkeypatch):
    """Swapping to sam2 OOB via experiment file works end-to-end."""
    monkeypatch.setenv("BP_FRAMES_ROOT", "/tmp/frames")
    monkeypatch.setenv("BP_RESULTS_ROOT", "/tmp/results")
    monkeypatch.setenv("SURGSAM_MANIFEST", "/tmp/manifest.db")
    monkeypatch.setenv("BP_REPO", "/tmp/repo")

    cfg_dict = _compose(["+experiment=sam2_oob_whip"])
    config = PipelineConfig.model_validate(cfg_dict)

    assert config.name == "sam2_oob_whip_v1"
    assert config.stage2_inference.model == "sam2"


def test_cli_override(monkeypatch):
    """A direct CLI override changes the leaf value."""
    monkeypatch.setenv("BP_FRAMES_ROOT", "/tmp/frames")
    monkeypatch.setenv("BP_RESULTS_ROOT", "/tmp/results")
    monkeypatch.setenv("SURGSAM_MANIFEST", "/tmp/manifest.db")
    monkeypatch.setenv("BP_REPO", "/tmp/repo")

    cfg_dict = _compose([
        "name=cli_override_test",
        "stage2_inference.bidirectional=false",
        "stage2_inference.device=cuda:1",
    ])
    config = PipelineConfig.model_validate(cfg_dict)

    assert config.stage2_inference.bidirectional is False
    assert config.stage2_inference.device == "cuda:1"


def test_missing_name_fails():
    """The mandatory `name` field must be set; otherwise validation fails."""
    with pytest.raises(Exception):
        cfg_dict = _compose([])  # no name given
        PipelineConfig.model_validate(cfg_dict)


def test_unknown_field_rejected(monkeypatch):
    """extra='forbid' on every model catches typos in YAML."""
    monkeypatch.setenv("BP_FRAMES_ROOT", "/tmp/frames")
    monkeypatch.setenv("BP_RESULTS_ROOT", "/tmp/results")
    monkeypatch.setenv("SURGSAM_MANIFEST", "/tmp/manifest.db")

    cfg_dict = _compose(["name=t", "+typo_field=oops"])
    with pytest.raises(Exception):
        PipelineConfig.model_validate(cfg_dict)
