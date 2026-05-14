"""Smoke tests for Hydra + pydantic config plumbing.

These tests don't load SAM2 or hit GPFS; they just verify the config layer is
correctly wired so a broken YAML or schema change fails fast in CI rather
than surfacing at job-launch time on bp.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pipeline.config import PipelineConfig
from pipeline.run import dump_config_snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch):
    """Stub all .env vars referenced by configs/. Applied to every test."""
    monkeypatch.setenv("BP_FRAMES_ROOT", "/tmp/frames")
    monkeypatch.setenv("BP_RESULTS_ROOT", "/tmp/results")
    monkeypatch.setenv("SURGSAM_MANIFEST", "/tmp/manifest.db")
    monkeypatch.setenv("BP_REPO", "/tmp/repo")


def _compose(overrides: list[str]) -> dict:
    """Helper: compose the default config with given CLI-style overrides."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


# ----------------------------------------------------------------------------
# Default config
# ----------------------------------------------------------------------------


def test_default_config_validates():
    """The grand config.yaml composes and pydantic accepts it with a name."""
    cfg_dict = _compose(["name=test_run"])
    config = PipelineConfig.model_validate(cfg_dict)

    assert config.name == "test_run"
    assert config.scope.cohort == "whip"
    assert config.stage1_prompting.method == "manual_box"
    assert config.stage2_inference.model == "surgsam2"
    assert config.stage2_inference.bidirectional is True
    assert config.infrastructure.frames_root == "/tmp/frames"


# ----------------------------------------------------------------------------
# Each experiment file composes and validates
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("experiment,expected_model,expected_name", [
    ("sam2_oob_whip",      "sam2",             "sam2_oob_whip_v1"),
    ("surgsam2_oob_whip",  "surgsam2",         "surgsam2_oob_whip_v1"),
    ("surgsam2_whip_ft",   "surgsam2_whip_ft", "surgsam2_whip_ft_v1"),
    ("sam3_oob_whip",      "sam3",             "sam3_oob_whip_v1"),
    ("sam3_whip_ft",       "sam3_whip_ft",     "sam3_whip_ft_v1"),
])
def test_experiment_composes(experiment, expected_model, expected_name):
    cfg_dict = _compose([f"+experiment={experiment}"])
    config = PipelineConfig.model_validate(cfg_dict)
    assert config.name == expected_name
    assert config.stage2_inference.model == expected_model


# ----------------------------------------------------------------------------
# CLI overrides
# ----------------------------------------------------------------------------


def test_cli_override_leaf_value():
    """A direct CLI override changes a leaf value."""
    cfg_dict = _compose([
        "name=cli_override_test",
        "stage2_inference.bidirectional=false",
        "stage2_inference.device=cuda:1",
        "scope.seed=2",
    ])
    config = PipelineConfig.model_validate(cfg_dict)

    assert config.stage2_inference.bidirectional is False
    assert config.stage2_inference.device == "cuda:1"
    assert config.scope.seed == 2


def test_cli_override_scope_videos_as_list():
    """scope.videos accepts an explicit list of video_ids."""
    cfg_dict = _compose([
        "name=t",
        "scope.videos=[DC_whip_11609423,DG_whip_16598313]",
    ])
    config = PipelineConfig.model_validate(cfg_dict)
    assert config.scope.videos == ["DC_whip_11609423", "DG_whip_16598313"]


def test_experiment_with_extra_cli_override():
    """Experiment file + extra CLI override on top works."""
    cfg_dict = _compose([
        "+experiment=sam2_oob_whip",
        "stage2_inference.bidirectional=false",
    ])
    config = PipelineConfig.model_validate(cfg_dict)
    assert config.stage2_inference.model == "sam2"
    assert config.stage2_inference.bidirectional is False


# ----------------------------------------------------------------------------
# Validation guards
# ----------------------------------------------------------------------------


def test_missing_name_fails():
    """The mandatory name field guard fires when not provided."""
    with pytest.raises(Exception, match="name"):
        cfg_dict = _compose([])
        PipelineConfig.model_validate(cfg_dict)


def test_blank_name_fails():
    """Empty / whitespace name is also rejected."""
    with pytest.raises(Exception, match="name"):
        cfg_dict = _compose(["name="])
        PipelineConfig.model_validate(cfg_dict)


def test_unknown_field_rejected():
    """extra='forbid' catches typos."""
    cfg_dict = _compose(["name=t", "+typo_field=oops"])
    with pytest.raises(Exception):
        PipelineConfig.model_validate(cfg_dict)


def test_invalid_model_rejected():
    """Stage2.model is Literal-typed; unknown values rejected."""
    cfg_dict = _compose(["name=t", "stage2_inference.model=fake_model"])
    with pytest.raises(Exception):
        PipelineConfig.model_validate(cfg_dict)


def test_invalid_metric_rejected():
    """Stage3.metrics is a Literal-typed list."""
    cfg_dict = _compose([
        "name=t",
        "stage3_eval.enabled=true",
        "stage3_eval.metrics=[dice,not_a_real_metric]",
    ])
    with pytest.raises(Exception):
        PipelineConfig.model_validate(cfg_dict)


# ----------------------------------------------------------------------------
# Config snapshot (the reproducibility hook)
# ----------------------------------------------------------------------------


def test_dump_config_snapshot_writes_yaml_and_json(tmp_path):
    cfg_dict = _compose(["+experiment=sam2_oob_whip"])
    PipelineConfig.model_validate(cfg_dict)   # ensure validation passes

    out_dir = tmp_path / "results" / "sam2_oob_whip_v1" / "DC_whip_11609423"
    snapshot = dump_config_snapshot(cfg_dict, out_dir)

    assert snapshot == out_dir / "_config.yaml"
    assert snapshot.exists()
    assert (out_dir / "_config.json").exists()

    # Round-trip: written JSON matches the resolved config.
    loaded = json.loads((out_dir / "_config.json").read_text())
    assert loaded["name"] == "sam2_oob_whip_v1"
    assert loaded["stage2_inference"]["model"] == "sam2"


def test_dump_config_snapshot_embeds_into_log_json(tmp_path):
    """If a log.json already exists, the resolved config is merged in."""
    cfg_dict = _compose(["+experiment=surgsam2_oob_whip"])

    out_dir = tmp_path / "results" / "surgsam2_oob_whip_v1" / "DC_whip_11609423"
    out_dir.mkdir(parents=True)
    log_path = out_dir / "log.json"
    log_path.write_text(json.dumps({
        "video": "DC_whip_11609423",
        "n_frames": 2072,
        "wall_seconds": 105.4,
    }))

    dump_config_snapshot(cfg_dict, out_dir, log_path=log_path)

    merged = json.loads(log_path.read_text())
    assert merged["video"] == "DC_whip_11609423"      # original preserved
    assert merged["n_frames"] == 2072
    assert merged["config"]["name"] == "surgsam2_oob_whip_v1"
    assert "config_snapshot_at" in merged
