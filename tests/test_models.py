"""Smoke tests for the model registry. Do NOT import torch/sam2 unless
absolutely necessary — these run on CPU-only laptops in CI."""
from __future__ import annotations

import pytest

from pipeline.config import Stage2Inference, Stage2Outputs
from pipeline.models import MODEL_REGISTRY, build_tracker
from pipeline.models.base import VideoTracker


# ----------------------------------------------------------------------------
# Registry shape
# ----------------------------------------------------------------------------


def test_registry_keys_match_config_literal():
    """Every Stage2Inference.model Literal value has a registry entry."""
    expected = {"sam2", "surgsam2", "surgsam2_whip_ft", "sam3", "sam3_whip_ft"}
    assert set(MODEL_REGISTRY) == expected


def test_registry_classes_subclass_basetracker():
    """All registered values are VideoTracker subclasses."""
    for key, cls in MODEL_REGISTRY.items():
        assert issubclass(cls, VideoTracker), f"{key} -> {cls} not a VideoTracker"


def test_registry_keys_match_class_attribute():
    """The dict key matches each class's REGISTRY_KEY attribute."""
    for key, cls in MODEL_REGISTRY.items():
        assert cls.REGISTRY_KEY == key


# ----------------------------------------------------------------------------
# build_tracker dispatch
# ----------------------------------------------------------------------------


def _stub_cfg(model: str) -> Stage2Inference:
    return Stage2Inference(
        model=model,
        checkpoint="/tmp/fake.pt",
        config="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cpu",
        bidirectional=True,
        outputs=Stage2Outputs(),
    )


def test_build_tracker_dispatches_correct_class():
    for key in MODEL_REGISTRY:
        tracker = build_tracker(_stub_cfg(key))
        assert tracker.REGISTRY_KEY == key
        assert isinstance(tracker, MODEL_REGISTRY[key])


def test_build_tracker_unknown_model_raises():
    cfg = _stub_cfg("sam2")
    # bypass pydantic Literal: poke the attribute after construction
    cfg.__dict__["model"] = "fake_model"
    with pytest.raises(KeyError, match="fake_model"):
        build_tracker(cfg)


# ----------------------------------------------------------------------------
# Placeholder models raise on .run()
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("model_key", ["sam3", "sam3_whip_ft"])
def test_sam3_placeholders_raise_on_run(model_key, tmp_path):
    tracker = build_tracker(_stub_cfg(model_key))
    with pytest.raises(NotImplementedError, match="SAM3"):
        tracker.run(
            video_id="x",
            frames_dir=tmp_path,
            prompts_json=tmp_path / "p.json",
            results_dir=tmp_path / "results",
        )


# ----------------------------------------------------------------------------
# SAM2 family shares one implementation
# ----------------------------------------------------------------------------


def test_sam2_family_uses_same_run_method():
    """SurgSAM2 + SurgSAM2WhipFT inherit SAM2VideoTracker.run unchanged.
    Their REGISTRY_KEY differs so the manifest can tell them apart."""
    from pipeline.models.sam2 import (
        SAM2VideoTracker,
        SurgSAM2VideoTracker,
        SurgSAM2WhipFTVideoTracker,
    )
    assert SurgSAM2VideoTracker.run is SAM2VideoTracker.run
    assert SurgSAM2WhipFTVideoTracker.run is SAM2VideoTracker.run
