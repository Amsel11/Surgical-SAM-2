"""Model registry. Stage 2 of the pipeline looks up `MODEL_REGISTRY[config.model]`.

Each concrete class lives in its own module and subclasses VideoTracker.
To add a new model, drop a file in this directory + register its class
below. The Hydra config's `Literal[...]` for `Stage2Inference.model` should
list the same keys so YAML typos fail validation.
"""
from __future__ import annotations

from pipeline.models.base import VideoTracker
from pipeline.models.sam2 import (
    SAM2VideoTracker,
    SurgSAM2VideoTracker,
    SurgSAM2WhipFTVideoTracker,
)
from pipeline.models.sam3 import SAM3VideoTracker, SAM3WhipFTVideoTracker

MODEL_REGISTRY: dict[str, type[VideoTracker]] = {
    "sam2":             SAM2VideoTracker,
    "surgsam2":         SurgSAM2VideoTracker,
    "surgsam2_whip_ft": SurgSAM2WhipFTVideoTracker,
    "sam3":             SAM3VideoTracker,
    "sam3_whip_ft":     SAM3WhipFTVideoTracker,
}


def build_tracker(stage2_cfg) -> VideoTracker:
    """Look up a model in the registry and instantiate it from stage2 config."""
    key = stage2_cfg.model
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {key!r}. Registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](stage2_cfg)


__all__ = ["VideoTracker", "MODEL_REGISTRY", "build_tracker"]
