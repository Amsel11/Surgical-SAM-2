"""Stage-1 prompter registry.

Each concrete prompter writes a prompts JSON in the same shape clicker.py
produces (objects_by_frame[src_idx] = [{obj_id, box, positive, negative}])
and UPSERTs a prompt_sets + prompt_objects rows into the manifest. The
keys here must match the literals listed in `pipeline.config.PromptMethod`
so YAML typos fail validation.

Only `dino` is registered today. Manual / GT methods don't need a registry
entry — they read pre-existing prompt_sets rows rather than generating new
ones at run time.
"""
from __future__ import annotations

from typing import Any

from pipeline.prompts.dino import GroundingDinoPrompter

PROMPTER_REGISTRY: dict[str, type] = {
    "dino": GroundingDinoPrompter,
}

# Cache prompter instances so the underlying detector (transformers + torch
# weights, ~1.3GB on tiny) loads once per process even when run_one_video
# is called repeatedly inside the orchestrator's per-video loop.
_PROMPTER_CACHE: dict[tuple, Any] = {}


def build_prompter(stage1_cfg) -> Any:
    """Instantiate (or fetch from cache) a prompter from Stage1Prompting config."""
    key = stage1_cfg.method
    if key not in PROMPTER_REGISTRY:
        raise KeyError(
            f"Unknown stage1 method {key!r}. Registered prompters: {sorted(PROMPTER_REGISTRY)}"
        )
    cache_key = (
        key,
        getattr(stage1_cfg, "checkpoint", None),
        getattr(stage1_cfg, "box_threshold", 0.35),
        getattr(stage1_cfg, "text_threshold", 0.25),
        stage1_cfg.n_prompt_frames,
    )
    if cache_key not in _PROMPTER_CACHE:
        cls = PROMPTER_REGISTRY[key]
        _PROMPTER_CACHE[cache_key] = cls(
            checkpoint=cache_key[1],
            box_threshold=cache_key[2],
            text_threshold=cache_key[3],
            n_prompt_frames=cache_key[4],
        )
    return _PROMPTER_CACHE[cache_key]


__all__ = ["PROMPTER_REGISTRY", "build_prompter", "GroundingDinoPrompter"]
