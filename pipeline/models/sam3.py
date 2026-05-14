"""SAM 3 placeholders. Models are not yet released; these stubs let ablation
configs reference them by name so the registry stays complete. Calling
`.run()` raises NotImplementedError until a real implementation lands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.models.base import VideoTracker


class SAM3VideoTracker(VideoTracker):
    REGISTRY_KEY = "sam3"

    def run(
        self,
        video_id: str,
        frames_dir: Path,
        prompts_json: Path,
        results_dir: Path,
        src_fps: float = 1.0,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "SAM3VideoTracker is a placeholder. Implement in pipeline/models/sam3.py "
            "once SAM 3 weights + API are available."
        )


class SAM3WhipFTVideoTracker(SAM3VideoTracker):
    """SAM 3 fine-tuned on the whip cohort. Placeholder until that
    fine-tuning is completed."""
    REGISTRY_KEY = "sam3_whip_ft"
