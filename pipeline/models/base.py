"""Abstract VideoTracker — the contract every Stage-2 model implements.

The orchestrator (pipeline.run) only sees this interface. It hands a video's
frames + a prompts JSON to the tracker, and gets back a log dict describing
the run. Mask PNGs, overlay JPGs, and overlay.mp4 are written to disk under
`results_dir` as a side effect.

Why a single .run() method instead of init/add_prompts/propagate/produce_outputs:
- SAM3 and other future models may use entirely different propagation APIs.
- The intermediate state (predictor object, inference_state dict) is model-
  specific and doesn't survive between method calls cleanly across models.
- A single fat method keeps the model integration self-contained.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pipeline.config import Stage2Inference


class VideoTracker(ABC):
    """Base class for video segmentation models in the pipeline."""

    REGISTRY_KEY: str = ""    # subclasses set this

    def __init__(self, cfg: Stage2Inference) -> None:
        self.cfg = cfg

    @abstractmethod
    def run(
        self,
        video_id: str,
        frames_dir: Path,
        prompts_json: Path,
        results_dir: Path,
        src_fps: float = 1.0,
    ) -> dict[str, Any]:
        """Run inference on one video.

        Side effects:
          - writes masks/<NNNNN>.png  (palette PNG, pixel = obj_id)
          - writes overlay/<NNNNN>.jpg (if cfg.outputs.overlay_jpgs)
          - writes overlay.mp4         (if cfg.outputs.overlay_video)
          - writes log.json            (the returned dict)

        Returns the log dict so the orchestrator can record it in
        inference_runs without re-reading log.json.
        """
        ...
