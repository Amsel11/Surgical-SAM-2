"""Grounding DINO text-prompted single-frame detector.

This is **not** a VideoTracker — it produces bounding boxes for an image
given free-text queries. The dino prompter (`pipeline.prompts.dino`)
wraps it to produce Stage-1 prompt JSON in the clicker's format.

Why this lives under `pipeline.prompts` rather than `pipeline.models`: the
models package's __init__ eagerly imports SAM2/SAM3 trackers, which need
heavy SAM-specific deps. The dino prompter is supposed to run in a slim
`.gdino_venv` with only `transformers`/`torch` — avoiding that package
trigger keeps the env minimal. The module is prefixed with `_` to signal
that callers should reach the detector via the prompter; tools that want
the raw detector can still import from here.

Implementation: HuggingFace `transformers` zero-shot object detection.
The `IDEA-Research/grounding-dino-tiny` checkpoint is ~1.3 GB and runs at
~10 fps on a single A100 for one frame with a handful of queries; the
`-base` variant is more accurate but ~3x slower.

Text query convention for Grounding DINO:
- Queries are joined with " . " so the model treats each as a distinct
  prompt and post-processing maps detections back to their source query.
- All queries should be lowercase, end without punctuation (the "." is
  added by us).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class GroundingDinoConfig:
    """Lightweight config for the detector. Mirrors the YAML structure but
    not a pydantic model — keeps this module importable without dragging
    the full pipeline.config dependency chain in."""

    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cuda"
    # Optional cache dir override so HF_HOME stays out of $HOME on bp.
    cache_dir: str | None = None


class GroundingDinoDetector:
    """Zero-shot text-prompted detector. One instance per process; the
    backbone is heavy (~1.3 GB tiny, ~3 GB base) so callers should reuse
    it across many frames.

    Example:
        det = GroundingDinoDetector(GroundingDinoConfig())
        boxes_by_query = det.detect(
            "frame.png",
            ["bipolar forceps", "grasping retractor"],
        )
        # → {"bipolar forceps": [[x0,y0,x1,y1,score], ...], ...}
    """

    def __init__(self, cfg: GroundingDinoConfig | Any) -> None:
        # Accept either a dataclass or an OmegaConf/dict-like; coerce to
        # GroundingDinoConfig so attribute access is uniform downstream.
        if isinstance(cfg, GroundingDinoConfig):
            self.cfg = cfg
        else:
            kwargs: dict[str, Any] = {}
            for field_name in (
                "model_id",
                "box_threshold",
                "text_threshold",
                "device",
                "cache_dir",
            ):
                if hasattr(cfg, field_name):
                    val = getattr(cfg, field_name)
                    if val is not None:
                        kwargs[field_name] = val
                elif isinstance(cfg, dict) and field_name in cfg:
                    kwargs[field_name] = cfg[field_name]
            self.cfg = GroundingDinoConfig(**kwargs)

        # Lazy heavy imports — torch + transformers — so simply importing
        # this module doesn't pay the cost.
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self._torch = torch

        device = self.cfg.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(
                f"[grounding_dino] device={device!r} requested but CUDA "
                f"unavailable — falling back to cpu"
            )
            device = "cpu"
        self._device = device

        proc_kwargs: dict[str, Any] = {}
        if self.cfg.cache_dir:
            proc_kwargs["cache_dir"] = self.cfg.cache_dir

        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_id, **proc_kwargs
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.cfg.model_id, **proc_kwargs
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_query(q: str) -> str:
        """Lowercase, strip whitespace, strip trailing punctuation.

        Grounding DINO expects lowercase queries joined by ' . '. Any
        trailing period in the user's query would create empty prompts
        after the join.
        """
        return q.strip().rstrip(".").strip().lower()

    @classmethod
    def _build_prompt(cls, queries: list[str]) -> tuple[str, list[str]]:
        """Return (prompt_string, normalized_queries) for the processor.

        Queries are joined with " . " followed by a trailing " ." — the
        trailing period is part of the model's expected input format (per
        the HF Grounding DINO usage examples).
        """
        norm = [cls._normalize_query(q) for q in queries if q.strip()]
        if not norm:
            raise ValueError("No non-empty text queries supplied to detect()")
        prompt = " . ".join(norm) + " ."
        return prompt, norm

    # ------------------------------------------------------------------
    def detect(
        self,
        image_path: str | Path,
        text_queries: list[str],
    ) -> dict[str, list[list[float]]]:
        """Run Grounding DINO on `image_path` with `text_queries`.

        Returns a dict mapping each (normalized) query to a list of
        `[x0, y0, x1, y1, score]` boxes in original-image pixel coords.
        Queries with no detections map to an empty list.

        All queries are run in a single forward pass via the joined-string
        format; post-processing then attributes each box to its source
        query via substring matching.
        """
        torch = self._torch
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt, norm_queries = self._build_prompt(text_queries)

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # post_process_grounded_object_detection returns per-image dicts
        # with keys: scores, labels (text label strings), boxes (xyxy).
        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            # Newer transformers (>=4.51) renamed the kwargs.
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.cfg.box_threshold,
                text_threshold=self.cfg.text_threshold,
                target_sizes=target_sizes,
            )
        result = results[0]

        scores = result["scores"].detach().cpu().numpy()
        boxes = result["boxes"].detach().cpu().numpy()
        labels = result.get("labels") or result.get("text_labels") or []
        labels = list(labels)

        by_query: dict[str, list[list[float]]] = {q: [] for q in norm_queries}
        for score, box, label in zip(scores, boxes, labels):
            # Grounding DINO sometimes emits a partial label ("forceps"
            # when the query was "bipolar forceps") — pick the query with
            # the longest matching token overlap, falling back to
            # substring matching.
            label_lc = str(label).strip().lower()
            best = _match_label_to_query(label_lc, norm_queries)
            if best is None:
                continue
            by_query[best].append(
                [float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score)]
            )

        for q in by_query:
            by_query[q].sort(key=lambda b: b[4], reverse=True)
        return by_query


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _match_label_to_query(label: str, queries: list[str]) -> str | None:
    """Attribute a decoded label string to one of the original queries.

    Strategy:
      1. Exact (lowercased) equality.
      2. Query is a substring of the label.
      3. Label is a substring of the query.
      4. Token overlap; ties go to the query with the most overlapping
         tokens, then longest query.
    """
    label = label.strip().lower()
    if not label:
        return None
    for q in queries:
        if q == label:
            return q
    for q in queries:
        if q in label:
            return q
    for q in queries:
        if label in q:
            return q
    label_tokens = set(label.split())
    best: tuple[int, int, str | None] = (0, 0, None)
    for q in queries:
        q_tokens = set(q.split())
        overlap = len(label_tokens & q_tokens)
        if overlap > best[0] or (overlap == best[0] and len(q) > best[1]):
            best = (overlap, len(q), q)
    if best[0] == 0:
        return None
    return best[2]


def iou_xyxy(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Intersection-over-union for two xyxy boxes. 0 for non-overlapping
    or zero-area boxes."""
    ax0, ay0, ax1, ay1 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx0, by0, bx1, by1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


__all__ = [
    "GroundingDinoConfig",
    "GroundingDinoDetector",
    "iou_xyxy",
]
