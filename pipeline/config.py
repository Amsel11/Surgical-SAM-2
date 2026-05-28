"""Pydantic schema for the pipeline's two-stage YAML config.

Hydra (via OmegaConf) loads YAML + applies overrides; pydantic then validates
the resolved dict into typed Python objects with sane defaults. The
distinction matters: Hydra is great at *composition* (defaults lists,
overrides), pydantic is great at *validation* (typed errors point at the
exact bad field).

Two-stage shape, plus optional stage 3:
  - scope             : which (videos × seed) does this run cover
  - stage1_prompting  : how do we tell the model what to track
  - stage2_inference  : which model + how does it propagate
  - stage3_eval       : optional, requires GT
  - infrastructure    : per-machine paths + slurm knobs
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class Scope(BaseModel):
    """Which (video × seed) pairs this pipeline run covers.

    `videos`:
      - 'all'                : every video in the cohort
      - 'qc_pass'            : only those with qc_results.status='pass'
      - 'failed_inference'   : only inference_runs with status='failed'
      - list[str]            : explicit video_ids (e.g. ['DC_whip_11609423'])
    """
    model_config = ConfigDict(extra="forbid")

    cohort: str = "whip"
    videos: Literal["all", "qc_pass", "failed_inference"] | list[str] = "all"
    seed: int = 1


# ---------------------------------------------------------------------------
# Stage 1 — Prompting
# ---------------------------------------------------------------------------

# Concrete prompt strategies. Each will get a registry entry in pipeline/prompts/.
PromptMethod = Literal["manual_box", "manual_click", "yolo", "dino", "gt_box", "auto"]


class AutoPromptConfig(BaseModel):
    """Knobs for the fully-automated, class-agnostic prompter (method='auto').

    Only consulted when `Stage1Prompting.method == 'auto'`. The defaults are the
    validated v0 whip hyperparameters. `model_id`/`checkpoint`/`box_threshold`/
    `text_threshold` on the parent Stage1Prompting drive the detector; the knobs
    here drive anchor box selection + cross-anchor track matching.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = "surgical instrument"  # single generic GD query
    score_floor: float = 0.15
    min_area_frac: float = 0.02
    max_area_frac: float = 0.50
    exclude_bottom_frac: float = 0.05   # drop boxes ENTIRELY in bottom strip (OCR)
    box_shrink_frac: float = 0.10       # tighten each anchor box before SAM2
    nms_iou: float = 0.30
    match_iou: float = 0.20             # new box vs existing track's last box
    max_tracks: int = 6                 # global cap on distinct obj_ids
    anchor_offset: int = 0              # frames past each segment start to anchor


class Stage1Prompting(BaseModel):
    """How prompts are produced for each video.

    `source` is method-specific:
      - manual_box / manual_click : 'clicker_db' (read from prompt_sets table)
      - yolo                      : path to YOLO weights, e.g. './weights/yolov8.pt'
      - dino                      : 'hf' for the HF zero-shot baseline, or a
                                    local checkpoint path (also see `checkpoint`)
      - gt_box                    : 'gt' (derive bbox from GT mask)
    """
    model_config = ConfigDict(extra="forbid")

    method: PromptMethod
    source: str = "clicker_db"
    n_prompt_frames: int = 3
    sample_strategy: Literal["stratified_uniform"] = "stratified_uniform"
    resample_empty: bool = True
    # Which instrument classes to label/track. ['all'] = every clicked obj.
    instrument_filter: list[str] = Field(default_factory=lambda: ["all"])
    # Grounding DINO knobs (used only when method='dino'). `checkpoint`, when
    # set, overrides the default HF model id — same hook the FT'd weights
    # drop into.
    checkpoint: str | None = None
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    # Used only when method='auto'.
    model_id: str = "IDEA-Research/grounding-dino-base"
    segments_root: str | None = None    # OCR segments root: <root>/<video_id>/segments.csv
    auto: AutoPromptConfig = Field(default_factory=AutoPromptConfig)


# ---------------------------------------------------------------------------
# Stage 2 — Inference
# ---------------------------------------------------------------------------

# Registry keys for swappable models.
ModelKey = Literal["sam2", "surgsam2", "surgsam2_whip_ft", "sam3", "sam3_whip_ft"]


class Stage2Outputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    masks: bool = True
    overlay_video: bool = True
    overlay_jpgs: bool = False
    preview_small: bool = True   # 320p CRF32 for cheap rsync


class Stage2Inference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelKey
    checkpoint: str
    config: str | None = None  # SAM 2 needs an architecture yaml; SAM 3 doesn't.
    device: str = "cuda:0"
    bidirectional: bool = True   # reverse pass first, then forward
    outputs: Stage2Outputs = Field(default_factory=Stage2Outputs)


# ---------------------------------------------------------------------------
# Stage 2 post-processing (optional, runs on the tracker's output masks)
# ---------------------------------------------------------------------------


class DedupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    iou_thr: float = 0.30               # mean bbox-IoU above which tracks merge
    min_lifespan_overlap: float = 0.60  # below this -> temporally distinct, keep
    min_mean_area_frac: float = 0.005   # mean-when-present below this -> ghost
    min_frames: int = 100               # pairs co-present in fewer frames skipped


class PostProcess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dedup: DedupConfig = Field(default_factory=DedupConfig)


# ---------------------------------------------------------------------------
# Stage 3 — Evaluation (optional, requires GT)
# ---------------------------------------------------------------------------


class Stage3EvalFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qc_status: Literal["pass", "review", "fail", "all"] = "pass"
    exclude_bad_time_ranges: bool = True


class Stage3Eval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    gt_source: str | None = None
    metrics: list[Literal["dice", "iou", "temporal_consistency"]] = Field(
        default_factory=lambda: ["dice", "iou"]
    )
    filter: Stage3EvalFilter = Field(default_factory=Stage3EvalFilter)


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


class SlurmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partition: str = "oermannlab,a100_short"
    exclude: str = "a100-8003"
    time: str = "03:00:00"
    array_concurrency: int = 8


class Infrastructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames_root: str
    results_root: str
    manifest_db: str
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Root of the pipeline config. One YAML = one run definition."""
    model_config = ConfigDict(extra="forbid")

    name: str                           # required, no default — Hydra '???'
    description: str = ""
    scope: Scope
    stage1_prompting: Stage1Prompting
    stage2_inference: Stage2Inference
    postprocess: PostProcess = Field(default_factory=PostProcess)
    stage3_eval: Stage3Eval = Field(default_factory=Stage3Eval)
    infrastructure: Infrastructure

    @field_validator("name")
    @classmethod
    def _name_must_be_set(cls, v: str) -> str:
        # Hydra leaves '???' as a literal string when a required field isn't
        # provided (instead of raising). Catch it here so misconfigured runs
        # fail at validation time, not silently with name='???'.
        if v == "???" or not v.strip():
            raise ValueError(
                "`name` is required. Set it via `name=<run_id>` on the CLI "
                "or inside an experiment file under configs/experiment/."
            )
        return v
