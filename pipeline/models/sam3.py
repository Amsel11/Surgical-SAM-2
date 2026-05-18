"""SAM 3 video tracker.

Implements the same VideoTracker interface as SAM 2 so it slots into the
same pipeline run loop. SAM 3 shares SAM 2's transformer encoder-decoder
shape but has three concrete API differences:

  1. Build is two-step: build_sam3_video_model() returns a container whose
     `.tracker` is the video predictor; the detector backbone is then
     manually grafted onto it before use.
  2. Box prompts are normalized [0,1] xyxy rather than absolute pixels.
  3. `propagate_in_video(...)` yields a 5-tuple
     `(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)`
     instead of SAM 2's 3-tuple. We use `video_res_masks`.

The rest (frame loader, log JSON shape, overlay encoding, bidirectional
propagation) is identical to SAM 2, so we go through the same `pipeline.io`
helpers and write the same on-disk artifacts. Downstream QC tools
(`tools/scan_masks.py`, `tools/qc_tier_a.py`, the overlay renderer) don't
care which model produced the masks/*.png — they're palette PNGs either way.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from pipeline.io import (
    davis_color_bgr,
    encode_mp4_from_jpgs,
    list_frames,
    overlay_mask,
    prepare_loader_dir,
    save_palette_mask,
)
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
        from sam3.model_builder import build_sam3_video_model  # heavy

        cfg = self.cfg
        results_dir = Path(results_dir).resolve()
        masks_dir = results_dir / "masks"
        overlay_dir = results_dir / "overlay"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        loader_dir, source_offset, cleanup_loader = prepare_loader_dir(str(frames_dir))
        try:
            return self._run_inner(
                video_id=video_id,
                loader_dir=loader_dir,
                source_offset=source_offset,
                prompts_json=Path(prompts_json),
                results_dir=results_dir,
                masks_dir=masks_dir,
                overlay_dir=overlay_dir,
                src_fps=src_fps,
                build_video_model=build_sam3_video_model,
            )
        finally:
            cleanup_loader()

    def _run_inner(
        self,
        *,
        video_id: str,
        loader_dir: str,
        source_offset: int,
        prompts_json: Path,
        results_dir: Path,
        masks_dir: Path,
        overlay_dir: Path,
        src_fps: float,
        build_video_model,
    ) -> dict[str, Any]:
        cfg = self.cfg
        frame_names = list_frames(loader_dir)
        if not frame_names:
            raise RuntimeError(f"No frames found in {loader_dir}")
        print(f"Loaded {len(frame_names)} frames (source_offset={source_offset})")

        device = cfg.device if torch.cuda.is_available() else "cpu"
        print(f"Using device {device}")
        if str(device).startswith("cuda"):
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        # SAM 3 two-step build + backbone graft.
        sam3_model = build_video_model(
            checkpoint_path=cfg.checkpoint,
            load_from_HF=False,
            device=device,
        )
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone

        state = predictor.init_state(video_path=loader_dir)
        H = state["video_height"]
        W = state["video_width"]

        with open(prompts_json) as fp:
            pj = json.load(fp)

        prompt_calls = []
        for frame_str, objs in pj.get("objects_by_frame", {}).items():
            src_idx = int(frame_str)
            fidx = src_idx - source_offset
            if fidx < 0 or fidx >= len(frame_names):
                print(f"WARNING: prompt at source frame {src_idx} (loader idx {fidx}) "
                      f"out of range — skipping")
                continue
            for o in objs:
                box = o.get("box")
                if not box:
                    # SAM 3 video predictor only takes box (or text) here; point
                    # prompts aren't covered by this tracker path.
                    continue
                # SAM 3 expects normalized [x0, y0, x1, y1] in [0, 1], shape (1, 4).
                x0, y0, x1, y1 = box
                rel_box = np.array(
                    [[x0 / W, y0 / H, x1 / W, y1 / H]], dtype=np.float32
                )
                prompt_calls.append((fidx, int(o["obj_id"]), rel_box))

        if not prompt_calls:
            raise RuntimeError(
                "No valid box prompts after range filtering — nothing to propagate"
            )

        by_frame: dict[int, int] = {}
        for f, *_ in prompt_calls:
            by_frame[f] = by_frame.get(f, 0) + 1
        print(f"Registering {len(prompt_calls)} (frame,obj) prompts at {sorted(by_frame)}")
        for fidx, oid, rel_box in prompt_calls:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=fidx,
                obj_id=oid,
                box=rel_box,
            )

        first_anchor_idx = min(f for f, *_ in prompt_calls)
        print(f"Propagating bidirectionally from anchor {first_anchor_idx}, "
              f"{len(frame_names)} frames at {W}x{H}")

        mask_area_sum: dict[int, int] = {}
        mask_area_count: dict[int, int] = {}
        empty_frames: dict[int, int] = {}
        written_frames: set[int] = set()

        def write_frame(out_frame_idx: int, out_obj_ids, video_res_masks) -> None:
            combined = np.zeros((H, W), dtype=np.uint8)
            per_obj_masks = []
            for oid, mask in zip(out_obj_ids, video_res_masks):
                m = (mask > 0.0).cpu().numpy().squeeze().astype(bool)
                per_obj_masks.append((int(oid), m))
                combined[m] = int(oid)
                area = int(m.sum())
                mask_area_sum[int(oid)] = mask_area_sum.get(int(oid), 0) + area
                mask_area_count[int(oid)] = mask_area_count.get(int(oid), 0) + 1
                if area == 0:
                    empty_frames[int(oid)] = empty_frames.get(int(oid), 0) + 1
            save_palette_mask(combined, masks_dir / f"{out_frame_idx:05d}.png")

            if cfg.outputs.overlay_video or cfg.outputs.overlay_jpgs:
                frame_path = os.path.join(loader_dir, frame_names[out_frame_idx])
                img_bgr = cv2.imread(frame_path)
                if img_bgr is None:
                    return
                overlay = img_bgr
                for oid, m in per_obj_masks:
                    overlay = overlay_mask(overlay, m, davis_color_bgr(oid), alpha=0.5)
                cv2.imwrite(str(overlay_dir / f"{out_frame_idx:05d}.jpg"), overlay)
            written_frames.add(out_frame_idx)

        t_start = time.time()
        # SAM 3 propagate yields (frame_idx, obj_ids, low_res, video_res, obj_scores).
        if cfg.bidirectional:
            print(f"  Reverse pass: {first_anchor_idx} -> 0 ...")
            for fi, oids, _lo, hi, _scores in predictor.propagate_in_video(
                state, start_frame_idx=first_anchor_idx, reverse=True,
            ):
                write_frame(fi, oids, hi)

        print(f"  Forward pass: {first_anchor_idx} -> {len(frame_names) - 1} ...")
        for fi, oids, _lo, hi, _scores in predictor.propagate_in_video(
            state, start_frame_idx=first_anchor_idx, reverse=False,
        ):
            if fi in written_frames:
                continue
            write_frame(fi, oids, hi)

        wall_seconds = time.time() - t_start
        n_processed = len(written_frames)
        print(f"Done: {n_processed} frames in {wall_seconds:.1f}s "
              f"({n_processed / max(wall_seconds, 1e-6):.1f} fps)")

        if cfg.outputs.overlay_video:
            print("Encoding overlay.mp4 ...")
            encode_mp4_from_jpgs(overlay_dir, results_dir / "overlay.mp4", src_fps)

        if not cfg.outputs.overlay_jpgs:
            for j in overlay_dir.glob("*.jpg"):
                j.unlink()

        mean_mask_area = {
            str(oid): mask_area_sum[oid] / max(mask_area_count[oid], 1)
            for oid in mask_area_sum
        }
        log = {
            "video": video_id,
            "n_frames": n_processed,
            "resolution": [W, H],
            "source_offset": source_offset,
            "checkpoint": cfg.checkpoint,
            "config": getattr(cfg, "config", None),
            "model": self.REGISTRY_KEY,
            "prompts_json": str(prompts_json),
            "prompts_frames": sorted(by_frame.keys()),
            "n_prompt_calls": len(prompt_calls),
            "bidirectional": cfg.bidirectional,
            "wall_seconds": round(wall_seconds, 2),
            "mean_mask_area_px": mean_mask_area,
            "frames_with_empty_mask": {str(k): v for k, v in empty_frames.items()},
            "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(results_dir / "log.json", "w") as fp:
            json.dump(log, fp, indent=2)
        return log


class SAM3WhipFTVideoTracker(SAM3VideoTracker):
    """SAM 3 fine-tuned on the whip cohort. Same code path; cfg.checkpoint
    points at the fine-tuned weights."""
    REGISTRY_KEY = "sam3_whip_ft"
