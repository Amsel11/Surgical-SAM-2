"""SAM 2 / SurgSAM-2 video trackers.

SurgSAM-2 is architecturally identical to SAM 2.1 hiera-s — it's the same
backbone with a different checkpoint (Endo18 fine-tuned). So all three
classes here (`SAM2VideoTracker`, `SurgSAM2VideoTracker`,
`SurgSAM2WhipFTVideoTracker`) share the same `run()` implementation; they
differ only in their `REGISTRY_KEY`. The actual model selection happens
via `cfg.checkpoint` + `cfg.config`.

Propagation is bidirectional: a reverse pass from the first anchor to
frame 0, then a forward pass from the first anchor to the end. mp4 is
assembled from per-frame overlay JPGs at the end since frames are produced
out of temporal order.
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

# Heavy import deferred to .run() so importing this module doesn't pull in
# the full SAM2 stack (helps unit tests that just want the class shape).


class SAM2VideoTracker(VideoTracker):
    """SAM 2 family video segmentation tracker.

    Subclasses (SurgSAM2, SurgSAM2WhipFT) reuse this implementation
    verbatim — they only override REGISTRY_KEY and rely on Hydra-resolved
    checkpoint paths to swap weights.
    """

    REGISTRY_KEY = "sam2"

    def run(
        self,
        video_id: str,
        frames_dir: Path,
        prompts_json: Path,
        results_dir: Path,
        src_fps: float = 1.0,
    ) -> dict[str, Any]:
        from sam2.build_sam import build_sam2_video_predictor  # heavy
        # SAM 2's build calls hydra.compose() expecting Hydra to be
        # initialized with sam2's own configs/ dir in the search path. Our
        # pipeline already initialized Hydra against our own configs/ — clear
        # and re-init pointing at sam2's configs.
        import sam2 as _sam2_pkg
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        # Init at the sam2 package root (not <pkg>/configs) so that the
        # cfg.config value `configs/sam2.1/sam2.1_hiera_s.yaml` resolves as
        # written, matching what SAM 2's own examples expect.
        sam2_configs = str(Path(_sam2_pkg.__file__).parent)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=sam2_configs, version_base=None)

        cfg = self.cfg
        results_dir = Path(results_dir).resolve()
        masks_dir = results_dir / "masks"
        overlay_dir = results_dir / "overlay"
        masks_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # ── Prepare frame loader (handles frame_NNN.png -> 0,1,2,...) ──────
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
                build_predictor=build_sam2_video_predictor,
            )
        finally:
            cleanup_loader()

    # ------------------------------------------------------------------
    # Inner body, split out so the cleanup path is bullet-proof.
    # ------------------------------------------------------------------

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
        build_predictor,
    ) -> dict[str, Any]:
        cfg = self.cfg
        frame_names = list_frames(loader_dir)
        if not frame_names:
            raise RuntimeError(f"No frames found in {loader_dir}")
        print(f"Loaded {len(frame_names)} frames (source_offset={source_offset})")

        # ── Build predictor + inference state ────────────────────────────
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")
        if device.type == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(device.index or 0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        predictor = build_predictor(cfg.config, cfg.checkpoint, device=device)
        # offload_video_to_cpu: 25,000 × 1920 × 1080 × 3 ≈ 150 GB; can't fit on GPU.
        state = predictor.init_state(
            video_path=loader_dir,
            offload_video_to_cpu=True,
            async_loading_frames=True,
        )
        H = state["video_height"]
        W = state["video_width"]

        # ── Load prompts JSON, subtract source_offset, register ─────────
        with open(prompts_json) as fp:
            pj = json.load(fp)

        prompt_calls = []
        for frame_str, objs in pj.get("objects_by_frame", {}).items():
            src_idx = int(frame_str)
            fidx = src_idx - source_offset
            if fidx < 0 or fidx >= len(frame_names):
                print(f"WARNING: prompt at source frame {src_idx} (loader idx {fidx}) "
                      f"out of range [0, {len(frame_names)}) — skipping")
                continue
            for o in objs:
                pos = o.get("positive", []) or []
                neg = o.get("negative", []) or []
                box = o.get("box")
                pts = pos + neg
                labels = [1] * len(pos) + [0] * len(neg)
                if not pts and not box:
                    continue
                prompt_calls.append((
                    fidx, int(o["obj_id"]),
                    np.asarray(pts, dtype=np.float32) if pts else None,
                    np.asarray(labels, dtype=np.int32) if pts else None,
                    np.asarray(box, dtype=np.float32) if box else None,
                ))

        if not prompt_calls:
            raise RuntimeError(
                "No valid prompts after range filtering — nothing to propagate"
            )

        by_frame: dict[int, int] = {}
        for f, *_ in prompt_calls:
            by_frame[f] = by_frame.get(f, 0) + 1
        print(f"Registering {len(prompt_calls)} (frame,obj) prompts at {sorted(by_frame)}")
        for fidx, oid, points_np, labels_np, box_np in prompt_calls:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=fidx,
                obj_id=oid,
                points=points_np,
                labels=labels_np,
                box=box_np,
            )

        first_anchor_idx = min(f for f, *_ in prompt_calls)
        print(f"Propagating bidirectionally from anchor {first_anchor_idx}, "
              f"{len(frame_names)} frames at {W}x{H}")

        # ── Per-frame writer ─────────────────────────────────────────────
        mask_area_sum: dict[int, int] = {}
        mask_area_count: dict[int, int] = {}
        empty_frames: dict[int, int] = {}
        written_frames: set[int] = set()

        def write_frame(out_frame_idx: int, out_obj_ids, out_mask_logits) -> None:
            # Combine per-object masks into one palette PNG (later obj_id wins on overlap).
            combined = np.zeros((H, W), dtype=np.uint8)
            per_obj_masks = []
            for oid, logits in zip(out_obj_ids, out_mask_logits):
                m = (logits > 0.0).cpu().numpy().squeeze().astype(bool)
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

        # ── Bidirectional propagation ────────────────────────────────────
        t_start = time.time()
        if cfg.bidirectional:
            print(f"  Reverse pass: {first_anchor_idx} -> 0 ...")
            for fi, oids, logits in predictor.propagate_in_video(state, reverse=True):
                write_frame(fi, oids, logits)

        print(f"  Forward pass: {first_anchor_idx} -> {len(frame_names) - 1} ...")
        for fi, oids, logits in predictor.propagate_in_video(state):
            if fi in written_frames:
                continue
            write_frame(fi, oids, logits)

        wall_seconds = time.time() - t_start
        n_processed = len(written_frames)
        print(f"Done: {n_processed} frames in {wall_seconds:.1f}s "
              f"({n_processed / max(wall_seconds, 1e-6):.1f} fps)")

        # ── Encode mp4 from overlay JPGs ─────────────────────────────────
        if cfg.outputs.overlay_video:
            print("Encoding overlay.mp4 ...")
            encode_mp4_from_jpgs(overlay_dir, results_dir / "overlay.mp4", src_fps)

        # Optionally drop the per-frame JPGs to save disk (mass-run default).
        if not cfg.outputs.overlay_jpgs:
            for j in overlay_dir.glob("*.jpg"):
                j.unlink()

        # ── Log ──────────────────────────────────────────────────────────
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
            "config": cfg.config,
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


class SurgSAM2VideoTracker(SAM2VideoTracker):
    """SurgSAM-2 = SAM 2.1 hiera-s + Endo18 fine-tune. Architecture identical
    to SAM 2 — the difference is the checkpoint loaded from cfg.checkpoint."""
    REGISTRY_KEY = "surgsam2"


class SurgSAM2WhipFTVideoTracker(SAM2VideoTracker):
    """SurgSAM-2 further fine-tuned on the whip cohort. Placeholder for the
    yet-to-be-trained checkpoint; the runtime path is identical to SAM 2."""
    REGISTRY_KEY = "surgsam2_whip_ft"
