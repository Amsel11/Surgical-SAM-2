"""Stage-1 prompter: fully-automated OCR-anchored, class-agnostic detector.

This is the canonical zero-shot prompter for the whip cohort. Per OCR segment
(from `segments.csv`, the da Vinci arm-slot timeline) it picks one anchor
frame, runs Grounding DINO with a single generic query ("surgical instrument"),
deduplicates the boxes, matches them to persistent tracks across anchors, and
emits a prompts JSON the SAM2/SAM3 trackers consume directly.

Why generic (class-agnostic) rather than per-instrument queries: GD's labels
for visually similar instruments are unreliable (the Qwen validator showed 87%
can't disambiguate), but its *localization* is decent. So we treat GD as a
localizer and let cross-anchor IoU matching carry a stable obj_id, exactly the
way `tools/anchor_and_propagate.py` did — that logic now lives in
`_anchor_select.py` + this module.

obj_ids are arm-slot-free track ids (1..max_tracks), allocated in order of
first appearance. The output is intentionally identity-free: clean masks +
persistent tracks, nothing else. OCR identity per arm/moment stays recoverable
from segments.csv but is not baked into the masks.

Fine-tuned GD drops in via `checkpoint` (overrides the HF model id) — the same
hook the `dino` prompter exposes. The generic query means **no vocab file is
needed**.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.db import connect, transaction
from pipeline.io import PROMPTS_DIR, frame_files_ordered
from pipeline.prompts._anchor_select import (
    _shrink_box,
    load_segments,
    select_boxes_class_agnostic,
)
from pipeline.prompts._grounding_dino_detector import (
    GroundingDinoConfig,
    GroundingDinoDetector,
    iou_xyxy,
)


def _src_of(filename: str) -> int:
    """Integer source index captured from a frame filename stem.

    Matches the offset prepare_loader_dir derives: 'frame_0000000518.png' -> 518,
    '0.png' -> 0. Used to map loader index -> tracker-resolvable source key.
    """
    stem = Path(filename).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


class AutoPrompter:
    """Automated class-agnostic stage-1 prompter (method ``auto``)."""

    PROMPT_METHOD: str = "auto"

    def __init__(
        self,
        *,
        model_id: str = "IDEA-Research/grounding-dino-base",
        checkpoint: str | None = None,
        query: str = "surgical instrument",
        box_threshold: float = 0.15,
        text_threshold: float = 0.10,
        device: str = "cuda",
        score_floor: float = 0.15,
        min_area_frac: float = 0.02,
        max_area_frac: float = 0.50,
        exclude_bottom_frac: float = 0.05,
        box_shrink_frac: float = 0.10,
        nms_iou: float = 0.30,
        match_iou: float = 0.20,
        max_tracks: int = 6,
        anchor_offset: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        # `checkpoint`, if set, replaces the HF model id (FT'd weights drop-in).
        self.model_id = checkpoint or model_id
        self.query = query
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.cache_dir = cache_dir
        # Selection / matching knobs.
        self.score_floor = score_floor
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.exclude_bottom_frac = exclude_bottom_frac
        self.box_shrink_frac = box_shrink_frac
        self.nms_iou = nms_iou
        self.match_iou = match_iou
        self.max_tracks = max_tracks
        self.anchor_offset = anchor_offset
        self._detector: GroundingDinoDetector | None = None

    def _ensure_detector(self) -> GroundingDinoDetector:
        if self._detector is None:
            self._detector = GroundingDinoDetector(
                GroundingDinoConfig(
                    model_id=self.model_id,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device,
                    cache_dir=self.cache_dir,
                )
            )
        return self._detector

    # ------------------------------------------------------------------
    def run(
        self,
        video_id: str,
        frames_dir: Path,
        seed: int,
        segments_csv: Path,
        prompts_dir: Path | None = None,
    ) -> Path:
        """Detect on each segment's anchor frame, write prompts JSON, UPSERT
        manifest rows. Returns the JSON path."""
        frames_dir = Path(frames_dir)
        segments_csv = Path(segments_csv)
        prompts_dir = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
        prompts_dir.mkdir(parents=True, exist_ok=True)

        segments = load_segments(segments_csv)
        if not segments:
            raise RuntimeError(f"No OCR segments in {segments_csv}")

        # Loader order is the single source of truth (matches prepare_loader_dir
        # and the mask filename indices). source_offset = src of the first frame
        # so that tracker's `fidx = pseudo_src - source_offset` lands on the
        # intended loader index.
        ordered = frame_files_ordered(frames_dir)
        n_frames = len(ordered)
        source_offset = _src_of(ordered[0])
        with Image.open(frames_dir / ordered[0]) as im:
            w, h = im.size
        print(f"=== {video_id}: {len(segments)} segments, {n_frames} frames "
              f"({w}x{h}), source_offset={source_offset} ===")

        detector = self._ensure_detector()

        # Persistent across segments: obj_id -> last anchor box [x0,y0,x1,y1].
        tracks: dict[int, list[float]] = {}
        next_obj_id = 1

        objects_by_frame: dict[str, list[dict[str, Any]]] = {}
        prompt_frames: list[int] = []
        unique_objs: set[int] = set()

        for seg_idx, seg in enumerate(segments):
            anchor_loader = min(seg["start_i"] + self.anchor_offset, seg["end_i"])
            if anchor_loader < 0 or anchor_loader >= n_frames:
                print(f"  seg {seg_idx}: anchor loader idx {anchor_loader} out of "
                      f"range [0,{n_frames}) — skipping")
                continue
            anchor_path = frames_dir / ordered[anchor_loader]
            pseudo_src = source_offset + anchor_loader

            by_query = detector.detect(anchor_path, [self.query])
            new_boxes = select_boxes_class_agnostic(
                by_query, w, h,
                score_floor=self.score_floor,
                min_af=self.min_area_frac,
                max_af=self.max_area_frac,
                nms_iou=self.nms_iou,
                exclude_bottom_frac=self.exclude_bottom_frac,
            )
            if not new_boxes:
                print(f"  seg {seg_idx} loader={anchor_loader}: no usable GD boxes — skipping")
                continue

            # Match each new box to the best unclaimed existing track by IoU vs
            # its last-known box (greedy on L->R box order). One existing track
            # gets at most one new box per anchor; misses allocate a fresh id
            # up to the global max_tracks cap.
            assigned: list[tuple[int, list[float]]] = []
            claimed: set[int] = set()
            for box in new_boxes:
                best_id, best_iou = None, 0.0
                for oid, prev_box in tracks.items():
                    if oid in claimed:
                        continue
                    iou = iou_xyxy(box, prev_box)
                    if iou > best_iou:
                        best_id, best_iou = oid, iou
                if best_id is not None and best_iou >= self.match_iou:
                    assigned.append((best_id, box))
                    tracks[best_id] = box
                    claimed.add(best_id)
                elif next_obj_id <= self.max_tracks:
                    assigned.append((next_obj_id, box))
                    tracks[next_obj_id] = box
                    claimed.add(next_obj_id)
                    next_obj_id += 1
                # else: global cap hit -> drop the box

            objs = []
            for obj_id, box in assigned:
                shrunk = _shrink_box(box, self.box_shrink_frac)
                objs.append({
                    "obj_id":   int(obj_id),
                    "box":      [float(x) for x in shrunk],
                    "positive": [],
                    "negative": [],
                })
                unique_objs.add(int(obj_id))
            objects_by_frame[str(pseudo_src)] = objs
            prompt_frames.append(pseudo_src)
            print(f"  seg {seg_idx} loader={anchor_loader} pseudo_src={pseudo_src}: "
                  f"{len(objs)} prompts -> ids {[o['obj_id'] for o in objs]} "
                  f"(tracks alive: {len(tracks)})")

        if not objects_by_frame:
            raise RuntimeError(
                f"AutoPrompter produced no anchors for {video_id}. Try lowering "
                f"box_threshold ({self.box_threshold}) / score_floor "
                f"({self.score_floor})."
            )

        payload = {
            "video":            video_id,
            "resolution":       [w, h],
            "n_frames":         n_frames,
            "prompt_frames":    sorted(prompt_frames),
            "objects_by_frame": objects_by_frame,
        }
        out_path = prompts_dir / f"{video_id}_seed{seed}_{self.PROMPT_METHOD}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Prompts -> {out_path} ({len(prompt_frames)} anchors, "
              f"{len(unique_objs)} tracks)")

        self._upsert_manifest(video_id, seed, out_path, unique_objs, prompt_frames)
        return out_path

    # ------------------------------------------------------------------
    def _upsert_manifest(
        self,
        video_id: str,
        seed: int,
        out_path: Path,
        unique_objs: set[int],
        prompt_frames: list[int],
    ) -> None:
        """UPSERT prompt_sets + prompt_objects (instrument_id='unknown_instrument'
        — the auto path is identity-free). Mirrors the dino prompter so re-runs
        cleanly overwrite the previous attempt."""
        conn = connect()
        with transaction(conn):
            existing = conn.execute(
                "SELECT prompt_set_id FROM prompt_sets "
                "WHERE video_id=? AND seed=? AND prompt_method=?",
                (video_id, seed, self.PROMPT_METHOD),
            ).fetchone()
            if existing:
                ps_id = existing["prompt_set_id"]
                conn.execute("DELETE FROM prompt_objects WHERE prompt_set_id=?", (ps_id,))
                conn.execute(
                    "UPDATE prompt_sets SET prompts_path=?, n_objects=?, "
                    "n_prompt_frames=?, status='ready', created_by='auto-prompter' "
                    "WHERE prompt_set_id=?",
                    (str(out_path), len(unique_objs), len(prompt_frames), ps_id),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO prompt_sets "
                    "(video_id, seed, prompt_method, prompts_path, n_objects, "
                    " n_prompt_frames, status, created_by) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'ready', 'auto-prompter')",
                    (video_id, seed, self.PROMPT_METHOD, str(out_path),
                     len(unique_objs), len(prompt_frames)),
                )
                ps_id = cur.lastrowid
            for obj_id in sorted(unique_objs):
                conn.execute(
                    "INSERT INTO prompt_objects (prompt_set_id, obj_id, instrument_id) "
                    "VALUES (?, ?, 'unknown_instrument')",
                    (ps_id, int(obj_id)),
                )


__all__ = ["AutoPrompter"]
