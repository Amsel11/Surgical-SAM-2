"""Stage-1 prompter: Grounding DINO zero-shot box detector.

Runs HF `grounding-dino-tiny` (or a fine-tuned checkpoint via cfg.checkpoint)
over 3 anchor frames per video, greedy-matches boxes across frames so each
instrument carries a stable obj_id through SAM propagation, writes a prompts
JSON in the clicker's format, and UPSERTs prompt_sets/prompt_objects rows.

Vocabulary (the text queries fed to GD) is read from
`configs/cardiac_whip_vocab.json`, the output of `tools/build_vocab.py`
(Phase 0c). When that file is missing — Phase 0 not yet complete — the
prompter falls back to the manifest's `instruments` table so smoke runs
remain possible.

Why greedy across-frame matching rather than Hungarian: there are only
~6 instruments × 3 anchor frames in scope; n is too small for Hungarian
to matter and dropping the scipy dep keeps the prompter env minimal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline.db import REPO_ROOT, connect, transaction
from pipeline.io import PROMPTS_DIR, list_frames, sample_frame_positions
from pipeline.prompts._grounding_dino_detector import (
    GroundingDinoConfig,
    GroundingDinoDetector,
    iou_xyxy,
)

VOCAB_PATH = REPO_ROOT / "configs" / "cardiac_whip_vocab.json"


@dataclass
class _VocabEntry:
    instrument_id: str
    canonical: str          # the text label fed to Grounding DINO
    ui_variants: list[str]  # reserved for the prompt-ensemble ablation


def load_vocab(path: Path | None = None) -> list[_VocabEntry]:
    """Load the cardiac vocab JSON. Falls back to the instruments table if missing."""
    p = Path(path) if path else VOCAB_PATH
    if p.exists():
        data = json.loads(p.read_text())
        return [
            _VocabEntry(
                instrument_id=e["instrument_id"],
                canonical=e["canonical"],
                ui_variants=e.get("ui_variants", []),
            )
            for e in data["instruments"]
        ]
    conn = connect()
    rows = conn.execute(
        "SELECT instrument_id, display_name FROM instruments ORDER BY instrument_id"
    ).fetchall()
    return [
        _VocabEntry(
            instrument_id=r["instrument_id"],
            canonical=r["display_name"],
            ui_variants=[],
        )
        for r in rows
    ]


def _greedy_match(
    boxes_a: list[list[float]],
    boxes_b: list[list[float]],
    iou_threshold: float = 0.30,
) -> dict[int, int]:
    """Greedy IoU matching: descending-IoU pair list, first-come-first-served.

    Returns {b_idx: a_idx} for matched pairs above `iou_threshold`. n is small
    here (≤ ~6×6) so the O(nm log nm) sort is fine.
    """
    if not boxes_a or not boxes_b:
        return {}
    pairs: list[tuple[float, int, int]] = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            iou = iou_xyxy(a, b)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    out: dict[int, int] = {}
    for _iou, i, j in pairs:
        if i in matched_a or j in matched_b:
            continue
        out[j] = i
        matched_a.add(i)
        matched_b.add(j)
    return out


def _assign_obj_ids_across_frames(
    frame_detections: list[list[tuple[str, list[float]]]],
    iou_threshold: float = 0.30,
) -> list[list[tuple[int, str, list[float]]]]:
    """Walk anchor frames in order; per-class greedy-match each frame's boxes
    against the running registry of objects. Matched dets inherit existing
    obj_ids; unmatched dets get fresh ids.

    Per-class only — never match a "scissors" box on frame 2 to a "forceps"
    box on frame 1, even if they overlap.
    """
    if not frame_detections:
        return []
    # obj_id -> (instrument_id, last_known_box) — drives matching of NEXT frame
    objects: dict[int, tuple[str, list[float]]] = {}
    out: list[list[tuple[int, str, list[float]]]] = []
    next_id = 1
    for dets in frame_detections:
        per_class_dets: dict[str, list[int]] = {}
        for di, (cls, _) in enumerate(dets):
            per_class_dets.setdefault(cls, []).append(di)
        per_class_objs: dict[str, list[int]] = {}
        for oid, (cls, _) in objects.items():
            per_class_objs.setdefault(cls, []).append(oid)

        assigned: dict[int, int] = {}
        for cls, det_indices in per_class_dets.items():
            obj_ids_for_class = per_class_objs.get(cls, [])
            if not obj_ids_for_class:
                continue
            obj_boxes = [objects[oid][1] for oid in obj_ids_for_class]
            det_boxes = [dets[di][1] for di in det_indices]
            matches = _greedy_match(obj_boxes, det_boxes, iou_threshold)
            for b_idx, a_idx in matches.items():
                det_idx = det_indices[b_idx]
                assigned[det_idx] = obj_ids_for_class[a_idx]

        frame_out: list[tuple[int, str, list[float]]] = []
        for di, (cls, box) in enumerate(dets):
            oid = assigned.get(di)
            if oid is None:
                oid = next_id
                next_id += 1
            frame_out.append((oid, cls, box))
            objects[oid] = (cls, box)
        out.append(frame_out)
    return out


class GroundingDinoPrompter:
    """Stage-1 prompter using Grounding DINO to detect instruments on the same
    25/50/75% anchor frames the clicker samples, so manual_box vs dino is
    apples-to-apples in downstream evaluation.

    Designed for the FT'd-checkpoint drop-in: pass `checkpoint=<path>` and the
    fine-tuned weights replace the HF zero-shot baseline without any other
    code path change.
    """

    PROMPT_METHOD: str = "dino"

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        checkpoint: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda",
        vocab_path: str | Path | None = None,
        n_prompt_frames: int = 3,
        match_iou_threshold: float = 0.30,
    ) -> None:
        # `checkpoint`, if provided, overrides the default HF model_id. The
        # detector accepts either an HF model id or a local path; transformers
        # routes the load.
        self.model_id = checkpoint or model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.vocab_path = Path(vocab_path) if vocab_path else VOCAB_PATH
        self.n_prompt_frames = n_prompt_frames
        self.match_iou_threshold = match_iou_threshold
        self._detector: GroundingDinoDetector | None = None
        self._vocab: list[_VocabEntry] | None = None

    def _ensure_detector(self) -> GroundingDinoDetector:
        if self._detector is None:
            self._detector = GroundingDinoDetector(
                GroundingDinoConfig(
                    model_id=self.model_id,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device,
                )
            )
        return self._detector

    def _ensure_vocab(self) -> list[_VocabEntry]:
        if self._vocab is None:
            self._vocab = load_vocab(self.vocab_path)
        return self._vocab

    def run(
        self,
        video_id: str,
        frames_dir: Path,
        seed: int,
        prompts_dir: Path | None = None,
    ) -> Path:
        """Detect on the anchor frames, write prompts JSON, UPSERT manifest rows.

        Returns the path to the JSON. UPSERT semantics match the clicker so
        re-running this prompter on the same (video, seed) cleanly overwrites
        the previous attempt.
        """
        prompts_dir = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
        prompts_dir.mkdir(parents=True, exist_ok=True)

        vocab = self._ensure_vocab()
        if not vocab:
            raise RuntimeError(
                "No vocabulary available. Either populate "
                f"{self.vocab_path} via tools/build_vocab.py, or seed the "
                "instruments table in the manifest."
            )
        # Detector returns labels normalized via _normalize_query (lowercase,
        # whitespace-stripped). Match that here so the reverse lookup works.
        canonical_to_id = {v.canonical.strip().lower(): v.instrument_id for v in vocab}
        text_queries = [v.canonical for v in vocab]

        # list_frames returns filenames; join to frames_dir for absolute paths.
        frames_dir = Path(frames_dir)
        frame_names = list_frames(frames_dir)
        if not frame_names:
            raise RuntimeError(f"No frames in {frames_dir}")
        frames = [frames_dir / n for n in frame_names]
        positions = sample_frame_positions(len(frames), self.n_prompt_frames)

        detector = self._ensure_detector()

        per_frame: list[list[tuple[str, list[float]]]] = []
        for arr_idx in positions:
            frame_path = frames[arr_idx]
            by_query = detector.detect(frame_path, text_queries)
            per_frame_dets: list[tuple[str, list[float]]] = []
            for query_norm, boxes in by_query.items():
                instrument_id = canonical_to_id.get(query_norm)
                if instrument_id is None:
                    continue
                for b in boxes:
                    per_frame_dets.append(
                        (instrument_id, [float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                    )
            per_frame.append(per_frame_dets)

        per_frame_with_ids = _assign_obj_ids_across_frames(
            per_frame, iou_threshold=self.match_iou_threshold
        )

        # Source-frame indices: this prompter is intended to run against bp's
        # full-extracted frame directories where local position == source idx.
        # Locally-subsampled frame sets (the clicker handles those via
        # parse_source_idx) are not in scope for the dino prompter.
        objects_by_frame: dict[str, list[dict[str, Any]]] = {}
        unique_objs: dict[int, str] = {}
        for fpos, arr_idx in enumerate(positions):
            dets = per_frame_with_ids[fpos]
            if not dets:
                continue
            entries = []
            for obj_id, instrument_id, box in dets:
                entries.append({
                    "obj_id": int(obj_id),
                    "box": [float(v) for v in box],
                    "positive": [],
                    "negative": [],
                })
                unique_objs[obj_id] = instrument_id
            objects_by_frame[str(arr_idx)] = entries

        if not objects_by_frame:
            raise RuntimeError(
                f"Grounding DINO produced no detections for {video_id} on any "
                f"anchor frame. Try lowering box_threshold (current "
                f"{self.box_threshold}) or text_threshold (current "
                f"{self.text_threshold})."
            )

        first_image = Image.open(frames[0])
        w, h = first_image.size

        conn = connect()
        row = conn.execute(
            "SELECT n_frames FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        bp_n_frames = row["n_frames"] if row and row["n_frames"] else len(frames)

        prompt_source_indices = sorted(int(k) for k in objects_by_frame.keys())
        payload = {
            "video": video_id,
            "resolution": [w, h],
            "n_frames": bp_n_frames,
            "prompt_frames": prompt_source_indices,
            "objects_by_frame": objects_by_frame,
        }
        out_path = prompts_dir / f"{video_id}_seed{seed}_{self.PROMPT_METHOD}.json"
        out_path.write_text(json.dumps(payload, indent=2))

        with transaction(conn):
            existing = conn.execute(
                "SELECT prompt_set_id FROM prompt_sets "
                "WHERE video_id=? AND seed=? AND prompt_method=?",
                (video_id, seed, self.PROMPT_METHOD),
            ).fetchone()
            if existing:
                ps_id = existing["prompt_set_id"]
                conn.execute(
                    "DELETE FROM prompt_objects WHERE prompt_set_id=?", (ps_id,)
                )
                conn.execute(
                    """
                    UPDATE prompt_sets SET
                        prompts_path=?, n_objects=?, n_prompt_frames=?,
                        status='ready', created_by='dino-prompter'
                    WHERE prompt_set_id=?
                    """,
                    (str(out_path), len(unique_objs), len(prompt_source_indices), ps_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO prompt_sets
                        (video_id, seed, prompt_method, prompts_path,
                         n_objects, n_prompt_frames, status, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, 'ready', 'dino-prompter')
                    """,
                    (video_id, seed, self.PROMPT_METHOD, str(out_path),
                     len(unique_objs), len(prompt_source_indices)),
                )
                ps_id = cur.lastrowid
            for obj_id, instrument_id in unique_objs.items():
                conn.execute(
                    "INSERT INTO prompt_objects (prompt_set_id, obj_id, instrument_id) "
                    "VALUES (?, ?, ?)",
                    (ps_id, int(obj_id), instrument_id),
                )

        return out_path


__all__ = ["GroundingDinoPrompter", "load_vocab"]
