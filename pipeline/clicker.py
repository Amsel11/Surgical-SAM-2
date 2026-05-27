"""Gradio click collector for SurgSAM-2 prompts (box mode).

Replaces collect_prompts.ipynb. For each pending (video, seed=1) pair in the
manifest, samples 3 prompt frames at 25/50/75% and lets the user draw a
bounding box around each instrument (2 clicks: opposite corners). Each box is
labelled with an instrument from the controlled vocabulary. Saves prompts JSON
+ manifest rows atomically on "Save & next video".

Why boxes, not single points: SAM/SAM2 reports box-prompt IoU ~75-85% vs
single-point ~50-65%, and the gap widens for elongated objects (surgical
instruments). Boxes also match what YOLO/DINO auto-prompting will produce
later, so manual vs auto prompt_method comparisons stay apples-to-apples.

Design choices:
- Box per object per frame (not multiple boxes / object / frame).
- Reference cutouts = the box content cropped from the source frame, resized to
  200 px on long side. Cheap, GPU-free, gives the user a real picture of
  "what was Obj 1" when labelling frame 2/3.
- One commit point per video: writes prompt_set + prompt_objects + JSON in
  one transaction on Save. Partial work is in-memory only.
- Empty-frame fallback: re-roll a random index for the current slot.
- Skip video: marks (video, seed=1, manual_box) as failed; won't reappear.
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFile

from .db import REPO_ROOT, connect, transaction
from .io import N_PROMPT_FRAMES, sample_frame_positions
from .io import PROMPTS_DIR as _DEFAULT_PROMPTS_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Module-mutable; the --prompts-dir CLI flag rebinds this without affecting
# pipeline.io.PROMPTS_DIR (which the dino prompter reads).
PROMPTS_DIR: Path = _DEFAULT_PROMPTS_DIR
CUTOUT_LONG_SIDE = 220
BOX_LINE_WIDTH = 3
CORNER_MARKER_RADIUS = 8
CORNER_MARKER_COLOR = (255, 255, 50)

# Write target — overridden by CLI flags in main(). Keep as module-level
# globals so existing functions don't need an extra Config parameter.
SEED: int = 1
PROMPT_METHOD: str = "manual_box"
DRY_RUN: bool = False    # if True, Save buttons are no-ops with a UI notice
# Edit mode: "next pending" finds saved sets that still have unknown_instrument
# rows (Phase 0b relabel pass) instead of unsaved videos. Existing boxes load
# onto the anchor frames so the dropdown can assign labels in-place and missed
# arms can be boxed without redoing the geometry.
EDIT_MODE: bool = False

# Per-object box colors; cycled by obj_id.
OBJ_COLORS = [
    (60, 200, 255),   # cyan
    (255, 150, 60),   # orange
    (180, 255, 60),   # lime
    (255, 100, 200),  # pink
    (150, 120, 255),  # purple
    (255, 230, 60),   # yellow
    (60, 255, 180),   # teal
    (255, 80, 100),   # red
]


def obj_color(obj_id: int) -> tuple[int, int, int]:
    return OBJ_COLORS[(int(obj_id) - 1) % len(OBJ_COLORS)]


def normalize_box(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Return xyxy with x1<x2, y1<y2 regardless of click order."""
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def safe_load_image(path: str | Path) -> np.ndarray:
    """Read fully into memory then decode. Dodges PIL's streaming truncation on GPFS."""
    with open(path, "rb") as f:
        data = f.read()
    img = Image.open(io.BytesIO(data))
    img.load()
    return np.array(img.convert("RGB"))


def list_frame_paths(frames_dir: str | Path) -> list[str]:
    p = Path(frames_dir)
    pngs = sorted(p.glob("*.png"))
    if pngs:
        return [str(f) for f in pngs]
    return [str(f) for f in sorted(p.glob("*.jpg"))]


def parse_source_idx(frame_path: str | Path) -> int | None:
    """Extract source-index from filename like 'frame_00_src00000315.png' -> 315.

    For local-sampled frames where the local-position differs from bp's frame index,
    this lets us emit the CORRECT bp index in the saved prompts JSON. Returns None
    for files that don't carry the marker (i.e. bp's own full extracts) — caller
    falls back to the in-array position in that case.
    """
    import re
    m = re.search(r"src(\d+)", Path(frame_path).name)
    return int(m.group(1)) if m else None


def extract_box_cutout(image: np.ndarray, box: list[float], long_side: int = CUTOUT_LONG_SIDE) -> np.ndarray | None:
    """Crop the box region from `image` and resize so the long side = long_side px."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    crop = image[yi1:yi2, xi1:xi2]
    ch, cw = crop.shape[:2]
    scale = long_side / max(ch, cw)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    pil = Image.fromarray(crop).resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil)


# ---------------------------------------------------------------------------
# Manifest access
# ---------------------------------------------------------------------------

def load_instrument_choices() -> list[tuple[str, str]]:
    """Returns [(display label, instrument_id), ...] sorted by category."""
    conn = connect()
    rows = conn.execute(
        "SELECT instrument_id, display_name, category FROM instruments ORDER BY category, instrument_id"
    ).fetchall()
    return [(f"[{r['category']}] {r['display_name']}", r["instrument_id"]) for r in rows]


def fetch_next_pending_video(conn) -> dict | None:
    """Next video to work on, mode-dependent.

    Normal mode: video has no (seed=SEED, prompt_method=PROMPT_METHOD) prompt_set.
    Edit mode: video has a 'ready' prompt_set for (SEED, PROMPT_METHOD) with at
    least one prompt_objects.instrument_id = 'unknown_instrument'."""
    if EDIT_MODE:
        row = conn.execute(
            """
            SELECT v.video_id, v.frames_dir, v.n_frames
            FROM videos v
            JOIN prompt_sets ps
              ON ps.video_id = v.video_id
             AND ps.seed = ?
             AND ps.prompt_method = ?
             AND ps.status = 'ready'
            WHERE EXISTS (
                SELECT 1 FROM prompt_objects po
                WHERE po.prompt_set_id = ps.prompt_set_id
                  AND po.instrument_id = 'unknown_instrument'
            )
            ORDER BY v.video_id
            LIMIT 1
            """,
            (SEED, PROMPT_METHOD),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT v.video_id, v.frames_dir, v.n_frames
            FROM videos v
            WHERE NOT EXISTS (
                SELECT 1 FROM prompt_sets ps
                WHERE ps.video_id = v.video_id
                  AND ps.seed = ?
                  AND ps.prompt_method = ?
            )
            ORDER BY v.video_id
            LIMIT 1
            """,
            (SEED, PROMPT_METHOD),
        ).fetchone()
    return dict(row) if row else None


def fetch_video_by_id(conn, video_id: str) -> dict | None:
    row = conn.execute(
        "SELECT video_id, frames_dir, n_frames FROM videos WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    return dict(row) if row else None


def fetch_video_table_rows() -> list[list]:
    """Build the sidebar rows: every video with its (seed=SEED, method=PROMPT_METHOD) prompt_set status.

    Normal mode order: pending → failed → ready (so unclicked videos surface first).
    Edit mode order: ready-with-unknowns → ready-fully-labelled → pending → failed
    (so the relabel pass surfaces first).

    The marker shows '?' for ready-but-still-has-unknown_instrument rows so the
    table communicates relabel progress regardless of mode.
    """
    conn = connect()
    rows = conn.execute(
        """
        SELECT v.video_id,
               v.n_frames,
               COALESCE(ps.n_objects, 0)            AS n_objects,
               COALESCE(ps.status, 'pending')       AS status,
               COALESCE((
                   SELECT SUM(CASE WHEN po.instrument_id='unknown_instrument' THEN 1 ELSE 0 END)
                   FROM prompt_objects po
                   WHERE po.prompt_set_id = ps.prompt_set_id
               ), 0)                                AS n_unknown
        FROM videos v
        LEFT JOIN prompt_sets ps
          ON ps.video_id = v.video_id
         AND ps.seed = ?
         AND ps.prompt_method = ?
        """,
        (SEED, PROMPT_METHOD),
    ).fetchall()

    def sort_key(r):
        status = r["status"]
        n_unknown = r["n_unknown"] or 0
        if EDIT_MODE:
            if status == "ready" and n_unknown > 0:
                bucket = 0
            elif status == "ready":
                bucket = 1
            elif status == "pending":
                bucket = 2
            elif status == "failed":
                bucket = 3
            else:
                bucket = 4
        else:
            if status == "pending":
                bucket = 0
            elif status == "failed":
                bucket = 1
            elif status == "ready":
                bucket = 2
            else:
                bucket = 3
        return (bucket, r["video_id"])

    rows = sorted(rows, key=sort_key)

    out = []
    for r in rows:
        status = r["status"]
        n_unknown = r["n_unknown"] or 0
        if status == "ready" and n_unknown > 0:
            mark = "?"
        elif status == "ready":
            mark = "✓"
        elif status == "failed":
            mark = "✗"
        else:
            mark = " "
        out.append([mark, r["video_id"], int(r["n_objects"]), int(r["n_frames"])])
    return out


def count_remaining(conn) -> int:
    """In normal mode: videos with no prompt_set yet.
    In edit mode: videos with a ready prompt_set that still has unknowns."""
    if EDIT_MODE:
        return conn.execute(
            """
            SELECT COUNT(DISTINCT v.video_id) FROM videos v
            JOIN prompt_sets ps
              ON ps.video_id = v.video_id
             AND ps.seed = ?
             AND ps.prompt_method = ?
             AND ps.status = 'ready'
            WHERE EXISTS (
                SELECT 1 FROM prompt_objects po
                WHERE po.prompt_set_id = ps.prompt_set_id
                  AND po.instrument_id = 'unknown_instrument'
            )
            """,
            (SEED, PROMPT_METHOD),
        ).fetchone()[0]
    return conn.execute(
        """
        SELECT COUNT(*) FROM videos v
        WHERE NOT EXISTS (
            SELECT 1 FROM prompt_sets ps
            WHERE ps.video_id = v.video_id
              AND ps.seed = ?
              AND ps.prompt_method = ?
        )
        """,
        (SEED, PROMPT_METHOD),
    ).fetchone()[0]


def save_prompt_set_to_manifest(state: dict) -> str:
    """Atomically write prompts JSON + insert prompt_set + prompt_objects rows.

    Under DRY_RUN, returns a placeholder string without touching disk or DB —
    so the user can exercise the click flow on a contested seed without
    overwriting it.
    """
    video_id = state["video_id"]
    objects = state["objects"]
    if not objects:
        raise ValueError("No objects clicked; nothing to save.")
    if DRY_RUN:
        return f"[DRY RUN] would save {video_id} seed={SEED} method={PROMPT_METHOD} with {len(objects)} objs"

    # Map each local-array index to the TRUE source-frame index that bp will use.
    # For locally-subsampled frames the filename carries the source idx; for full
    # bp extracts the local position IS the source idx (parse returns None then).
    def resolve_src(arr_idx: int) -> int:
        parsed = parse_source_idx(state["frame_paths"][arr_idx])
        return parsed if parsed is not None else arr_idx

    objects_by_frame: dict[str, list[dict]] = {}
    for frame_pos, arr_idx in enumerate(state["source_frame_indices"]):
        actual_src = resolve_src(arr_idx)
        per_frame: list[dict] = []
        for obj_id, obj in objects.items():
            box = obj["boxes_by_frame"].get(frame_pos)
            if not box:
                continue
            per_frame.append({
                "obj_id": int(obj_id),
                "box": [float(v) for v in box],
                "positive": [],
                "negative": [],
            })
        if per_frame:
            objects_by_frame[str(actual_src)] = per_frame

    prompt_source_indices_used = []
    for i in range(len(state["source_frame_indices"])):
        actual_src = resolve_src(state["source_frame_indices"][i])
        if str(actual_src) in objects_by_frame:
            prompt_source_indices_used.append(actual_src)

    first_image = safe_load_image(state["frame_paths"][0])
    h, w = first_image.shape[:2]

    # n_frames should be the FULL video's frame count (bp's count), pulled from
    # the manifest videos row at save time so the JSON is correct downstream.
    conn_q = connect()
    row = conn_q.execute("SELECT n_frames FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    bp_n_frames = row["n_frames"] if row and row["n_frames"] else len(state["frame_paths"])

    payload = {
        "video": video_id,
        "resolution": [w, h],
        "n_frames": bp_n_frames,
        "prompt_frames": prompt_source_indices_used,
        "objects_by_frame": objects_by_frame,
    }

    PROMPTS_DIR.mkdir(exist_ok=True)
    out_path = PROMPTS_DIR / f"{video_id}_seed{SEED}_{PROMPT_METHOD}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    conn = connect()
    with transaction(conn):
        # UPSERT: if a prompt_set already exists for this (video, seed, method),
        # delete its prompt_objects and replace the row in place. Otherwise insert.
        existing = conn.execute(
            "SELECT prompt_set_id FROM prompt_sets WHERE video_id=? AND seed=? AND prompt_method=?",
            (video_id, SEED, PROMPT_METHOD),
        ).fetchone()
        if existing:
            ps_id = existing["prompt_set_id"]
            conn.execute("DELETE FROM prompt_objects WHERE prompt_set_id=?", (ps_id,))
            conn.execute(
                """
                UPDATE prompt_sets SET
                    prompts_path=?, n_objects=?, n_prompt_frames=?,
                    status='ready', created_by='gradio-clicker'
                WHERE prompt_set_id=?
                """,
                (str(out_path), len(objects), len(prompt_source_indices_used), ps_id),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO prompt_sets
                    (video_id, seed, prompt_method, prompts_path, n_objects, n_prompt_frames, status, created_by)
                VALUES (?, ?, ?, ?, ?, ?, 'ready', 'gradio-clicker')
                """,
                (video_id, SEED, PROMPT_METHOD, str(out_path), len(objects), len(prompt_source_indices_used)),
            )
            ps_id = cur.lastrowid
        for obj_id, obj in objects.items():
            conn.execute(
                "INSERT INTO prompt_objects (prompt_set_id, obj_id, instrument_id) VALUES (?, ?, ?)",
                (ps_id, int(obj_id), obj["instrument_id"]),
            )
    return str(out_path)


def mark_video_skipped(video_id: str) -> None:
    """Record a 'failed' prompt_set so we don't re-show this video."""
    if DRY_RUN:
        return
    conn = connect()
    PROMPTS_DIR.mkdir(exist_ok=True)
    conn.execute(
        """
        INSERT INTO prompt_sets (video_id, seed, prompt_method, status, created_by, notes)
        VALUES (?, ?, ?, 'failed', 'gradio-clicker', 'skipped: no usable instruments')
        ON CONFLICT(video_id, seed, prompt_method) DO UPDATE SET status='failed'
        """,
        (video_id, SEED, PROMPT_METHOD),
    )


# ---------------------------------------------------------------------------
# State + rendering
# ---------------------------------------------------------------------------

def empty_state() -> dict:
    return {
        "video_id": None,
        "frames_dir": None,
        "frame_paths": [],
        "source_frame_indices": [],
        "cur_frame_pos": 0,
        "objects": {},                # obj_id -> {instrument_id, boxes_by_frame, cutout, label}
        # active_obj_id: which obj the next-drawn box will UPDATE. None = create new obj.
        "active_obj_id": None,
        # Next obj_id to assign when creating new. Auto-increments past existing keys.
        "current_obj_id": 1,
        "current_instrument_id": "unknown_instrument",
        "current_instrument_label": "[other] Unknown / Unidentified",
        "pending_corner": None,
        "pending_box": None,
        # Undo log: list of (action, obj_id, frame_pos, ...payload). Payload differs:
        #   ("new",    obj_id, frame_pos)              -> undo deletes obj if empty
        #   ("update", obj_id, frame_pos, old_box)     -> undo restores old_box (or removes if None)
        "commit_log": [],
        "status_msg": "",
    }


def point_in_box(box: list[float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def find_obj_at_point(state: dict, x: float, y: float) -> int | None:
    """Return obj_id of an existing box on the CURRENT frame that contains (x,y), else None.
       If multiple overlap, prefer the smallest box (likely the one user meant)."""
    cur = state["cur_frame_pos"]
    hits: list[tuple[int, int]] = []  # (obj_id, area)
    for obj_id, obj in state["objects"].items():
        box = obj["boxes_by_frame"].get(cur)
        if box and point_in_box(box, x, y):
            area = (box[2] - box[0]) * (box[3] - box[1])
            hits.append((obj_id, area))
    if not hits:
        return None
    hits.sort(key=lambda t: t[1])
    return hits[0][0]


def commit_pending_new_obj(state: dict) -> int:
    """Promote pending_box -> a new obj with current_instrument_id. Logged for undo.
       Returns the new obj_id."""
    obj_id = state["current_obj_id"]
    while obj_id in state["objects"]:
        obj_id += 1
    box = state["pending_box"]
    frame_pos = state["cur_frame_pos"]
    state["objects"][obj_id] = {
        "instrument_id": state["current_instrument_id"],
        "label": state["current_instrument_label"],
        "boxes_by_frame": {frame_pos: box},
        "cutout": None,
    }
    idx = state["source_frame_indices"][frame_pos]
    cur_img = safe_load_image(state["frame_paths"][idx])
    cutout = extract_box_cutout(cur_img, box)
    if cutout is not None:
        state["objects"][obj_id]["cutout"] = cutout
    state["commit_log"].append(("new", obj_id, frame_pos))
    state["current_obj_id"] = obj_id + 1
    state["pending_box"] = None
    state["pending_corner"] = None
    return obj_id


def update_active_obj_box(state: dict, box: list[float]) -> None:
    """Set/replace the active obj's box on the current frame. Captures old box for undo."""
    obj_id = state["active_obj_id"]
    frame_pos = state["cur_frame_pos"]
    obj = state["objects"][obj_id]
    old_box = obj["boxes_by_frame"].get(frame_pos)
    obj["boxes_by_frame"][frame_pos] = box
    state["commit_log"].append(("update", obj_id, frame_pos, old_box))
    # Refresh cutout if obj didn't have one yet.
    if obj.get("cutout") is None:
        idx = state["source_frame_indices"][frame_pos]
        cur_img = safe_load_image(state["frame_paths"][idx])
        co = extract_box_cutout(cur_img, box)
        if co is not None:
            obj["cutout"] = co


def start_new_video(state: dict) -> dict:
    """Load the next pending video from manifest into state."""
    conn = connect()
    row = fetch_next_pending_video(conn)
    if row is None:
        state = empty_state()
        state["status_msg"] = "All videos done! Nothing more to click."
        return state

    frames = list_frame_paths(row["frames_dir"])
    if not frames:
        # frames dir is empty for some reason — mark skipped and try again
        mark_video_skipped(row["video_id"])
        return start_new_video(state)

    positions = sample_frame_positions(len(frames), N_PROMPT_FRAMES)
    state = empty_state()
    state["video_id"] = row["video_id"]
    state["frames_dir"] = row["frames_dir"]
    state["frame_paths"] = frames
    state["source_frame_indices"] = positions
    state["status_msg"] = ""
    return state


ACTIVE_BORDER_COLOR = (255, 140, 0)  # bright orange


def render_current_frame(state: dict) -> np.ndarray:
    """Draw all boxes for the current frame; active obj gets a thick orange border."""
    if not state["frame_paths"]:
        return np.zeros((480, 720, 3), dtype=np.uint8)
    idx = state["source_frame_indices"][state["cur_frame_pos"]]
    img = safe_load_image(state["frame_paths"][idx])
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    active = state.get("active_obj_id")
    for obj_id, obj in state["objects"].items():
        box = obj["boxes_by_frame"].get(state["cur_frame_pos"])
        if not box:
            continue
        x1, y1, x2, y2 = box
        is_active = obj_id == active
        if is_active:
            # Thick orange outer border + inner color border so user sees BOTH the obj's color and the active highlight
            draw.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=ACTIVE_BORDER_COLOR, width=4)
            draw.rectangle([x1, y1, x2, y2], outline=obj_color(obj_id), width=BOX_LINE_WIDTH)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=obj_color(obj_id), width=BOX_LINE_WIDTH)
        label_color = ACTIVE_BORDER_COLOR if is_active else obj_color(obj_id)
        label = f"{obj_id}: {obj.get('label') or obj['instrument_id']}"
        tx, ty = x1 + 4, max(0, y1 - 16)
        draw.rectangle([tx - 2, ty - 2, tx + 8 * len(label), ty + 12], fill=label_color)
        draw.text((tx, ty), label, fill=(0, 0, 0))
    pc = state.get("pending_corner")
    if pc is not None:
        x, y = pc
        r = CORNER_MARKER_RADIUS
        draw.line([x - r, y, x + r, y], fill=CORNER_MARKER_COLOR, width=2)
        draw.line([x, y - r, x, y + r], fill=CORNER_MARKER_COLOR, width=2)
        hint = "click opposite corner -> updates Obj " + str(active) if active else "click opposite corner -> NEW obj"
        draw.text((x + r + 4, y - r), hint, fill=CORNER_MARKER_COLOR)
    return np.array(pil)


def build_cutout_gallery(state: dict) -> list[tuple[np.ndarray, str]]:
    """Each entry's caption shows frame-count + ACTIVE marker. Click selects."""
    items: list[tuple[np.ndarray, str]] = []
    total_frames = len(state["source_frame_indices"]) or 1
    active = state.get("active_obj_id")
    for obj_id, obj in state["objects"].items():
        if obj.get("cutout") is None:
            continue
        n_frames = len(obj["boxes_by_frame"])
        active_mark = "★ " if obj_id == active else ""
        inst = obj.get("label") or obj["instrument_id"]
        caption = f"{active_mark}Obj {obj_id}: {inst}  ({n_frames}/{total_frames})"
        items.append((obj["cutout"], caption))
    return items


def status_text(state: dict, remaining: int | None = None) -> str:
    if not state.get("video_id"):
        return state.get("status_msg") or "Click 'Start session' to begin."
    vid = state["video_id"]
    pos = state["cur_frame_pos"]
    total = len(state["source_frame_indices"])
    src_idx = state["source_frame_indices"][pos] if state["source_frame_indices"] else "?"
    n_frames = len(state["frame_paths"])
    n_objs = len(state["objects"])
    remaining_str = f"  |  {remaining} videos remaining" if remaining is not None else ""
    head = f"**{vid}**  |  frame {pos+1}/{total}  (source idx {src_idx} of {n_frames})  |  {n_objs} objects total"

    active = state.get("active_obj_id")
    phase_drawing = "click 2nd corner" if state.get("pending_corner") else "click 1st corner"
    if active is not None:
        obj = state["objects"][active]
        n_marked = len(obj["boxes_by_frame"])
        instr = obj.get("label") or obj["instrument_id"]
        return f"{head}\n🟠 **Active: Obj {active}** ({instr}, {n_marked}/{total} frames). Drawing a box will UPDATE this obj on the current frame. To create a NEW obj instead, click **Deselect**. ({phase_drawing})"
    next_obj = state["current_obj_id"]
    while next_obj in state["objects"]:
        next_obj += 1
    return f"{head}\nNo active obj — drawing a box will create **NEW Obj {next_obj}** with instrument = current dropdown ({state.get('current_instrument_label') or '(none)'}). Click an existing box (or its cutout) to make it active. ({phase_drawing})"


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def handler_start(state: dict):
    state = start_new_video(state)
    img = render_current_frame(state) if state["video_id"] else None
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), img, build_cutout_gallery(state), fetch_video_table_rows(), active_dd_update(state)


def start_specific_video(state: dict, video_id: str) -> tuple[dict, str | None]:
    """Load a specific video by id. If a saved prompt_set exists, reconstruct
       its boxes/objects from the JSON so the user can verify / edit incrementally.
       To wipe and start over, use 'Redo this video'."""
    conn = connect()
    row = fetch_video_by_id(conn, video_id)
    if row is None:
        return state, f"Unknown video_id {video_id}."
    frames = list_frame_paths(row["frames_dir"])
    if not frames:
        return state, f"{video_id}: no frames found in {row['frames_dir']}."

    fresh = empty_state()
    fresh["video_id"] = row["video_id"]
    fresh["frames_dir"] = row["frames_dir"]
    fresh["frame_paths"] = frames

    existing = conn.execute(
        "SELECT prompt_set_id, prompts_path, status FROM prompt_sets WHERE video_id=? AND seed=? AND prompt_method=?",
        (video_id, SEED, PROMPT_METHOD),
    ).fetchone()

    if existing and existing["status"] == "ready" and existing["prompts_path"] and Path(existing["prompts_path"]).exists():
        # Load saved data so the user can verify / edit.
        with open(existing["prompts_path"]) as f:
            data = json.load(f)
        # Map bp source idx -> local frame_paths position via 'src(\d+)' in filename.
        bp_to_local: dict[int, int] = {}
        for i, fp in enumerate(frames):
            parsed = parse_source_idx(fp)
            if parsed is not None:
                bp_to_local[parsed] = i
        # Build source_frame_indices from JSON prompt_frames (only those we can resolve locally).
        local_indices: list[int] = []
        for bp_idx in data.get("prompt_frames", []):
            li = bp_to_local.get(int(bp_idx))
            if li is not None and li not in local_indices:
                local_indices.append(li)
        if not local_indices:
            local_indices = sample_frame_positions(len(frames), N_PROMPT_FRAMES)
        fresh["source_frame_indices"] = local_indices

        # Instrument-id -> pretty label
        inst_pretty: dict[str, str] = {v: lab for lab, v in load_instrument_choices()}
        # prompt_objects rows
        po_rows = conn.execute(
            "SELECT obj_id, instrument_id FROM prompt_objects WHERE prompt_set_id=?",
            (existing["prompt_set_id"],),
        ).fetchall()
        obj_instr = {r["obj_id"]: r["instrument_id"] for r in po_rows}

        # Walk the JSON, rebuild objects + boxes_by_frame
        for frame_str, objs in data.get("objects_by_frame", {}).items():
            bp_idx = int(frame_str)
            li = bp_to_local.get(bp_idx)
            if li is None:
                continue
            try:
                frame_pos = local_indices.index(li)
            except ValueError:
                continue
            for o in objs:
                oid = int(o["obj_id"])
                box = o.get("box")
                if not box:
                    continue
                if oid not in fresh["objects"]:
                    instr = obj_instr.get(oid, "unknown_instrument")
                    fresh["objects"][oid] = {
                        "instrument_id": instr,
                        "label": inst_pretty.get(instr, instr),
                        "boxes_by_frame": {},
                        "cutout": None,
                    }
                fresh["objects"][oid]["boxes_by_frame"][frame_pos] = [float(v) for v in box]

        # Generate cutouts from the first box of each obj
        for oid, obj in fresh["objects"].items():
            if not obj["boxes_by_frame"]:
                continue
            first_pos = sorted(obj["boxes_by_frame"].keys())[0]
            first_box = obj["boxes_by_frame"][first_pos]
            idx = fresh["source_frame_indices"][first_pos]
            img = safe_load_image(fresh["frame_paths"][idx])
            co = extract_box_cutout(img, first_box)
            if co is not None:
                obj["cutout"] = co

        # Set current_obj_id past the highest existing
        if fresh["objects"]:
            fresh["current_obj_id"] = max(fresh["objects"].keys()) + 1
        return fresh, None

    # Fresh / not-yet-clicked video
    fresh["source_frame_indices"] = sample_frame_positions(len(frames), N_PROMPT_FRAMES)
    return fresh, None


def handler_select_video_row(state: dict, evt: gr.SelectData):
    """User clicked a row in the video table — load that video.
       evt.index is [row, col]; we read the row from the table data."""
    rows = fetch_video_table_rows()
    if not rows:
        return state, "No videos.", None, [], rows, active_dd_update(state)
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        chosen_video_id = rows[row_idx][1]
    except Exception:
        return state, "Couldn't parse selection.", render_current_frame(state) if state.get("video_id") else None, build_cutout_gallery(state), rows, active_dd_update(state)

    new_state, msg = start_specific_video(state, chosen_video_id)
    rem = count_remaining(connect())
    if msg:
        return state, msg + "\n\n" + status_text(state, remaining=rem), render_current_frame(state) if state.get("video_id") else None, build_cutout_gallery(state), rows, active_dd_update(state)
    return new_state, status_text(new_state, remaining=rem), render_current_frame(new_state), build_cutout_gallery(new_state), rows, active_dd_update(new_state)


def handler_redo_video(state: dict):
    """Force-overwrite the existing prompt_set for the currently-displayed video.
       Sets its DB row to status='pending' (or deletes), clears prompt_objects, then loads fresh."""
    if not state.get("video_id"):
        return state, "No video loaded — select one first.", None, [], fetch_video_table_rows(), active_dd_update(state)
    video_id = state["video_id"]
    conn = connect()
    with transaction(conn):
        row = conn.execute(
            "SELECT prompt_set_id FROM prompt_sets WHERE video_id=? AND seed=? AND prompt_method=?",
            (video_id, SEED, PROMPT_METHOD),
        ).fetchone()
        if row:
            conn.execute("DELETE FROM prompt_objects WHERE prompt_set_id=?", (row["prompt_set_id"],))
            conn.execute("DELETE FROM prompt_sets WHERE prompt_set_id=?", (row["prompt_set_id"],))
        # Also remove the stale prompts JSON so resave is clean
        json_path = PROMPTS_DIR / f"{video_id}_seed{SEED}_{PROMPT_METHOD}.json"
        if json_path.exists():
            json_path.unlink()
    new_state, msg = start_specific_video(state, video_id)
    rem = count_remaining(connect())
    if msg:
        return state, msg, render_current_frame(state) if state.get("video_id") else None, build_cutout_gallery(state), fetch_video_table_rows(), active_dd_update(state)
    return new_state, "Cleared prior labels for " + video_id + ". Re-do from scratch.\n\n" + status_text(new_state, remaining=rem), render_current_frame(new_state), build_cutout_gallery(new_state), fetch_video_table_rows(), active_dd_update(new_state)


def handler_pick_instrument(state: dict, instrument_label_value):
    """Dropdown handler. Two cases:
       - active_obj_id set: rename that obj's instrument label.
       - no active obj: sets current_instrument_id which applies to the NEXT new obj.
    """
    rem = count_remaining(connect())
    if not instrument_label_value:
        if state.get("active_obj_id") is None:
            state["current_instrument_id"] = None
            state["current_instrument_label"] = ""
        return state, status_text(state, remaining=rem), gr.update()
    pretty_label = ""
    for label, value in load_instrument_choices():
        if value == instrument_label_value:
            pretty_label = label
            break
    if state.get("active_obj_id") is not None:
        # Re-label the active obj.
        obj = state["objects"].get(state["active_obj_id"])
        if obj:
            obj["instrument_id"] = instrument_label_value
            obj["label"] = pretty_label
    else:
        state["current_instrument_id"] = instrument_label_value
        state["current_instrument_label"] = pretty_label
    return state, status_text(state, remaining=rem), gr.update()


def handler_image_click(state: dict, evt: gr.SelectData):
    """Image clicks ALWAYS draw — never select. Selection happens via the active-obj
       dropdown / gallery / Deselect button, so clicking on a visible box in the image
       won't trap you in a state you didn't intend."""
    if not state.get("video_id"):
        return state, "Start a session first.", None, [], gr.update()

    x, y = float(evt.index[0]), float(evt.index[1])
    rem_count = count_remaining(connect())

    if state["pending_corner"] is None:
        state["pending_corner"] = (x, y)
        return state, status_text(state, remaining=rem_count), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)

    x1, y1 = state["pending_corner"]
    box = normalize_box(x1, y1, x, y)
    state["pending_corner"] = None
    if state.get("active_obj_id") is not None:
        update_active_obj_box(state, box)
        state["active_obj_id"] = None
    else:
        state["pending_box"] = box
        if state.get("current_instrument_id"):
            commit_pending_new_obj(state)
    return state, status_text(state, remaining=rem_count), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)


def handler_deselect(state: dict):
    """Clear active obj + any pending corner so the next box creates a NEW obj."""
    state["active_obj_id"] = None
    state["pending_corner"] = None
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)


def active_dd_update(state: dict):
    """Reusable gr.update value for the active-obj dropdown, sync'd to current state."""
    return gr.update(choices=build_active_obj_choices(state), value=state.get("active_obj_id"))


def build_active_obj_choices(state: dict) -> list[tuple[str, int | None]]:
    """Choices for the Active-obj dropdown. None = new-obj mode."""
    out: list[tuple[str, int | None]] = [("(none — NEW obj mode)", None)]
    for oid, obj in state["objects"].items():
        n_frames = len(obj["boxes_by_frame"])
        instr = obj.get("label") or obj["instrument_id"]
        total = len(state["source_frame_indices"]) or 1
        out.append((f"Obj {oid}: {instr}  ({n_frames}/{total} frames)", oid))
    return out


def handler_pick_active_obj(state: dict, value):
    """Dropdown alternative to clicking the cutout. value is obj_id int or None."""
    state["active_obj_id"] = value
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), gr.update(choices=build_active_obj_choices(state), value=state.get("active_obj_id"))


def handler_delete_active_obj(state: dict):
    """Remove the active obj entirely — all frames, all boxes — and clear from manifest cutout list.
       Useful when you accidentally created a wrong obj and have already moved frames."""
    active = state.get("active_obj_id")
    if active is None:
        rem = count_remaining(connect())
        return state, "No active obj to delete. Pick one from the dropdown or gallery first.\n\n" + status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), gr.update(choices=build_active_obj_choices(state), value=None)
    # Remove all commit_log entries for this obj (so undo doesn't try to restore it)
    state["commit_log"] = [e for e in state["commit_log"] if e[1] != active]
    state["objects"].pop(active, None)
    state["active_obj_id"] = None
    state["pending_corner"] = None
    rem = count_remaining(connect())
    return state, f"Deleted obj {active}.\n\n" + status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), gr.update(choices=build_active_obj_choices(state), value=None)


def handler_select_obj_from_gallery(state: dict, evt: gr.SelectData):
    """Click a cutout in the side gallery -> make that obj active (or deselect if same)."""
    idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if isinstance(evt.index, (list, tuple)) else 0)
    rendered = [oid for oid, obj in state["objects"].items() if obj.get("cutout") is not None]
    if 0 <= idx < len(rendered):
        chosen = rendered[idx]
        state["active_obj_id"] = None if state.get("active_obj_id") == chosen else chosen
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)


def handler_undo(state: dict):
    """Undo priority:
       1. pending_corner -> clear it
       2. Most recent commit ANYWHERE (any frame) -> reverse it. (Crosses frame boundaries
          so you can fix a wrong-obj you only noticed after advancing.)
    """
    if state.get("pending_corner") is not None:
        state["pending_corner"] = None
    elif state["commit_log"]:
        entry = state["commit_log"].pop()
        action, obj_id, fp = entry[0], entry[1], entry[2]
        obj = state["objects"].get(obj_id)
        if obj is not None:
            if action == "new":
                obj["boxes_by_frame"].pop(fp, None)
                if not obj["boxes_by_frame"]:
                    del state["objects"][obj_id]
                    if state.get("active_obj_id") == obj_id:
                        state["active_obj_id"] = None
            elif action == "update":
                old_box = entry[3] if len(entry) > 3 else None
                if old_box is None:
                    obj["boxes_by_frame"].pop(fp, None)
                else:
                    obj["boxes_by_frame"][fp] = old_box
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)


def handler_next_frame(state: dict):
    if not state.get("video_id"):
        return state, "Start a session first.", None, [], gr.update()
    last = len(state["source_frame_indices"]) - 1
    if state["cur_frame_pos"] >= last:
        # Don't auto-save anymore — user might just be navigating to verify.
        # Loop back to frame 0 so they can cycle through.
        state["cur_frame_pos"] = 0
    else:
        state["cur_frame_pos"] += 1
    state["pending_corner"] = None
    state["active_obj_id"] = None
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)


def handler_prev_frame(state: dict):
    """Go to the previous prompt frame. Wraps from frame 1 -> last frame so you can cycle."""
    if not state.get("video_id"):
        return state, "Start a session first.", None, [], gr.update()
    last = len(state["source_frame_indices"]) - 1
    if state["cur_frame_pos"] <= 0:
        state["cur_frame_pos"] = last
    else:
        state["cur_frame_pos"] -= 1
    state["pending_corner"] = None
    state["active_obj_id"] = None
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)


def handler_resample_frame(state: dict):
    """Replace the source index for the current frame slot. Drops boxes drawn on this slot."""
    if not state.get("video_id"):
        return state, "Start a session first.", None, [], gr.update()
    n = len(state["frame_paths"])
    new_idx = random.randint(0, n - 1)
    state["source_frame_indices"][state["cur_frame_pos"]] = new_idx
    cur = state["cur_frame_pos"]
    for obj in state["objects"].values():
        obj["boxes_by_frame"].pop(cur, None)
    state["commit_log"] = [e for e in state["commit_log"] if e[2] != cur]
    state["pending_corner"] = None
    state["active_obj_id"] = None
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state), active_dd_update(state)


def handler_save_and_next(state: dict):
    table = fetch_video_table_rows
    if not state.get("video_id"):
        return state, "Nothing to save.", None, [], table(), active_dd_update(state)
    if not state["objects"]:
        return state, "No boxes drawn — use Skip video if there's nothing labelable.", render_current_frame(state), [], table(), active_dd_update(state)
    try:
        path = save_prompt_set_to_manifest(state)
    except Exception as e:
        return state, f"Save failed: {e}", render_current_frame(state), build_cutout_gallery(state), table(), active_dd_update(state)
    msg = f"Saved {path}. Loading next video..."
    state = start_new_video(state)
    img = render_current_frame(state) if state["video_id"] else None
    rem = count_remaining(connect())
    return state, msg + "\n\n" + status_text(state, remaining=rem), img, build_cutout_gallery(state), table(), active_dd_update(state)


def handler_skip_video(state: dict):
    if not state.get("video_id"):
        return state, "Nothing to skip.", None, [], fetch_video_table_rows(), active_dd_update(state)
    skipped_id = state["video_id"]
    mark_video_skipped(skipped_id)
    state = start_new_video(state)
    img = render_current_frame(state) if state["video_id"] else None
    rem = count_remaining(connect())
    msg = f"Skipped {skipped_id}. Loading next..."
    return state, msg + "\n\n" + status_text(state, remaining=rem), img, build_cutout_gallery(state), fetch_video_table_rows(), active_dd_update(state)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    instrument_choices = load_instrument_choices()

    # Banner spelling out where this session will write. Visible at the top so
    # the user can confirm before clicking. DRY_RUN paints it red.
    mode_tag = " — **EDIT MODE** (Phase 0b relabel)" if EDIT_MODE else ""
    if DRY_RUN:
        target_banner = (
            f"### ⚠️ DRY RUN — nothing is being saved{mode_tag}\n"
            f"Target (would be): **seed={SEED}, method={PROMPT_METHOD}** "
            f"under `{PROMPTS_DIR}/`. Save buttons are no-ops."
        )
    else:
        target_banner = (
            f"### Writing to: **seed={SEED}, method=`{PROMPT_METHOD}`**{mode_tag} "
            f"under `{PROMPTS_DIR}/`"
        )

    with gr.Blocks(title=f"SurgSAM-2 Clicker (seed={SEED}, method={PROMPT_METHOD})") as demo:
        gr.Markdown(target_banner)
        gr.Markdown(
            "## SurgSAM-2 box collector\n"
            "**Frame 1** (any frame with no prior objects): draw a box around an instrument "
            "(2 clicks, opposite corners) -> pick instrument from dropdown to commit. Repeat for each instrument.\n\n"
            "**Frames 2 & 3**: the app walks you through re-boxing each existing object in turn "
            "(instrument is inherited — no dropdown needed). After all existing objs are done (or skipped), "
            "you can box new objects same as frame 1.\n\n"
            "**Same obj_id across frames** is required for SAM2 to refresh its tracks rather than start new ones."
        )
        status_md = gr.Markdown("Click 'Start session' to begin.")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Videos (click row to load) — ✓ done, ✗ failed, blank = pending")
                video_table = gr.Dataframe(
                    headers=["", "video_id", "objs", "frames"],
                    datatype=["str", "str", "number", "number"],
                    value=fetch_video_table_rows(),
                    interactive=False,
                    wrap=False,
                )
                btn_redo = gr.Button("Redo this video (clears prior labels)", variant="stop")
            with gr.Column(scale=3):
                image = gr.Image(label="Click two opposite corners to draw a box", interactive=False, height=540)
                with gr.Row():
                    instrument_dd = gr.Dropdown(
                        choices=instrument_choices,
                        label="Instrument (default: unknown — auto-applied; change if you know)",
                        value="unknown_instrument",
                        interactive=True,
                    )
                with gr.Row():
                    active_obj_dd = gr.Dropdown(
                        choices=build_active_obj_choices(empty_state()),
                        label="Active obj (target for next box draw)",
                        value=None,
                        interactive=True,
                    )
                with gr.Row():
                    btn_deselect = gr.Button("⬜ DESELECT (back to NEW obj mode)", variant="primary")
                    btn_delete_obj = gr.Button("🗑 Delete active obj (all frames)", variant="stop")
                with gr.Row():
                    btn_prev_frame = gr.Button("← Prev frame")
                    btn_undo = gr.Button("Undo")
                with gr.Row():
                    btn_resample = gr.Button("This frame is empty -> sample a different one")
                    btn_next_frame = gr.Button("Next frame", variant="primary")
                with gr.Row():
                    btn_save = gr.Button("Save & next video", variant="primary")
                    btn_skip = gr.Button("Skip this video (nothing labelable)")
                btn_start = gr.Button("Start session", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Reference: boxes drawn")
                cutout_gallery = gr.Gallery(
                    label="Objects boxed (click to make active, click image to preview bigger)",
                    columns=2,
                    height=540,
                    allow_preview=True,
                    object_fit="contain",
                )

        state = gr.State(empty_state())

        btn_start.click(handler_start, inputs=[state], outputs=[state, status_md, image, cutout_gallery, video_table, active_obj_dd])
        instrument_dd.change(handler_pick_instrument, inputs=[state, instrument_dd], outputs=[state, status_md, instrument_dd])
        image.select(handler_image_click, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        active_obj_dd.change(handler_pick_active_obj, inputs=[state, active_obj_dd], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        btn_deselect.click(handler_deselect, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        btn_delete_obj.click(handler_delete_active_obj, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        cutout_gallery.select(handler_select_obj_from_gallery, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        btn_undo.click(handler_undo, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        btn_resample.click(handler_resample_frame, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        btn_prev_frame.click(handler_prev_frame, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        btn_next_frame.click(handler_next_frame, inputs=[state], outputs=[state, status_md, image, cutout_gallery, active_obj_dd])
        btn_save.click(handler_save_and_next, inputs=[state], outputs=[state, status_md, image, cutout_gallery, video_table, active_obj_dd])
        btn_skip.click(handler_skip_video, inputs=[state], outputs=[state, status_md, image, cutout_gallery, video_table, active_obj_dd])
        video_table.select(handler_select_video_row, inputs=[state], outputs=[state, status_md, image, cutout_gallery, video_table, active_obj_dd])
        btn_redo.click(handler_redo_video, inputs=[state], outputs=[state, status_md, image, cutout_gallery, video_table, active_obj_dd])

    return demo


def main(argv: list[str] | None = None) -> int:
    """Launch the click collector.

    Write-target control (avoids stomping on existing prompt_sets):
      --seed N           : write under seed=N (default 1). Use --seed 2 to
                            iterate on UX without touching seed=1 data.
      --method NAME      : prompt_method tag (default 'manual_box'). Useful
                            for marking experimental runs (e.g.
                            --method manual_box_v2_2026_05_14).
      --prompts-dir DIR  : where JSON files go (default ./prompts).
      --dry-run          : run the full UI but make Save buttons no-ops.
                           Banner turns red. No file or DB write occurs.

    SURGSAM_MANIFEST and other paths are read from .env at the repo root via
    python-dotenv. Without that, pipeline.db falls back to ./manifest.db, which
    is usually NOT the one with your 38 videos in it.
    """
    import argparse
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    global SEED, PROMPT_METHOD, PROMPTS_DIR, DRY_RUN, EDIT_MODE
    p = argparse.ArgumentParser(description="Gradio click collector")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind interface. Use 0.0.0.0 on HPC for tunnel access.")
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"Seed for prompt_sets row (default {SEED}). "
                        f"Use a different value to test without overwriting prior work.")
    p.add_argument("--method", default=PROMPT_METHOD,
                   help=f"prompt_method tag (default {PROMPT_METHOD!r}).")
    p.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR,
                   help=f"Directory for prompts JSON output (default {PROMPTS_DIR}).")
    p.add_argument("--dry-run", action="store_true",
                   help="UI works but Save buttons are no-ops. Nothing is written.")
    p.add_argument("--edit", action="store_true",
                   help="Phase 0b relabel pass. Navigates to existing prompt_sets "
                        "with unknown_instrument rows so the dropdown can assign "
                        "labels in-place. Click a '?' row in the table to load it.")
    args = p.parse_args(argv)

    SEED = args.seed
    PROMPT_METHOD = args.method
    PROMPTS_DIR = args.prompts_dir
    DRY_RUN = args.dry_run
    EDIT_MODE = args.edit

    print(f"Clicker write target: seed={SEED} method={PROMPT_METHOD} dir={PROMPTS_DIR}"
          + ("  [DRY RUN]" if DRY_RUN else "")
          + ("  [EDIT MODE]" if EDIT_MODE else ""))

    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
