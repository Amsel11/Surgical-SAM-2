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

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROMPTS_DIR = REPO_ROOT / "prompts"
N_PROMPT_FRAMES = 3
CUTOUT_LONG_SIDE = 200
BOX_LINE_WIDTH = 3
CORNER_MARKER_RADIUS = 8
CORNER_MARKER_COLOR = (255, 255, 50)
SEED = 1  # MVP: single seed. Multi-seed flow in a later iteration.
PROMPT_METHOD = "manual_box"

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


def sample_frame_positions(n_total: int, n_samples: int = N_PROMPT_FRAMES) -> list[int]:
    """Uniform stratified sampling: 25/50/75% by default for n_samples=3."""
    return [int(round((i + 1) * n_total / (n_samples + 1))) for i in range(n_samples)]


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
    """Next video without a (seed=SEED, prompt_method=PROMPT_METHOD) prompt_set."""
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


def count_remaining(conn) -> int:
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
    """Atomically write prompts JSON + insert prompt_set + prompt_objects rows."""
    video_id = state["video_id"]
    objects = state["objects"]
    if not objects:
        raise ValueError("No objects clicked; nothing to save.")

    objects_by_frame: dict[str, list[dict]] = {}
    for frame_pos, source_idx in enumerate(state["source_frame_indices"]):
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
            objects_by_frame[str(source_idx)] = per_frame

    prompt_source_indices_used = [
        state["source_frame_indices"][i]
        for i in range(len(state["source_frame_indices"]))
        if str(state["source_frame_indices"][i]) in objects_by_frame
    ]

    first_image = safe_load_image(state["frame_paths"][0])
    h, w = first_image.shape[:2]

    payload = {
        "video": video_id,
        "resolution": [w, h],
        "n_frames": len(state["frame_paths"]),
        "prompt_frames": prompt_source_indices_used,
        "objects_by_frame": objects_by_frame,
    }

    PROMPTS_DIR.mkdir(exist_ok=True)
    out_path = PROMPTS_DIR / f"{video_id}_seed{SEED}_{PROMPT_METHOD}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    conn = connect()
    with transaction(conn):
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
        "source_frame_indices": [],  # frame positions in frame_paths we're labelling
        "cur_frame_pos": 0,           # 0..N_PROMPT_FRAMES-1
        "objects": {},                # obj_id -> {instrument_id, boxes_by_frame, cutout, label}
        # New-object mode state:
        "current_obj_id": 1,          # next obj_id to assign for a new instrument
        "current_instrument_id": None,
        "current_instrument_label": "",
        "pending_corner": None,       # (x, y) of first corner of an in-progress box
        "pending_box": None,          # [x1,y1,x2,y2] drawn but waiting for instrument label
        # Reboxing-mode state (frames 2/3):
        "rebox_queue": [],            # remaining obj_ids to rebox on this frame
        "rebox_current": None,        # currently reboxing this one, or None = new-object mode
        # Undo log: list of ("rebox"|"new", obj_id, frame_pos) in commit order.
        "commit_log": [],
        "status_msg": "",
    }


def objs_needing_rebox(state: dict) -> list[int]:
    """Existing obj_ids that don't yet have a box on the current frame, in order."""
    cur = state["cur_frame_pos"]
    return [oid for oid, obj in sorted(state["objects"].items()) if cur not in obj["boxes_by_frame"]]


def setup_rebox_for_current_frame(state: dict) -> None:
    """Initialise the rebox queue when moving to a new frame.
       First existing-but-unboxed obj becomes rebox_current; rest queue up.
    """
    queue = objs_needing_rebox(state)
    state["rebox_current"] = queue.pop(0) if queue else None
    state["rebox_queue"] = queue
    state["pending_corner"] = None
    state["pending_box"] = None
    state["current_instrument_id"] = None
    state["current_instrument_label"] = ""


def advance_rebox_or_finish(state: dict) -> None:
    state["rebox_current"] = state["rebox_queue"].pop(0) if state["rebox_queue"] else None


def commit_pending_new_obj(state: dict) -> None:
    """Promote pending_box -> a new obj with current_instrument_id. Logged for undo."""
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
    state["current_instrument_id"] = None
    state["current_instrument_label"] = ""


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


def render_current_frame(state: dict) -> np.ndarray:
    """Draw all finalized boxes + the pending first corner (if any) on top of the raw frame."""
    if not state["frame_paths"]:
        return np.zeros((480, 720, 3), dtype=np.uint8)
    idx = state["source_frame_indices"][state["cur_frame_pos"]]
    img = safe_load_image(state["frame_paths"][idx])
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for obj_id, obj in state["objects"].items():
        box = obj["boxes_by_frame"].get(state["cur_frame_pos"])
        if not box:
            continue
        color = obj_color(obj_id)
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_LINE_WIDTH)
        # tiny solid label corner with the obj_id
        label = f"{obj_id}: {obj.get('label') or obj['instrument_id']}"
        tx, ty = x1 + 4, max(0, y1 - 16)
        draw.rectangle([tx - 2, ty - 2, tx + 8 * len(label), ty + 12], fill=color)
        draw.text((tx, ty), label, fill=(0, 0, 0))
    pb = state.get("pending_box")
    if pb is not None:
        x1, y1, x2, y2 = pb
        draw.rectangle([x1, y1, x2, y2], outline=CORNER_MARKER_COLOR, width=BOX_LINE_WIDTH)
        draw.text((x1 + 4, max(0, y1 - 14)), "pick instrument ->", fill=CORNER_MARKER_COLOR)
    pc = state.get("pending_corner")
    if pc is not None:
        x, y = pc
        r = CORNER_MARKER_RADIUS
        draw.line([x - r, y, x + r, y], fill=CORNER_MARKER_COLOR, width=2)
        draw.line([x, y - r, x, y + r], fill=CORNER_MARKER_COLOR, width=2)
        draw.text((x + r + 4, y - r), "click opposite corner", fill=CORNER_MARKER_COLOR)
    return np.array(pil)


def build_cutout_gallery(state: dict) -> list[tuple[np.ndarray, str]]:
    items: list[tuple[np.ndarray, str]] = []
    for obj_id, obj in state["objects"].items():
        if obj.get("cutout") is not None:
            label = f"Obj {obj_id}: {obj.get('label') or obj['instrument_id']}"
            items.append((obj["cutout"], label))
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

    rebox = state.get("rebox_current")
    if rebox is not None:
        obj = state["objects"][rebox]
        n_more = len(state["rebox_queue"])
        instr = obj.get("label") or obj["instrument_id"]
        if state.get("pending_corner") is not None:
            phase = "click 2nd corner"
        else:
            phase = "click 1st corner"
        return f"{head}\n**Re-box obj {rebox}: {instr}** ({n_more} more after this).  Phase: {phase}.{remaining_str}"

    # New-object mode
    if state.get("pending_box") is not None:
        return f"{head}\n**New object**: box drawn -> pick instrument from dropdown to commit.{remaining_str}"
    if state.get("pending_corner") is not None:
        return f"{head}\n**New object**: click 2nd corner.{remaining_str}"
    next_obj = state["current_obj_id"]
    while next_obj in state["objects"]:
        next_obj += 1
    return f"{head}\n**New object** (will be obj {next_obj}): click 1st corner. Or 'Next frame'/'Save & next video'.{remaining_str}"


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def handler_start(state: dict):
    state = start_new_video(state)
    img = render_current_frame(state) if state["video_id"] else None
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), img, build_cutout_gallery(state)


def handler_pick_instrument(state: dict, instrument_label_value):
    """Dropdown handler. In new-object mode + pending_box set -> commits.
       In reboxing mode -> ignored (instrument is inherited).
    """
    rem = count_remaining(connect())
    if state.get("rebox_current") is not None:
        # No-op in reboxing mode.
        return state, status_text(state, remaining=rem), gr.update()
    if not instrument_label_value:
        state["current_instrument_id"] = None
        state["current_instrument_label"] = ""
        return state, status_text(state, remaining=rem), gr.update()
    state["current_instrument_id"] = instrument_label_value
    for label, value in load_instrument_choices():
        if value == instrument_label_value:
            state["current_instrument_label"] = label
            break
    # Auto-commit if a box is waiting.
    if state.get("pending_box") is not None:
        commit_pending_new_obj(state)
        return state, status_text(state, remaining=rem), gr.update(value=None)
    return state, status_text(state, remaining=rem), gr.update()


def handler_image_click(state: dict, evt: gr.SelectData):
    if not state.get("video_id"):
        return state, "Start a session first.", None, []

    x, y = float(evt.index[0]), float(evt.index[1])
    frame_pos = state["cur_frame_pos"]
    rebox = state.get("rebox_current")
    rem_count = count_remaining(connect())

    # --- Reboxing mode: 2 clicks -> commit to existing obj, advance queue.
    if rebox is not None:
        if state["pending_corner"] is None:
            state["pending_corner"] = (x, y)
        else:
            x1, y1 = state["pending_corner"]
            box = normalize_box(x1, y1, x, y)
            state["objects"][rebox]["boxes_by_frame"][frame_pos] = box
            state["commit_log"].append(("rebox", rebox, frame_pos))
            state["pending_corner"] = None
            advance_rebox_or_finish(state)
        return state, status_text(state, remaining=rem_count), render_current_frame(state), build_cutout_gallery(state)

    # --- New-object mode: refuse new clicks if a pending_box is still awaiting instrument.
    if state.get("pending_box") is not None:
        return (
            state,
            "Pick an instrument from the dropdown for the drawn box (or Undo to redraw).\n\n"
            + status_text(state, remaining=rem_count),
            render_current_frame(state),
            build_cutout_gallery(state),
        )

    if state["pending_corner"] is None:
        state["pending_corner"] = (x, y)
    else:
        x1, y1 = state["pending_corner"]
        state["pending_box"] = normalize_box(x1, y1, x, y)
        state["pending_corner"] = None
        # If user pre-selected an instrument, auto-commit.
        if state.get("current_instrument_id"):
            commit_pending_new_obj(state)

    return state, status_text(state, remaining=rem_count), render_current_frame(state), build_cutout_gallery(state)


def handler_skip_rebox(state: dict):
    """In reboxing mode: skip the current obj on this frame (it's not visible here)."""
    if state.get("rebox_current") is None:
        rem = count_remaining(connect())
        return state, "Not in re-boxing mode — nothing to skip.\n\n" + status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state)
    state["pending_corner"] = None
    advance_rebox_or_finish(state)
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state)


def handler_undo(state: dict):
    """Undo priority:
       1. pending_box  -> clear it (last box drawn but not labelled)
       2. pending_corner -> clear it (first corner of an in-progress box)
       3. Most recent commit ON THIS FRAME -> reverse it (rebox: put back as current; new: delete obj if empty)
    """
    if state.get("pending_box") is not None:
        state["pending_box"] = None
    elif state.get("pending_corner") is not None:
        state["pending_corner"] = None
    else:
        cur = state["cur_frame_pos"]
        # Scan commit_log backwards for most recent entry on this frame.
        for i in range(len(state["commit_log"]) - 1, -1, -1):
            action, obj_id, fp = state["commit_log"][i]
            if fp != cur:
                continue
            state["commit_log"].pop(i)
            obj = state["objects"].get(obj_id)
            if obj is not None:
                obj["boxes_by_frame"].pop(cur, None)
                if action == "rebox":
                    # Put obj_id back as current rebox target.
                    if state["rebox_current"] is not None:
                        state["rebox_queue"].insert(0, state["rebox_current"])
                    state["rebox_current"] = obj_id
                elif action == "new":
                    if not obj["boxes_by_frame"]:
                        del state["objects"][obj_id]
            break
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state)


def handler_next_frame(state: dict):
    if not state.get("video_id"):
        return state, "Start a session first.", None, []
    state["cur_frame_pos"] = min(state["cur_frame_pos"] + 1, len(state["source_frame_indices"]) - 1)
    setup_rebox_for_current_frame(state)
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state)


def handler_resample_frame(state: dict):
    """Replace the source index for the current frame slot with a new random one.
       Drops any boxes already drawn on this slot and re-sets the rebox queue."""
    if not state.get("video_id"):
        return state, "Start a session first.", None, []
    n = len(state["frame_paths"])
    new_idx = random.randint(0, n - 1)
    state["source_frame_indices"][state["cur_frame_pos"]] = new_idx
    cur = state["cur_frame_pos"]
    for obj in state["objects"].values():
        obj["boxes_by_frame"].pop(cur, None)
    # Purge commit_log entries for this frame.
    state["commit_log"] = [e for e in state["commit_log"] if e[2] != cur]
    setup_rebox_for_current_frame(state)
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), render_current_frame(state), build_cutout_gallery(state)


def handler_save_and_next(state: dict):
    if not state.get("video_id"):
        return state, "Nothing to save.", None, []
    if not state["objects"]:
        return state, "No boxes drawn — use Skip video if there's nothing labelable.", render_current_frame(state), []
    try:
        path = save_prompt_set_to_manifest(state)
    except Exception as e:
        return state, f"Save failed: {e}", render_current_frame(state), build_cutout_gallery(state)
    msg = f"Saved {path}. Loading next video..."
    state = start_new_video(state)
    img = render_current_frame(state) if state["video_id"] else None
    rem = count_remaining(connect())
    return state, msg + "\n\n" + status_text(state, remaining=rem), img, build_cutout_gallery(state)


def handler_skip_video(state: dict):
    if not state.get("video_id"):
        return state, "Nothing to skip.", None, []
    skipped_id = state["video_id"]
    mark_video_skipped(skipped_id)
    state = start_new_video(state)
    img = render_current_frame(state) if state["video_id"] else None
    rem = count_remaining(connect())
    msg = f"Skipped {skipped_id}. Loading next..."
    return state, msg + "\n\n" + status_text(state, remaining=rem), img, build_cutout_gallery(state)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    instrument_choices = load_instrument_choices()

    with gr.Blocks(title="SurgSAM-2 Click Collector") as demo:
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
            with gr.Column(scale=3):
                image = gr.Image(label="Click two opposite corners to draw a box", interactive=False, height=540)
                with gr.Row():
                    instrument_dd = gr.Dropdown(
                        choices=instrument_choices,
                        label="Instrument for the drawn box (new-object mode only)",
                        value=None,
                        interactive=True,
                    )
                with gr.Row():
                    btn_skip_rebox = gr.Button("Skip this obj on this frame (not visible)")
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
                cutout_gallery = gr.Gallery(label="Objects boxed", columns=1, height=540, allow_preview=False)

        state = gr.State(empty_state())

        btn_start.click(handler_start, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        instrument_dd.change(handler_pick_instrument, inputs=[state, instrument_dd], outputs=[state, status_md, instrument_dd])
        image.select(handler_image_click, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        btn_skip_rebox.click(handler_skip_rebox, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        btn_undo.click(handler_undo, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        btn_resample.click(handler_resample_frame, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        btn_next_frame.click(handler_next_frame, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        btn_save.click(handler_save_and_next, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        btn_skip.click(handler_skip_video, inputs=[state], outputs=[state, status_md, image, cutout_gallery])

    return demo


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Gradio click collector")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--host", default="0.0.0.0", help="Bind interface. Use 0.0.0.0 on HPC for tunnel access.")
    args = p.parse_args(argv)

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
