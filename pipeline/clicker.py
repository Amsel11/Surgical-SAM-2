"""Gradio click collector for SurgSAM-2 prompts.

Replaces collect_prompts.ipynb. Pulls (video, seed) pairs from the manifest where
no prompt_set exists yet, walks 3 sampled prompt frames per video, lets the
user click positive points and label each object with an instrument from the
controlled vocabulary, then writes a prompts JSON + manifest rows on save.

Design choices:
- Reference cutouts are simple bounding-box crops (120x120 px) around the first
  click for each object. Not real SAM2 masks. Cheap, GPU-free, fast — purpose
  is human consistency ("yes that's the same object I clicked on frame 1"), not
  perfect masks.
- One commit point per video: prompt_set + prompt_objects + JSON written
  atomically on "Save & next video". Partial work is in-memory only.
- Empty-frame handling: if a sampled frame has nothing relevant, click "Skip
  this frame -> next sample" to re-roll a different frame in the same slot.
- Skip video: marks the (video, seed) as failed in the manifest so it gets
  skipped on next session.
"""

from __future__ import annotations

import io
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFile

from .db import REPO_ROOT, connect, transaction

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROMPTS_DIR = REPO_ROOT / "prompts"
N_PROMPT_FRAMES = 3
CUTOUT_SIZE = 120
POINT_RADIUS = 6
POINT_COLOR_POS = (50, 255, 50)
POINT_COLOR_OBJ_BORDER = (255, 255, 255)
SEED = 1  # MVP: single seed. Multi-seed flow in a later iteration.
PROMPT_METHOD = "manual_click"


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


def extract_cutout(image: np.ndarray, x: float, y: float, size: int = CUTOUT_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    half = size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)
    return image[y1:y2, x1:x2].copy()


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
    """Next video without a (seed=SEED, prompt_method=manual_click) prompt_set."""
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
            pts = obj["points_by_frame"].get(frame_pos)
            if not pts or not pts["positive"]:
                continue
            per_frame.append({
                "obj_id": int(obj_id),
                "positive": [[float(x), float(y)] for x, y in pts["positive"]],
                "negative": [[float(x), float(y)] for x, y in pts.get("negative", [])],
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
        "source_frame_indices": [],  # actual frame numbers in `frame_paths` we're clicking
        "cur_frame_pos": 0,           # 0..N_PROMPT_FRAMES-1
        "objects": {},                # obj_id -> {instrument_id, points_by_frame, cutout, label}
        "current_obj_id": 1,
        "current_instrument_id": None,
        "current_instrument_label": "",
    }


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


def render_current_frame_with_points(state: dict) -> np.ndarray:
    """Draw existing points for the current frame on top of the raw image."""
    if not state["frame_paths"]:
        return np.zeros((480, 720, 3), dtype=np.uint8)
    idx = state["source_frame_indices"][state["cur_frame_pos"]]
    img = safe_load_image(state["frame_paths"][idx])
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for obj_id, obj in state["objects"].items():
        pts = obj["points_by_frame"].get(state["cur_frame_pos"], {"positive": [], "negative": []})
        for (x, y) in pts["positive"]:
            r = POINT_RADIUS
            draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], outline=POINT_COLOR_OBJ_BORDER, width=2)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=POINT_COLOR_POS)
            draw.text((x + r + 4, y - r), f"{obj_id}", fill=POINT_COLOR_OBJ_BORDER)
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
    cur_obj = state["current_obj_id"]
    cur_ins = state["current_instrument_label"] or "(pick from dropdown)"
    n_objs = len(state["objects"])
    remaining_str = f"  |  {remaining} videos remaining" if remaining is not None else ""
    return (
        f"**{vid}**  |  frame {pos+1}/{total}  (source idx {src_idx} of {n_frames})  |  "
        f"obj_id={cur_obj}, instrument={cur_ins}  |  {n_objs} objects clicked total{remaining_str}"
    )


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def handler_start(state: dict):
    state = start_new_video(state)
    img = render_current_frame_with_points(state) if state["video_id"] else None
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), img, build_cutout_gallery(state)


def handler_pick_instrument(state: dict, instrument_label_value):
    """Dropdown returns the display label by default; map back to instrument_id."""
    # gr.Dropdown with `choices=[(label, value), ...]` returns the value
    if not instrument_label_value:
        state["current_instrument_id"] = None
        state["current_instrument_label"] = ""
    else:
        state["current_instrument_id"] = instrument_label_value
        # Find pretty label
        for label, value in load_instrument_choices():
            if value == instrument_label_value:
                state["current_instrument_label"] = label
                break
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem)


def handler_image_click(state: dict, evt: gr.SelectData):
    if not state.get("video_id"):
        return state, "Start a session first.", None, []
    if not state.get("current_instrument_id"):
        return state, "Pick an instrument from the dropdown before clicking.", render_current_frame_with_points(state), build_cutout_gallery(state)

    x, y = float(evt.index[0]), float(evt.index[1])
    obj_id = state["current_obj_id"]
    frame_pos = state["cur_frame_pos"]

    if obj_id not in state["objects"]:
        state["objects"][obj_id] = {
            "instrument_id": state["current_instrument_id"],
            "label": state["current_instrument_label"],
            "points_by_frame": {},
            "cutout": None,
        }

    pts = state["objects"][obj_id]["points_by_frame"].setdefault(
        frame_pos, {"positive": [], "negative": []}
    )
    pts["positive"].append([x, y])

    if state["objects"][obj_id]["cutout"] is None:
        idx = state["source_frame_indices"][frame_pos]
        cur_img = safe_load_image(state["frame_paths"][idx])
        state["objects"][obj_id]["cutout"] = extract_cutout(cur_img, x, y)

    rem = count_remaining(connect())
    return (
        state,
        status_text(state, remaining=rem),
        render_current_frame_with_points(state),
        build_cutout_gallery(state),
    )


def handler_next_object(state: dict):
    """Commit current obj_id (if any points), advance to obj_id+1."""
    if not state.get("video_id"):
        return state, "Start a session first."
    existing_ids = list(state["objects"].keys())
    state["current_obj_id"] = (max(existing_ids) + 1) if existing_ids else 1
    state["current_instrument_id"] = None
    state["current_instrument_label"] = ""
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem)


def handler_undo_last_point(state: dict):
    """Pop the most recent positive point from the current obj on current frame."""
    obj_id = state["current_obj_id"]
    frame_pos = state["cur_frame_pos"]
    obj = state["objects"].get(obj_id)
    if obj:
        pts = obj["points_by_frame"].get(frame_pos)
        if pts and pts["positive"]:
            pts["positive"].pop()
            if not pts["positive"] and not pts.get("negative"):
                obj["points_by_frame"].pop(frame_pos, None)
            if not obj["points_by_frame"]:
                state["objects"].pop(obj_id, None)
    rem = count_remaining(connect())
    return (
        state,
        status_text(state, remaining=rem),
        render_current_frame_with_points(state),
        build_cutout_gallery(state),
    )


def handler_next_frame(state: dict):
    if not state.get("video_id"):
        return state, "Start a session first.", None, []
    state["cur_frame_pos"] = min(state["cur_frame_pos"] + 1, len(state["source_frame_indices"]) - 1)
    state["current_obj_id"] = max(list(state["objects"].keys()) + [0]) + 1
    state["current_instrument_id"] = None
    state["current_instrument_label"] = ""
    rem = count_remaining(connect())
    return (
        state,
        status_text(state, remaining=rem),
        render_current_frame_with_points(state),
        build_cutout_gallery(state),
    )


def handler_resample_frame(state: dict):
    """Replace the source index for the current frame slot with a new random one."""
    if not state.get("video_id"):
        return state, "Start a session first.", None, []
    n = len(state["frame_paths"])
    new_idx = random.randint(0, n - 1)
    state["source_frame_indices"][state["cur_frame_pos"]] = new_idx
    # Drop points on the now-discarded frame slot for all objects
    for obj in state["objects"].values():
        obj["points_by_frame"].pop(state["cur_frame_pos"], None)
    rem = count_remaining(connect())
    return (
        state,
        status_text(state, remaining=rem),
        render_current_frame_with_points(state),
        build_cutout_gallery(state),
    )


def handler_save_and_next(state: dict):
    if not state.get("video_id"):
        return state, "Nothing to save.", None, []
    if not state["objects"]:
        return state, "No objects clicked — use Skip Video if there's nothing to label.", render_current_frame_with_points(state), []
    try:
        path = save_prompt_set_to_manifest(state)
    except Exception as e:
        return state, f"Save failed: {e}", render_current_frame_with_points(state), build_cutout_gallery(state)
    msg = f"Saved {path}. Loading next video..."
    state = start_new_video(state)
    img = render_current_frame_with_points(state) if state["video_id"] else None
    rem = count_remaining(connect())
    return state, msg + "\n\n" + status_text(state, remaining=rem), img, build_cutout_gallery(state)


def handler_skip_video(state: dict):
    if not state.get("video_id"):
        return state, "Nothing to skip.", None, []
    skipped_id = state["video_id"]
    mark_video_skipped(skipped_id)
    state = start_new_video(state)
    img = render_current_frame_with_points(state) if state["video_id"] else None
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
            "## SurgSAM-2 click collector\n"
            "1. **Start session** to load the next un-clicked video.  "
            "2. Pick an instrument.  "
            "3. Click on it in the image (multiple clicks refine the same object).  "
            "4. **Next object** when moving to a different instrument.  "
            "5. **Next frame** after labelling everything on this frame.  "
            "6. **Save & next video** after all 3 frames."
        )
        status_md = gr.Markdown("Click 'Start session' to begin.")

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Click on instruments", interactive=True, height=540)
                with gr.Row():
                    instrument_dd = gr.Dropdown(
                        choices=instrument_choices,
                        label="Current object's instrument",
                        value=None,
                        interactive=True,
                    )
                with gr.Row():
                    btn_next_obj = gr.Button("Next object (start clicking a different instrument)")
                    btn_undo = gr.Button("Undo last click")
                with gr.Row():
                    btn_resample = gr.Button("This frame is empty -> sample a different one")
                    btn_next_frame = gr.Button("Next frame", variant="primary")
                with gr.Row():
                    btn_save = gr.Button("Save & next video", variant="primary")
                    btn_skip = gr.Button("Skip this video (nothing labelable)")
                btn_start = gr.Button("Start session", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Reference cutouts")
                cutout_gallery = gr.Gallery(label="Objects clicked", columns=1, height=540, allow_preview=False)

        state = gr.State(empty_state())

        # Wire handlers
        btn_start.click(handler_start, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        instrument_dd.change(handler_pick_instrument, inputs=[state, instrument_dd], outputs=[state, status_md])
        image.select(handler_image_click, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
        btn_next_obj.click(handler_next_object, inputs=[state], outputs=[state, status_md])
        btn_undo.click(handler_undo_last_point, inputs=[state], outputs=[state, status_md, image, cutout_gallery])
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
