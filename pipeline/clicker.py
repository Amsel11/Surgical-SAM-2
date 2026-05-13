"""Gradio drag-to-box collector for SurgSAM-2 prompts (v2 — drag mode).

Replaces the earlier 2-click corner version. Uses the third-party
`gradio_image_annotation` component for native drag-rectangle drawing.
Default label for every box is `unknown_instrument` so the user can move
through quickly without being blocked on the dropdown — they can refine
labels later via a DB update or future per-obj UI.

v2 simplifications vs the previous clicker:
- One prompt frame per video (middle of the video) instead of 3.
  Multi-frame re-prompting protocol will return in a later iteration; for now
  the focus is "drag + drop a box around each instrument, save, move on."
- No state machine for in-progress clicks — the annotator owns box state.
- No reboxing flow — irrelevant with a single prompt frame.

Same backend: writes prompts JSON + prompt_set + prompt_objects rows. JSON
shape unchanged (still `box: [x1,y1,x2,y2]` per obj), so run_on_video.py
continues to work without changes.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageFile

from .db import REPO_ROOT, connect, transaction

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROMPTS_DIR = REPO_ROOT / "prompts"
SEED = 1
PROMPT_METHOD = "manual_box"
DEFAULT_INSTRUMENT = "unknown_instrument"

# Per-object box colors; cycled by index. Used as label_colors for the annotator.
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def safe_load_image(path: str | Path) -> np.ndarray:
    """Read fully into memory then decode. Dodges PIL streaming truncation on GPFS."""
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


def normalize_box(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_instrument_ids() -> list[str]:
    """All instrument_ids, with unknown_instrument forced to the front so it's
       the default label for new boxes in the annotator."""
    conn = connect()
    rows = conn.execute(
        "SELECT instrument_id FROM instruments ORDER BY category, instrument_id"
    ).fetchall()
    ids = [r["instrument_id"] for r in rows]
    if DEFAULT_INSTRUMENT in ids:
        ids = [DEFAULT_INSTRUMENT] + [i for i in ids if i != DEFAULT_INSTRUMENT]
    else:
        ids = [DEFAULT_INSTRUMENT] + ids
    return ids


def fetch_next_pending_video(conn) -> dict | None:
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


def save_prompt_set(state: dict, boxes: list[dict]) -> str:
    """Write prompts JSON + insert prompt_set + prompt_objects in one transaction.

    `boxes`: list of dicts from the annotator, each shaped
        {"xmin": float, "ymin": float, "xmax": float, "ymax": float, "label": str, ...}
    """
    video_id = state["video_id"]
    n_frames_total = len(state["frame_paths"])
    prompt_frame_src = state["prompt_frame_src_idx"]

    first_image = safe_load_image(state["frame_paths"][0])
    h, w = first_image.shape[:2]

    objs_for_frame: list[dict] = []
    for i, b in enumerate(boxes):
        obj_id = i + 1
        box = normalize_box(b["xmin"], b["ymin"], b["xmax"], b["ymax"])
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue  # ignore degenerate
        objs_for_frame.append({
            "obj_id": obj_id,
            "box": box,
            "positive": [],
            "negative": [],
            "_instrument_id": b.get("label") or DEFAULT_INSTRUMENT,
        })

    if not objs_for_frame:
        raise ValueError("No usable boxes (all empty / degenerate).")

    payload = {
        "video": video_id,
        "resolution": [w, h],
        "n_frames": n_frames_total,
        "prompt_frames": [prompt_frame_src],
        "objects_by_frame": {
            str(prompt_frame_src): [
                {k: v for k, v in o.items() if not k.startswith("_")}
                for o in objs_for_frame
            ]
        },
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
            VALUES (?, ?, ?, ?, ?, 1, 'ready', 'gradio-clicker-drag')
            """,
            (video_id, SEED, PROMPT_METHOD, str(out_path), len(objs_for_frame)),
        )
        ps_id = cur.lastrowid
        for o in objs_for_frame:
            conn.execute(
                "INSERT INTO prompt_objects (prompt_set_id, obj_id, instrument_id) VALUES (?, ?, ?)",
                (ps_id, o["obj_id"], o["_instrument_id"]),
            )
    return str(out_path)


def mark_video_skipped(video_id: str) -> None:
    conn = connect()
    conn.execute(
        """
        INSERT INTO prompt_sets (video_id, seed, prompt_method, status, created_by, notes)
        VALUES (?, ?, ?, 'failed', 'gradio-clicker-drag', 'skipped: no usable instruments')
        ON CONFLICT(video_id, seed, prompt_method) DO UPDATE SET status='failed'
        """,
        (video_id, SEED, PROMPT_METHOD),
    )


# ---------------------------------------------------------------------------
# State + UI handlers
# ---------------------------------------------------------------------------

def empty_state() -> dict:
    return {
        "video_id": None,
        "frames_dir": None,
        "frame_paths": [],
        "prompt_frame_src_idx": 0,
    }


def load_video_into_state(state: dict) -> tuple[dict, dict | None]:
    """Load next pending video. Returns (state, annotator_value_or_None_if_done)."""
    conn = connect()
    row = fetch_next_pending_video(conn)
    if row is None:
        return empty_state(), None
    frames = list_frame_paths(row["frames_dir"])
    if not frames:
        mark_video_skipped(row["video_id"])
        return load_video_into_state(state)

    middle = len(frames) // 2
    state = empty_state()
    state["video_id"] = row["video_id"]
    state["frames_dir"] = row["frames_dir"]
    state["frame_paths"] = frames
    state["prompt_frame_src_idx"] = middle

    img = safe_load_image(frames[middle])
    annot_val = {"image": img, "boxes": []}
    return state, annot_val


def status_text(state: dict, remaining: int | None = None, msg: str = "") -> str:
    head = msg + ("\n\n" if msg else "")
    if not state.get("video_id"):
        return head + "All videos done." if remaining == 0 else head + "Click 'Start session' to begin."
    vid = state["video_id"]
    src_idx = state["prompt_frame_src_idx"]
    n = len(state["frame_paths"])
    remaining_str = f"  |  {remaining} videos remaining" if remaining is not None else ""
    return (
        head
        + f"**{vid}**  |  prompt frame: source idx {src_idx} of {n}{remaining_str}\n\n"
        + "Drag a rectangle around each instrument. Default label = `unknown_instrument` "
        + "(right-click a box or use the picker to change). Save & next when done."
    )


def handler_start(state: dict):
    state, annot_val = load_video_into_state(state)
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem), annot_val


def handler_save_and_next(state: dict, annotator_value):
    if not state.get("video_id"):
        return state, "Nothing to save.", annotator_value
    boxes = (annotator_value or {}).get("boxes", []) or []
    try:
        path = save_prompt_set(state, boxes)
        msg = f"Saved {Path(path).name}. Loading next..."
    except ValueError as e:
        rem = count_remaining(connect())
        return state, str(e), annotator_value
    except Exception as e:
        rem = count_remaining(connect())
        return state, f"Save failed: {e}", annotator_value

    state, new_annot = load_video_into_state(state)
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem, msg=msg), new_annot


def handler_skip_video(state: dict):
    if not state.get("video_id"):
        return state, "Nothing to skip.", None
    skipped = state["video_id"]
    mark_video_skipped(skipped)
    state, new_annot = load_video_into_state(state)
    rem = count_remaining(connect())
    return state, status_text(state, remaining=rem, msg=f"Skipped {skipped}."), new_annot


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    from gradio_image_annotation import image_annotator  # imported here so module loads even if pkg missing

    instrument_ids = load_instrument_ids()
    # Pad colors out to match instrument count by cycling.
    label_colors = [OBJ_COLORS[i % len(OBJ_COLORS)] for i in range(len(instrument_ids))]

    with gr.Blocks(title="SurgSAM-2 Drag-to-Box Collector") as demo:
        gr.Markdown(
            "## SurgSAM-2 box collector — drag mode\n"
            "1. **Start session** -> next un-labelled video loads at its middle frame.\n"
            "2. **Drag a rectangle** around each instrument. New boxes default to `unknown_instrument`.\n"
            "3. (Optional) Right-click a box (or use the in-component picker) to relabel.\n"
            "4. **Save & next video** to commit and load the next.\n\n"
            "_v1 simplification_: 1 prompt frame per video. Multi-frame re-prompting will return in v2."
        )
        status_md = gr.Markdown("Click 'Start session' to begin.")

        annotator = image_annotator(
            value={"image": None, "boxes": []},
            label_list=instrument_ids,
            label_colors=label_colors,
            height=620,
            show_label=False,
            disable_edit_boxes=False,
            single_box=False,
        )

        with gr.Row():
            btn_start = gr.Button("Start session", variant="primary")
            btn_save = gr.Button("Save & next video", variant="primary")
            btn_skip = gr.Button("Skip this video (nothing labelable)")

        state = gr.State(empty_state())

        btn_start.click(handler_start, inputs=[state], outputs=[state, status_md, annotator])
        btn_save.click(handler_save_and_next, inputs=[state, annotator], outputs=[state, status_md, annotator])
        btn_skip.click(handler_skip_video, inputs=[state], outputs=[state, status_md, annotator])

    return demo


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Gradio drag-to-box collector")
    p.add_argument("--port", type=int, default=9876)
    p.add_argument("--host", default="0.0.0.0")
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
