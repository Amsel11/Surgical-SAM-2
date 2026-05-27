"""IO helpers shared between the orchestrator and the CLI shim.

Frame discovery + symlink renaming, mask palette, overlay compositing,
mp4 assembly. Kept dependency-light (PIL, numpy, cv2) so it can be
imported by both model implementations and CLI scripts.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

# PIL's lazy decoder hits spurious "image file is truncated" errors on GPFS
# even when the PNG is fully written. Tell it to be tolerant globally.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
FRAME_RE = re.compile(r"^frame_(\d+)$")

# Shared anchor-frame sampling + default prompts dir. Lives here rather than in
# pipeline.clicker so the dino prompter (which runs in a non-gradio env) can
# reuse them without pulling gradio in via the clicker module import.
N_PROMPT_FRAMES = 3
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def sample_frame_positions(n_total: int, n_samples: int = N_PROMPT_FRAMES) -> list[int]:
    """Uniform stratified sampling: 25/50/75% by default for n_samples=3."""
    return [int(round((i + 1) * n_total / (n_samples + 1))) for i in range(n_samples)]

# DAVIS palette so palette-PNG masks render with consistent colors.
DAVIS_PALETTE = (
    b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80"
    b"\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00"
    b"\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80"
    b"\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0"
    b"\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@"
)


def davis_color_bgr(obj_id: int) -> tuple[int, int, int]:
    """RGB triplet from DAVIS_PALETTE at index obj_id, returned as BGR for OpenCV."""
    i = (obj_id * 3) % len(DAVIS_PALETTE)
    r, g, b = DAVIS_PALETTE[i], DAVIS_PALETTE[i + 1], DAVIS_PALETTE[i + 2]
    return (int(b), int(g), int(r))


def list_frames(frames_dir: str | Path) -> list[str]:
    """Return frame filenames in frames_dir, sorted by integer stem."""
    frames_dir = str(frames_dir)
    names = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[1] in IMG_EXTS]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return names


def prepare_loader_dir(frames_dir: str | Path) -> tuple[str, int, callable]:
    """Return (loader_dir, source_offset, cleanup_fn).

    SAM2's video loader sorts frames by ``int(splitext(name)[0])``, so file
    stems must be plain integers. Whip frames are named ``frame_NNNNNNNNNN.png``,
    which crashes the loader. When we detect the ``frame_<digits>`` pattern
    we build a temp dir of symlinks named ``<i>.<ext>`` (re-indexed from 0,
    sorted by the captured integer) and use that as the loader dir.

    ``source_offset`` is the original integer of the first kept frame, so
    callers can map output index ``i`` back to source frame ``offset + i``
    (and map clicker-emitted source indices into loader space by subtracting).
    """
    frames_dir = str(frames_dir)
    entries = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[1] in IMG_EXTS]
    if not entries:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    # Already integer-named — pass through.
    try:
        sorted(entries, key=lambda p: int(os.path.splitext(p)[0]))
        return frames_dir, 0, lambda: None
    except ValueError:
        pass

    # Try ``frame_<digits>`` pattern.
    parsed = []
    for p in entries:
        stem, ext = os.path.splitext(p)
        m = FRAME_RE.match(stem)
        if not m:
            raise RuntimeError(
                f"Frames in {frames_dir} use an unsupported naming scheme; "
                f"expected '<int>.ext' or 'frame_<digits>.ext', got {p!r}"
            )
        parsed.append((int(m.group(1)), ext, p))
    parsed.sort()

    tmp = tempfile.mkdtemp(prefix="surgsam2_renamed_")
    frames_dir_abs = os.path.abspath(frames_dir)
    for new_idx, (_, ext, original) in enumerate(parsed):
        os.symlink(os.path.join(frames_dir_abs, original), os.path.join(tmp, f"{new_idx}{ext}"))
    offset = parsed[0][0]
    print(f"Renamed-symlink dir: {tmp} (source offset = {offset}, {len(parsed)} frames)")

    def _cleanup():
        shutil.rmtree(tmp, ignore_errors=True)

    return tmp, offset, _cleanup


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color, alpha: float = 0.5) -> np.ndarray:
    """Blend a colored mask onto a BGR image at the given alpha."""
    out = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[:] = color
    m = mask.astype(bool)
    out[m] = (alpha * color_layer[m] + (1 - alpha) * image_bgr[m]).astype(np.uint8)
    return out


def save_palette_mask(combined: np.ndarray, path: Path) -> None:
    """Save a per-pixel obj_id map as an 8-bit palette PNG with DAVIS colors."""
    mask_img = Image.fromarray(combined, mode="P")
    mask_img.putpalette(DAVIS_PALETTE)
    mask_img.save(path)


def encode_mp4_from_jpgs(jpg_dir: Path, output_mp4: Path, src_fps: float) -> bool:
    """Assemble a directory of overlay JPGs into an H.264 mp4 via ffmpeg's
    concat demuxer. Returns True iff at least one JPG was written.

    H.264 via the bundled imageio-ffmpeg binary so the resulting mp4 plays
    inline in browsers + Jupyter; OpenCV's mp4v fourcc produces MPEG-4 Part 2
    which Firefox/Chrome refuse to play.
    """
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    jpgs = sorted(jpg_dir.glob("*.jpg"))
    if not jpgs:
        return False

    # ffmpeg's concat demuxer resolves `file 'X.jpg'` paths *relative to the
    # concat file's own directory*. Put the concat file IN jpg_dir so the bare
    # basenames in it resolve correctly. (Previously we wrote it to
    # jpg_dir.parent, which made ffmpeg look in the parent dir and fail.)
    concat_path = jpg_dir / "_concat.txt"
    with open(concat_path, "w") as f:
        for j in jpgs:
            f.write(f"file '{j.name}'\n")
            f.write(f"duration {1.0 / src_fps}\n")
        # ffmpeg concat needs the last file repeated for the final duration to apply.
        f.write(f"file '{jpgs[-1].name}'\n")

    subprocess.run(
        [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vsync", "vfr",
            "-movflags", "+faststart", str(output_mp4),
        ],
        cwd=jpg_dir, check=False,
    )
    concat_path.unlink(missing_ok=True)
    return True
