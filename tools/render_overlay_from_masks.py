"""Regenerate overlay.mp4 + preview_small.mp4 from existing masks/ + frames.

Use case: a previous inference run wrote masks/ + log.json but the overlay.mp4
encoding step failed (e.g. the ffmpeg concat-path bug that hit seed_2 of the
2026-05-14 array). The model output is fine; we just lost the visualization
step. This tool re-composites overlays from the persisted palette PNGs and
re-encodes — no GPU needed, no model inference.

Per video, reads:
  <results_dir>/log.json        for source_offset
  <results_dir>/masks/*.png     palette PNGs (pixel value = obj_id)
  <frames_root>/<video>/*.png   source frames

Writes:
  <results_dir>/overlay.mp4
  <results_dir>/preview_small.mp4   (320p CRF32, optional)

Usage (per-video):
  python tools/render_overlay_from_masks.py \
      --results-dir results/DC_whip_11609423/seed_2 \
      --frames-root /gpfs/.../whip/frames_attempt2

Usage (batch — all videos with masks/ but no overlay.mp4 under a tree):
  python tools/render_overlay_from_masks.py \
      --batch results/ \
      --seed 2 \
      --frames-root /gpfs/.../whip/frames_attempt2
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# DAVIS palette — same as in pipeline/io.py / run_on_video.py.
DAVIS_PALETTE = (
    b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80"
    b"\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00"
    b"\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80"
    b"\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0"
    b"\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@"
)


def davis_color_bgr(obj_id: int) -> tuple[int, int, int]:
    i = (obj_id * 3) % len(DAVIS_PALETTE)
    r, g, b = DAVIS_PALETTE[i], DAVIS_PALETTE[i + 1], DAVIS_PALETTE[i + 2]
    return (int(b), int(g), int(r))


def overlay_mask(image_bgr, mask, color, alpha: float = 0.5):
    out = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[:] = color
    m = mask.astype(bool)
    out[m] = (alpha * color_layer[m] + (1 - alpha) * image_bgr[m]).astype(np.uint8)
    return out


def render_one(results_dir: Path, frames_root: Path, src_fps: float = 1.0,
               make_preview: bool = True, force: bool = False) -> bool:
    """Render overlay.mp4 (and preview_small.mp4) for one results dir.
    Returns True if rendering succeeded, False on skip/no-op."""
    log_path = results_dir / "log.json"
    masks_dir = results_dir / "masks"
    if not log_path.exists() or not masks_dir.is_dir():
        print(f"[skip] {results_dir}: missing log.json or masks/")
        return False

    # Use absolute path because ffmpeg runs with cwd=tmpdir below.
    out_mp4 = (results_dir / "overlay.mp4").resolve()
    if out_mp4.exists() and not force:
        print(f"[skip] {results_dir}: overlay.mp4 exists (use --force to overwrite)")
        return False

    log = json.loads(log_path.read_text())
    video_id = log["video"]
    frames_dir = frames_root / video_id
    if not frames_dir.is_dir():
        print(f"[skip] {video_id}: frames dir not found: {frames_dir}")
        return False

    # bp frames are sorted by integer in filename ("frame_NNNNNNNNNN.png").
    # Loader index N corresponds to the Nth sorted source frame.
    frame_files = sorted(frames_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[1]))
    if not frame_files:
        # Some videos might use .jpg
        frame_files = sorted(frames_dir.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    if not frame_files:
        print(f"[skip] {video_id}: no source frames found")
        return False

    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        print(f"[skip] {video_id}: no mask files in {masks_dir}")
        return False

    print(f"[{video_id}] {len(mask_files)} masks, {len(frame_files)} source frames")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"overlay_render_{video_id}_"))
    try:
        n_written = 0
        for mask_path in mask_files:
            n = int(mask_path.stem)
            if n >= len(frame_files):
                continue
            img_bgr = cv2.imread(str(frame_files[n]))
            if img_bgr is None:
                continue
            mask_palette = np.array(Image.open(mask_path))
            overlay = img_bgr.copy()
            for oid in np.unique(mask_palette):
                if oid == 0:
                    continue
                m = (mask_palette == oid).astype(bool)
                overlay = overlay_mask(overlay, m, davis_color_bgr(int(oid)))
            cv2.imwrite(str(tmpdir / f"{n:05d}.jpg"), overlay,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            n_written += 1
            if n_written % 500 == 0:
                print(f"  composited {n_written}/{len(mask_files)} frames")

        if n_written == 0:
            print(f"[skip] {video_id}: no frames composited")
            return False

        # Encode with ffmpeg concat. Concat file goes INSIDE tmpdir so its
        # `file 'X.jpg'` references resolve correctly.
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        jpgs = sorted(tmpdir.glob("*.jpg"))
        concat_path = tmpdir / "_concat.txt"
        with open(concat_path, "w") as f:
            for j in jpgs:
                f.write(f"file '{j.name}'\n")
                f.write(f"duration {1.0 / src_fps}\n")
            f.write(f"file '{jpgs[-1].name}'\n")

        subprocess.run(
            [ffmpeg_bin, "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(concat_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vsync", "vfr",
             "-movflags", "+faststart", str(out_mp4)],
            cwd=tmpdir, check=True,
        )
        print(f"[{video_id}] wrote {out_mp4}")

        if make_preview:
            preview_mp4 = (results_dir / "preview_small.mp4").resolve()
            subprocess.run(
                [ffmpeg_bin, "-y", "-loglevel", "error",
                 "-i", str(out_mp4),
                 "-vf", "scale=320:-2",
                 "-c:v", "libx264", "-crf", "32", "-preset", "veryfast",
                 "-an", str(preview_mp4)],
                check=True,
            )
            print(f"[{video_id}] wrote {preview_mp4}")
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path,
                    help="Single result dir (with masks/ + log.json) to render.")
    ap.add_argument("--batch", type=Path,
                    help="Batch mode: a results/ root. Renders every "
                         "<video>/seed_<N>/ that has masks but no overlay.mp4.")
    ap.add_argument("--seed", type=int, default=1, help="Seed (batch mode only).")
    ap.add_argument("--frames-root", type=Path, required=True,
                    help="Root of per-video frame dirs, e.g. .../whip/frames_attempt2")
    ap.add_argument("--src-fps", type=float, default=1.0)
    ap.add_argument("--no-preview", action="store_true",
                    help="Skip generating preview_small.mp4.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing overlay.mp4.")
    args = ap.parse_args(argv)

    if not args.results_dir and not args.batch:
        ap.error("Specify either --results-dir or --batch.")
    if args.results_dir and args.batch:
        ap.error("Pass --results-dir or --batch, not both.")

    if args.results_dir:
        ok = render_one(args.results_dir, args.frames_root, args.src_fps,
                        not args.no_preview, args.force)
        return 0 if ok else 1

    # Batch
    seed_dir_name = f"seed_{args.seed}"
    candidates = sorted(args.batch.glob(f"*/{seed_dir_name}"))
    if not candidates:
        print(f"No {seed_dir_name}/ dirs found under {args.batch}")
        return 1
    n_done = 0
    n_skip = 0
    for d in candidates:
        if render_one(d, args.frames_root, args.src_fps,
                      not args.no_preview, args.force):
            n_done += 1
        else:
            n_skip += 1
    print(f"\nBatch: {n_done} rendered, {n_skip} skipped (already had mp4, missing inputs, etc.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
