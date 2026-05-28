"""Post-hoc outline overlay: read palette mask PNGs and source frames,
draw a coloured outline (no fill) per obj_id, encode mp4.

Why outline-only: solid masks hide drift and mask-vs-mask overlap. With
outlines you can see when a mask blows up, when two masks engulf the same
region, and when a track has drifted off its instrument.

Inputs:
  --masks-dir      directory of {NNNNN}.png palette masks (pixel = obj_id)
  --frames-dir     directory of frame_<src>.png source frames
  --out-mp4        output mp4 path
  --fps            playback fps (default 10 = ~4:20 for 2590 frames)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.io import davis_color_bgr  # noqa: E402


def outline_of(mask: np.ndarray) -> np.ndarray:
    """Boolean outline = mask - erode(mask) thickened by dilation."""
    m8 = mask.astype(np.uint8)
    eroded = cv2.erode(m8, np.ones((3, 3), np.uint8), iterations=1)
    edge = (m8 - eroded).astype(bool)
    # thicken so it's visible at video resolution
    edge8 = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1)
    return edge8.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--masks-dir",  required=True, type=Path)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--out-mp4",    required=True, type=Path)
    ap.add_argument("--fps",        type=float, default=10.0)
    args = ap.parse_args()

    mask_files = sorted(args.masks_dir.glob("*.png"))
    if not mask_files:
        print(f"FATAL: no mask PNGs in {args.masks_dir}", file=sys.stderr); return 2
    frame_files = sorted(args.frames_dir.glob("frame_*.png"))
    if not frame_files:
        print(f"FATAL: no frames in {args.frames_dir}", file=sys.stderr); return 2
    if len(mask_files) != len(frame_files):
        print(f"WARNING: {len(mask_files)} masks vs {len(frame_files)} frames — "
              f"will iterate over min({len(mask_files)}, {len(frame_files)})")

    n = min(len(mask_files), len(frame_files))
    print(f"Rendering {n} frames @ {args.fps} fps")

    tmpdir = args.out_mp4.parent / f".{args.out_mp4.stem}_jpgs"
    tmpdir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        # masks are saved 1:1 with loader index → frame_files sorted matches
        img = cv2.imread(str(frame_files[i]))
        if img is None:
            continue
        # PIL preserves palette indices (= obj_ids); cv2 decodes the palette
        # into 3-channel RGB which loses the obj_id information.
        try:
            m_pil = np.array(Image.open(mask_files[i]))
        except Exception:
            cv2.imwrite(str(tmpdir / f"{i:05d}.jpg"), img); continue
        out = img.copy()
        for oid in np.unique(m_pil):
            if oid == 0: continue
            m = (m_pil == oid)
            if not m.any(): continue
            edge = outline_of(m)
            color = davis_color_bgr(int(oid))
            # Explicit per-channel write avoids broadcast ambiguity with grayscale frames.
            for c in range(3):
                out[..., c][edge] = color[c]
            # label at centroid of the mask
            ys, xs = np.where(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(out, f"{int(oid)}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(tmpdir / f"{i:05d}.jpg"), out)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{n}")

    # Encode via imageio-ffmpeg, libx264.
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    jpgs = sorted(tmpdir.glob("*.jpg"))
    concat = tmpdir / "_concat.txt"
    with open(concat, "w") as f:
        for j in jpgs:
            f.write(f"file '{j.name}'\nduration {1.0/args.fps}\n")
        f.write(f"file '{jpgs[-1].name}'\n")
    r = subprocess.run([
        ffmpeg_bin, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vsync", "vfr",
        "-movflags", "+faststart", str(args.out_mp4),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FFMPEG FAILED rc={r.returncode}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        return 3
    # cleanup
    for j in jpgs: j.unlink()
    concat.unlink(missing_ok=True)
    try: tmpdir.rmdir()
    except OSError: pass
    print(f"Wrote {args.out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
