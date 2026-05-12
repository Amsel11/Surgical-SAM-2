"""Run SurgSAM-2 / SAM 2.1 on your own surgical video.

Examples:

    # mp4 input, one positive click on frame 0
    python run_on_video.py \
        --video /path/to/clip.mp4 \
        --point 640,360 \
        --output-dir ./results/clip

    # frames-folder input, two positive clicks + one negative click
    python run_on_video.py \
        --video /path/to/frames_dir \
        --point 1100,350 --point 980,420 \
        --neg-point 200,200 \
        --output-dir ./results/clip

    # box prompt instead of clicks
    python run_on_video.py \
        --video /path/to/clip.mp4 \
        --box 580,300,720,440 \
        --output-dir ./results/clip
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
FRAME_RE = re.compile(r"^frame_(\d+)$")

from sam2.build_sam import build_sam2_video_predictor

# DAVIS palette so saved PNG masks are viewable with consistent colors.
DAVIS_PALETTE = (
    b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80"
    b"\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00"
    b"\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80"
    b"\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0"
    b"\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@"
)


def parse_point(s):
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"point must be 'x,y', got {s!r}")
    return [float(parts[0]), float(parts[1])]


def parse_box(s):
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"box must be 'x1,y1,x2,y2', got {s!r}")
    return [float(p) for p in parts]


def parse_object(s):
    """Parse '--object OBJID:x,y[,x,y...]'. Points prefixed with '!' are negative."""
    if ':' not in s:
        raise argparse.ArgumentTypeError(f"--object must be 'OBJID:x,y[,x,y...]', got {s!r}")
    oid_str, rest = s.split(':', 1)
    try:
        oid = int(oid_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--object OBJID must be an int, got {oid_str!r}")
    if oid <= 0:
        raise argparse.ArgumentTypeError(f"--object OBJID must be > 0, got {oid}")
    toks = [t.strip() for t in rest.split(',')]
    if len(toks) % 2 != 0:
        raise argparse.ArgumentTypeError(f"--object {oid}: coords must come in x,y pairs, got {rest!r}")
    pts, labels = [], []
    for i in range(0, len(toks), 2):
        x_tok = toks[i]
        neg = x_tok.startswith('!')
        if neg: x_tok = x_tok[1:]
        pts.append([float(x_tok), float(toks[i+1])])
        labels.append(0 if neg else 1)
    return {'obj_id': oid, 'points': pts, 'labels': labels}


def davis_color_bgr(obj_id: int):
    """RGB triplet from DAVIS_PALETTE at index obj_id, returned as BGR for OpenCV overlay."""
    i = (obj_id * 3) % len(DAVIS_PALETTE)
    r, g, b = DAVIS_PALETTE[i], DAVIS_PALETTE[i+1], DAVIS_PALETTE[i+2]
    return (int(b), int(g), int(r))


def extract_frames_to_dir(video_path: str, out_dir: Path) -> float:
    """Decode video to JPEG frames 00000.jpg, 00001.jpg, ... using opencv. Returns source fps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1
    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return float(fps)


def detect_fps_for_dir(frames_dir: str) -> float:
    return 30.0  # No native fps when input is a frames dir; user can override via --fps.


def list_frames(frames_dir: str):
    names = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[1] in IMG_EXTS]
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    return names


def prepare_loader_dir(frames_dir: str):
    """Return (loader_dir, source_index_offset, cleanup_fn).

    SAM2's video loader sorts frames by ``int(splitext(name)[0])``, so basenames
    must be plain integers. Whip frames are named ``frame_NNNNNNNNNN.png`` —
    that crashes the loader. When we detect the ``frame_<digits>`` pattern we
    build a temp directory of symlinks named ``<i>.<ext>`` (re-indexed from 0,
    sorted by the captured integer) and use that as the loader dir.

    ``source_index_offset`` is the original integer of the first kept frame,
    so callers can map output index ``i`` back to source frame ``offset + i``.
    """
    entries = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[1] in IMG_EXTS]
    if not entries:
        raise RuntimeError(f"No image frames found in {frames_dir}")

    # Already integer-named — pass through.
    try:
        entries_sorted = sorted(entries, key=lambda p: int(os.path.splitext(p)[0]))
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


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color, alpha=0.5):
    out = image_bgr.copy()
    color_layer = np.zeros_like(image_bgr)
    color_layer[:] = color
    m = mask.astype(bool)
    out[m] = (alpha * color_layer[m] + (1 - alpha) * image_bgr[m]).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to .mp4 OR a directory of frames named 00000.jpg, 00001.jpg, ...")
    ap.add_argument("--output-dir", required=True, help="Where to write masks and overlay video")
    ap.add_argument("--checkpoint", default="./checkpoints/sam2.1_hiera_s_endo18.pth",
                    help="Path to model weights (default: SurgSAM-2 endo18 finetuned)")
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_s.yaml",
                    help="Hydra config path relative to sam2/ (default: sam2.1_hiera_s)")
    ap.add_argument("--point", type=parse_point, action="append", default=[],
                    help="Positive click 'x,y' in original-video pixel coords (repeatable). Single-object mode.")
    ap.add_argument("--neg-point", type=parse_point, action="append", default=[],
                    help="Negative click 'x,y' (repeatable). Single-object mode.")
    ap.add_argument("--box", type=parse_box, default=None,
                    help="Box prompt 'x1,y1,x2,y2'. Single-object mode.")
    ap.add_argument("--object", dest="objects", type=parse_object, action="append", default=[],
                    help="Multi-object prompt 'OBJID:x,y[,x,y...]'. Repeat once per object. "
                         "Prefix a point with '!' for a negative click "
                         "(e.g. '--object 2:280,180,!100,100'). "
                         "When given, --point/--neg-point/--box/--obj-id are ignored.")
    ap.add_argument("--prompt-frame", type=int, default=0, help="Index of the frame on which clicks/box are given")
    ap.add_argument("--obj-id", type=int, default=1, help="Single-object mode: object id (any non-zero integer)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=float, default=None, help="Override fps of overlay video (defaults to source for mp4, 30 for frame dir)")
    ap.add_argument("--no-overlay-video", action="store_true", help="Skip writing the overlay mp4")
    ap.add_argument("--keep-extracted-frames", action="store_true", help="When --video is an mp4, keep the extracted JPEG frames inside output dir")
    args = ap.parse_args()

    if not args.objects and not args.point and not args.box:
        ap.error("You must provide at least one --object, --point, or --box")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download the SurgSAM-2 endo18 weights from\n"
            "  https://drive.google.com/file/d/1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI/view\n"
            "and place the file at the path above, OR pass --checkpoint pointing to a generic SAM 2.1 checkpoint."
        )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlay"
    masks_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)

    # --- Prepare frames directory ---
    cleanup_frames = False
    if os.path.isdir(args.video):
        frames_dir = args.video
        src_fps = args.fps or detect_fps_for_dir(frames_dir)
    else:
        if args.keep_extracted_frames:
            frames_dir = str(out_dir / "frames")
        else:
            frames_tmp = tempfile.mkdtemp(prefix="surgsam2_frames_")
            frames_dir = frames_tmp
            cleanup_frames = True
        print(f"Extracting frames from {args.video} -> {frames_dir}")
        src_fps = extract_frames_to_dir(args.video, Path(frames_dir))
        if args.fps:
            src_fps = args.fps

    loader_dir, source_offset, cleanup_loader = prepare_loader_dir(frames_dir)
    frame_names = list_frames(loader_dir)
    if not frame_names:
        raise RuntimeError(f"No frames found in {loader_dir}")
    print(f"Loaded {len(frame_names)} frames @ ~{src_fps:.2f} fps "
          f"(source frame offset: {source_offset})")

    # --- Build predictor ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if device.type == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(device.index or 0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    predictor = build_sam2_video_predictor(args.config, args.checkpoint, device=device)
    state = predictor.init_state(video_path=loader_dir)

    # --- Add prompts on the chosen frame ---
    # Build a normalized list of (obj_id, points_or_None, labels_or_None, box_or_None) tuples.
    object_specs = []
    if args.objects:
        for spec in args.objects:
            object_specs.append((spec['obj_id'],
                                 np.asarray(spec['points'], dtype=np.float32),
                                 np.asarray(spec['labels'], dtype=np.int32),
                                 None))
    else:
        pts = args.point + args.neg_point
        labels = [1] * len(args.point) + [0] * len(args.neg_point)
        object_specs.append((
            args.obj_id,
            np.asarray(pts, dtype=np.float32) if pts else None,
            np.asarray(labels, dtype=np.int32) if pts else None,
            np.asarray(args.box, dtype=np.float32) if args.box else None,
        ))

    print(f"Prompting {len(object_specs)} object(s) at frame {args.prompt_frame}")
    for oid, points_np, labels_np, box_np in object_specs:
        n_pos = int((labels_np == 1).sum()) if labels_np is not None else 0
        n_neg = int((labels_np == 0).sum()) if labels_np is not None else 0
        print(f"  obj {oid}: {n_pos} pos, {n_neg} neg, box={box_np is not None}")
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=args.prompt_frame,
            obj_id=oid,
            points=points_np,
            labels=labels_np,
            box=box_np,
        )

    # --- Propagate through video and write outputs ---
    H = state["video_height"]
    W = state["video_width"]
    print(f"Propagating across {len(frame_names)} frames at {W}x{H} ...")

    # Write the overlay as H.264 via the bundled imageio-ffmpeg binary so the
    # resulting mp4 plays inline in browsers / Jupyter. OpenCV's mp4v fourcc
    # produces MPEG-4 Part 2 which Firefox/Chrome refuse to play.
    ffmpeg_proc = None
    overlay_video_path = out_dir / "overlay.mp4"
    if not args.no_overlay_video:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_proc = subprocess.Popen(
            [ffmpeg_bin, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-r", f"{src_fps}", "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(overlay_video_path)],
            stdin=subprocess.PIPE,
        )

    n_processed = 0
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        # Combine all per-object masks into one palette PNG (later object id wins on overlap).
        combined = np.zeros((H, W), dtype=np.uint8)
        per_obj_masks = []
        for oid, logits in zip(out_obj_ids, out_mask_logits):
            m = (logits > 0.0).cpu().numpy().squeeze().astype(bool)
            per_obj_masks.append((int(oid), m))
            combined[m] = int(oid)
        mask_img = Image.fromarray(combined, mode="P")
        mask_img.putpalette(DAVIS_PALETTE)
        mask_img.save(masks_dir / f"{out_frame_idx:05d}.png")

        # Build overlay: blend each object with its DAVIS color.
        frame_path = os.path.join(loader_dir, frame_names[out_frame_idx])
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            continue
        overlay = img_bgr
        for oid, m in per_obj_masks:
            overlay = overlay_mask(overlay, m, davis_color_bgr(oid), alpha=0.5)
        cv2.imwrite(str(overlay_dir / f"{out_frame_idx:05d}.jpg"), overlay)
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.write(overlay.tobytes())
        n_processed += 1

    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    print(f"Done: {n_processed} frames processed.")
    print(f"Masks   -> {masks_dir}")
    print(f"Overlays -> {overlay_dir}")
    if not args.no_overlay_video:
        print(f"Overlay video -> {out_dir / 'overlay.mp4'}")

    cleanup_loader()
    if cleanup_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
