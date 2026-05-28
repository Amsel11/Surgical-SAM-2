"""Production instrument-presence timeline via SPATIAL per-slot OCR — PaddleOCR backend.

Drop-in alternative to tools/ocr_slots_timeline.py (Qwen3-VL backend). Identical
slot-crop logic, identical outputs (segments.csv, timeline.json, qc.json) so
the two can be diffed directly. PaddleOCR is ~10x faster on a single GPU and
free of LLM hallucination, but at 480p it drops leading/trailing characters
(observed: FENESTRATED -> ENESTRATE / ENETRATE). We counter with:
  - upscaling each slot crop --upscale before OCR (default 3x)
  - rapidfuzz token_set_ratio mapping to the canonical name list
  - mode-indicator stripping (coag/seal/cut etc.) BEFORE fuzzy match so
    short noise tokens don't poison the score

Persists raw paddle reads per (sample, slot) to raw_reads.csv so failures
can be audited after the fact — this is the diagnostic gap the Qwen
pipeline currently has.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools._ocr_vocab import load_vocab  # noqa: E402


# ---------------------------------------------------------------------------
# Slot geometry — kept byte-identical to ocr_slots_timeline.py so outputs
# are directly comparable. Do not edit without updating the Qwen version too.
# ---------------------------------------------------------------------------
SLOT_FRACS_169 = {
    1: (0.13, 0.35),
    2: (0.31, 0.52),
    3: (0.48, 0.68),
    4: (0.64, 0.85),
}
SLOT_FRACS_32 = {
    1: (0.00, 0.30),
    2: (0.25, 0.52),
    3: (0.45, 0.76),
    4: (0.70, 1.00),
}
SLOTS = (1, 2, 3, 4)

# Mode-indicator tokens shown on the strip alongside instrument names. These
# are NOT instruments and must be stripped before fuzzy matching, otherwise
# short tokens like "cut" / "seal" wrongly inflate scores against canonical
# names that share those substrings. Paddle sometimes drops the trailing
# letter (LCOAG -> LCOA), so the set is more permissive than the vocab file.
_MODE_HINTS_DEFAULT = {
    "coag", "lcoag", "rcoag", "lcoa", "rcoa",
    "seal", "lseal", "rseal",
    "cut", "rcut", "lcut",
    "laser", "liser", "laseroff",
}


def get_slot_fracs(w: int, h: int) -> dict:
    return SLOT_FRACS_169 if (w / max(h, 1)) >= 1.6 else SLOT_FRACS_32


def src_idx_from_name(name: str):
    stem = Path(name).stem
    if stem.startswith("frame_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return None
    return None


def strip_crop(img, strip_px):
    return img[img.shape[0] - strip_px:, :]


def strip_diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 255.0
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def mode_window(seq, W):
    out = [None] * len(seq)
    half = W // 2
    for i in range(len(seq)):
        win = seq[max(0, i - half): i + half + 1]
        c = collections.Counter(x for x in win if x is not None)
        out[i] = c.most_common(1)[0][0] if c else None
    return out


# ---------------------------------------------------------------------------
# PaddleOCR + fuzzy canonical mapping
# ---------------------------------------------------------------------------

def upscale(img, factor):
    if factor <= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(
        img, (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_CUBIC,
    )


def build_paddle(use_gpu: bool):
    """Construct a PaddleOCR reader across the 2.x and 3.x APIs.

    2.x: PaddleOCR(use_angle_cls=, show_log=); 3.x dropped both and renamed to
    use_textline_orientation. Try newest-first, fall back, so whatever pip
    installed on the Mac works.
    """
    from paddleocr import PaddleOCR
    attempts = [
        dict(lang="en", use_textline_orientation=False),       # 3.x
        dict(lang="en", use_angle_cls=False, show_log=False),  # 2.x
        dict(lang="en"),                                       # bare
    ]
    last = None
    for kw in attempts:
        if use_gpu:
            kw = {**kw, "use_gpu": True}
        try:
            return PaddleOCR(**kw)
        except (TypeError, ValueError) as e:
            last = e
            if use_gpu:  # retry once without the gpu kwarg (3.x removed it)
                try:
                    return PaddleOCR(**{k: v for k, v in kw.items() if k != "use_gpu"})
                except (TypeError, ValueError) as e2:
                    last = e2
    raise RuntimeError(f"Could not construct PaddleOCR with any known API: {last}")


def _extract_rows(result):
    """Normalize PaddleOCR output to [(x0, text, conf), ...] across versions.

    3.x .predict(): list of dict-like OCRResult with rec_texts / rec_scores /
    rec_polys (or rec_boxes). 2.x .ocr(): [[ [box, (text, conf)], ... ]].
    """
    rows = []
    if not result:
        return rows
    first = result[0]
    # 3.x OCRResult (dict-like with rec_texts)
    if hasattr(first, "get") and first.get("rec_texts") is not None:
        texts = first.get("rec_texts") or []
        scores = first.get("rec_scores") or []
        polys = (first.get("rec_polys") or first.get("dt_polys")
                 or first.get("rec_boxes") or [])
        for i, t in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 0.0
            x0 = 0.0
            if i < len(polys) and polys[i] is not None:
                pts = polys[i]
                try:
                    x0 = float(min(p[0] for p in pts))      # polygon
                except (TypeError, IndexError):
                    try:
                        x0 = float(pts[0])                   # [x0,y0,x1,y1]
                    except (TypeError, IndexError):
                        x0 = 0.0
            rows.append((x0, str(t), conf))
        return rows
    # 2.x nested-list format
    page = first if isinstance(first, list) else result
    for line in page:
        if not line or len(line) < 2:
            continue
        box, payload = line[0], line[1]
        if not payload:
            continue
        text = str(payload[0])
        conf = float(payload[1]) if len(payload) > 1 else 0.0
        x0 = min(p[0] for p in box) if box else 0.0
        rows.append((float(x0), text, conf))
    return rows


def paddle_read(ocr, region, upscale_factor):
    """Run PaddleOCR on a single slot region. Returns (concatenated text,
    mean confidence, raw detections), left-to-right. API-version agnostic."""
    up = upscale(region, upscale_factor)
    result = None
    if hasattr(ocr, "predict"):           # 3.x preferred
        try:
            result = ocr.predict(up)
        except Exception:
            result = None
    if not result:
        try:
            result = ocr.ocr(up, cls=False)   # 2.x
        except TypeError:
            try:
                result = ocr.ocr(up)
            except Exception as e:
                print(f"  paddle error: {e}", file=sys.stderr)
                return "", 0.0, []
        except Exception as e:
            print(f"  paddle error: {e}", file=sys.stderr)
            return "", 0.0, []
    rows = _extract_rows(result)
    if not rows:
        return "", 0.0, []
    rows.sort(key=lambda r: r[0])
    text = " ".join(t for _, t, _ in rows).strip().lower()
    mean_conf = float(np.mean([c for _, _, c in rows]))
    raw = [{"text": t, "conf": c} for _, t, c in rows]
    return text, mean_conf, raw


def build_fuzz_choices(canonical_names, ui_to_canonical):
    """Map every searchable string -> canonical name. Includes canonical
    names themselves so an exact canonical read is a 100-score match."""
    choices = {c: c for c in canonical_names}
    for variant, canon in ui_to_canonical.items():
        choices[variant] = canon
    return choices


def strip_mode_hints(text, mode_hints):
    toks = re.findall(r"[a-z]+", text.lower())
    keep = [t for t in toks if t not in mode_hints]
    return " ".join(keep)


def fuzzy_canonical(text, choices, fuzz_threshold, mode_hints, fuzz_mod):
    """Returns (label, score) where label is one of:
       canonical name | 'camera' | 'unknown' | None
       and score is the rapidfuzz score (0-100) or 100 for special-case hits.

       'empty' is NOT returned here — empty crops produce text='' which
       returns (None, 0.0). The mode-filter pipeline downstream treats
       None and 'empty' equivalently (both dropped from the arm map),
       so we don't need to disambiguate.
    """
    if not text:
        return None, 0.0

    t = text.lower()
    # Camera arm — strip shows scope angle / zoom / "LASER OFF" / "UNDOCK
    # BEFORE MOVING TABLE" instead of an instrument name. Detect by any
    # camera-only keyword OR by zoom-and-angle pattern (which paddle may
    # render with or without separators: "1x30" joined or "1x 30").
    _CAMERA_KEYWORDS = (
        "camera", "scope", "endoscop",
        "laser off", "laseroff",
        "undock", "moving table", "movingtadlc", "movingtable",
        "1x30", "0x30", "1x0", "30 laser",
    )
    if any(k in t for k in _CAMERA_KEYWORDS):
        return "camera", 100.0
    if re.search(r"\b\d{1,3}\s*°|\b\d+\s*x\b", t):
        return "camera", 100.0
    # NOTE: we used to map bare integer ("3") -> camera, but that's wrong.
    # Slot 3 is 94% occluded in cardiac whip footage and paddle frequently
    # reads just the slot-number indicator "3" — that should be 'unknown',
    # not 'camera'. The Qwen pipeline correctly returns None here.

    cleaned = strip_mode_hints(t, mode_hints)
    if not cleaned:
        # Slot showed only mode indicators — instrument name unreadable
        # but the arm IS active. Surface as 'unknown' so it's distinguishable
        # from empty in raw_reads.csv even though both get filtered downstream.
        return "unknown", 0.0

    # Length gate: paddle fragments shorter than 5 chars (e.g. "ctor",
    # "vez", single digits) cannot be reliably mapped — partial_ratio
    # will happily match a 4-char fragment to a 24-char canonical name.
    # Surface as unknown so the mode filter can vote it out.
    if len(cleaned.replace(" ", "")) < 5:
        return "unknown", 0.0

    # Cheap exact-match pass first (high precision, no fuzzy noise).
    if cleaned in choices:
        return choices[cleaned], 100.0

    # Score every candidate with two scorers and take the max. Dropped
    # partial_ratio after v2 showed it false-matching 4-char fragments
    # like "ctor" to "small grasping retractor" via the trailing-suffix
    # alignment. ratio handles fused-word cases ("shallgraspingretractor"
    # -> "small grasping retractor" ~70%); token_set_ratio handles
    # word-order / extra-word cases.
    best_key, best_score = None, -1.0
    for k in choices:
        s = max(
            fuzz_mod.fuzz.ratio(cleaned, k),
            fuzz_mod.fuzz.token_set_ratio(cleaned, k),
        )
        if s > best_score:
            best_score, best_key = s, k
    if best_key is None or best_score < fuzz_threshold:
        return None, float(best_score if best_score > 0 else 0.0)
    return choices[best_key], float(best_score)


# ---------------------------------------------------------------------------
# Frame access — a frames-dir of PNGs, OR a video decoded in-memory (no files)
# ---------------------------------------------------------------------------

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg")


def _strip_px(h, strip_px, strip_frac):
    return strip_px if strip_px > 0 else max(40, int(h * strip_frac))


def iter_video_files(videos_dir: Path):
    return sorted(p for p in Path(videos_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in _VIDEO_EXTS)


def decode_video_strips(video_path: Path, target_fps: float,
                        strip_px: int, strip_frac: float):
    """Stream a video and keep ONLY the bottom-strip crop of every
    (native_fps / target_fps)-th frame, in memory. No full frames retained, no
    files written. Returns (strips, srcs, w, h, native_fps).

    Decoding is sequential (reliable across codecs — no frame seeking).
    `srcs` are the true source frame indices, so seconds = src / native_fps.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / max(target_fps, 1e-6)))
    strips, srcs = [], []
    w = h = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if not strips:
                h, w = frame.shape[:2]
            sp = _strip_px(frame.shape[0], strip_px, strip_frac)
            strips.append(frame[frame.shape[0] - sp:, :].copy())
            srcs.append(idx)
        idx += 1
    cap.release()
    return strips, srcs, w, h, native


# ---------------------------------------------------------------------------
# Per-video pipeline (backend-agnostic: takes strip-crop accessors)
# ---------------------------------------------------------------------------

def process_one(*, video_id, n, get_strip, get_src, w, h, src_fps, out_dir,
                ocr, choices, mode_hints, fuzz_mod, args):
    """Run the OCR slot-timeline pipeline for one video.

    `get_strip(i)` returns the bottom-strip crop (ndarray, full frame width) or
    None; `get_src(i)` returns that sample's source frame index.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fracs = get_slot_fracs(w, h)

    sample_idx = list(range(0, n, args.stride))
    if (n - 1) not in sample_idx:
        sample_idx.append(n - 1)
    raw_canon = {s: [] for s in SLOTS}
    raw_text  = {s: [] for s in SLOTS}
    raw_score = {s: [] for s in SLOTS}
    raw_conf  = {s: [] for s in SLOTS}

    t0 = time.time()
    n_calls = 0
    for k, fi in enumerate(sample_idx):
        strip = get_strip(fi)
        if strip is None:
            for s in SLOTS:
                raw_canon[s].append(None); raw_text[s].append("")
                raw_score[s].append(0.0); raw_conf[s].append(0.0)
            continue
        for slot in SLOTS:
            a, b = fracs[slot]
            region = strip[:, int(w * a): int(w * b)]
            text, conf, _ = paddle_read(ocr, region, args.upscale)
            if conf < args.paddle_conf_floor:
                canon, score = None, 0.0
            else:
                canon, score = fuzzy_canonical(
                    text, choices, args.fuzz_threshold, mode_hints, fuzz_mod)
            raw_canon[slot].append(canon)
            raw_text[slot].append(text)
            raw_score[slot].append(score)
            raw_conf[slot].append(conf)
            n_calls += 1
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(sample_idx)} samples  {time.time()-t0:.0f}s")

    sm = {s: mode_window(raw_canon[s], args.mode_window) for s in SLOTS}

    def keep(v):
        return v is not None and v not in ("empty", "unknown")
    sample_cfg = [{s: sm[s][j] for s in SLOTS if keep(sm[s][j])}
                  for j in range(len(sample_idx))]

    seg_bounds = [0]
    for j in range(1, len(sample_cfg)):
        if sample_cfg[j] != sample_cfg[j - 1]:
            seg_bounds.append(j)
    seg_bounds.append(len(sample_idx))

    def refine_boundary(lo_fi, hi_fi):
        prev = None
        best_fi, best_d = lo_fi, -1.0
        for fi in range(lo_fi, hi_fi + 1):
            st = get_strip(fi)
            if st is None:
                continue
            if prev is not None:
                d = strip_diff(prev, st)
                if d > best_d:
                    best_d, best_fi = d, fi
            prev = st
        return best_fi if best_d >= args.refine_thresh else hi_fi

    segments = []
    for b in range(len(seg_bounds) - 1):
        j0 = seg_bounds[b]
        j1 = seg_bounds[b + 1] - 1
        cfg = sample_cfg[j0]
        start_fi = sample_idx[j0]
        if b > 0:
            start_fi = refine_boundary(sample_idx[seg_bounds[b] - 1], sample_idx[j0])
        end_fi = sample_idx[j1]
        segments.append({"start": start_fi, "end": end_fi, "cfg": cfg})
    for b in range(len(segments) - 1):
        segments[b]["end"] = segments[b + 1]["start"] - 1
    if segments:
        segments[-1]["end"] = n - 1

    merged = []
    for seg in segments:
        if merged and merged[-1]["cfg"] == seg["cfg"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    segments = merged

    timeline = []
    for seg in segments:
        for fi in range(seg["start"], seg["end"] + 1):
            src = get_src(fi)
            timeline.append({
                "i": fi, "src_frame": src,
                "sec": round(src / src_fps, 2) if src is not None else None,
                "arms": {str(k): v for k, v in sorted(seg["cfg"].items())},
                "instruments": sorted(set(seg["cfg"].values())),
            })
    elapsed = time.time() - t0

    (out_dir / "timeline.json").write_text(json.dumps({
        "video_id": video_id, "n_frames": n, "stride": args.stride,
        "n_ocr_calls": n_calls, "n_segments": len(segments),
        "elapsed_sec": round(elapsed, 1), "frames": timeline,
        "backend": "paddle", "src_fps": src_fps, "upscale": args.upscale,
        "fuzz_threshold": args.fuzz_threshold,
        "paddle_conf_floor": args.paddle_conf_floor,
    }, indent=2))

    with (out_dir / "segments.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["start_i", "end_i", "n_frames", "start_src", "end_src",
                      "start_sec", "end_sec", "arm1", "arm2", "arm3", "arm4"])
        for seg in segments:
            s_src = get_src(seg["start"])
            e_src = get_src(seg["end"])
            cfg = seg["cfg"]
            wtr.writerow([
                seg["start"], seg["end"], seg["end"] - seg["start"] + 1,
                s_src, e_src,
                round(s_src / src_fps, 1) if s_src is not None else "",
                round(e_src / src_fps, 1) if e_src is not None else "",
                cfg.get(1, ""), cfg.get(2, ""), cfg.get(3, ""), cfg.get(4, ""),
            ])

    with (out_dir / "raw_reads.csv").open("w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["sample_idx", "frame_idx", "slot", "paddle_text",
                      "paddle_conf", "fuzz_score",
                      "canon_pre_mode_filter", "canon_post_mode_filter"])
        for j, fi in enumerate(sample_idx):
            for s in SLOTS:
                wtr.writerow([j, fi, s, raw_text[s][j], round(raw_conf[s][j], 3),
                              round(raw_score[s][j], 1), raw_canon[s][j] or "",
                              sm[s][j] or ""])

    distinct = sorted({i for seg in segments for i in seg["cfg"].values()})
    (out_dir / "qc.json").write_text(json.dumps({
        "video_id": video_id, "n_frames": n, "n_ocr_calls": n_calls,
        "n_segments": len(segments), "distinct_instruments": distinct,
        "stride": args.stride, "elapsed_sec": round(elapsed, 1),
        "sec_per_sample": round(elapsed / max(len(sample_idx), 1), 3),
        "backend": "paddle", "src_fps": src_fps, "upscale": args.upscale,
        "fuzz_threshold": args.fuzz_threshold,
        "paddle_conf_floor": args.paddle_conf_floor,
    }, indent=2))

    print(f"\n{video_id} done: {n} samples, {n_calls} OCR calls, "
          f"{len(segments)} segments, {elapsed:.0f}s; instruments: {distinct}")
    for seg in segments:
        print(f"  [{seg['start']:5d}-{seg['end']:5d}] {seg['cfg']}")
    print(f"  wrote {out_dir}/segments.csv (+ timeline.json, raw_reads.csv, qc.json)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Input: exactly one of --frames-dir / --video / --videos-dir.
    ap.add_argument("--frames-dir", type=Path,
                    help="Dir of frame_<n>.png for ONE video.")
    ap.add_argument("--video", type=Path,
                    help="A single video file (decoded in-memory, no frames saved).")
    ap.add_argument("--videos-dir", type=Path,
                    help="Dir of video files; each is processed in turn, in-memory.")
    ap.add_argument("--video-id",
                    help="Override the output id (default: dir/file name).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--queries-file", type=Path,
                    default=REPO_ROOT / "configs/cardiac_whip_vocabulary.json")
    ap.add_argument("--strip-px", type=int, default=0,
                    help="Fixed strip height in px; 0 = use --strip-frac.")
    ap.add_argument("--strip-frac", type=float, default=0.17,
                    help="Strip height as fraction of frame height.")
    ap.add_argument("--stride", type=int, default=30,
                    help="OCR every Nth sampled frame.")
    ap.add_argument("--mode-window", type=int, default=3)
    ap.add_argument("--refine-thresh", type=float, default=12.0)
    ap.add_argument("--source-fps", type=float, default=30.0,
                    help="fps for the seconds columns in --frames-dir mode "
                         "(video modes read fps from the file).")
    ap.add_argument("--video-fps", type=float, default=1.0,
                    help="Sampling fps when decoding a video (default 1/s).")
    ap.add_argument("--upscale", type=float, default=3.0)
    ap.add_argument("--fuzz-threshold", type=int, default=70)
    ap.add_argument("--paddle-conf-floor", type=float, default=0.5)
    ap.add_argument("--use-gpu", action="store_true")
    args = ap.parse_args()

    n_inputs = sum(x is not None for x in (args.frames_dir, args.video, args.videos_dir))
    if n_inputs != 1:
        ap.error("provide exactly one of --frames-dir / --video / --videos-dir")

    canonical_names, ui_to_canonical, modes = load_vocab(args.queries_file)
    choices = build_fuzz_choices(canonical_names, ui_to_canonical)
    mode_hints = set(_MODE_HINTS_DEFAULT) | {m.lower() for m in modes}

    print("Loading PaddleOCR + rapidfuzz ...")
    import rapidfuzz as _rf

    class _Fuzz:
        process = _rf.process
        fuzz = _rf.fuzz
    fuzz_mod = _Fuzz()
    ocr = build_paddle(args.use_gpu)
    print("PaddleOCR ready.\n")

    # Build the list of videos to process.
    if args.videos_dir:
        vids = iter_video_files(args.videos_dir)
        if not vids:
            print(f"No video files in {args.videos_dir}", file=sys.stderr)
            return 1
        jobs = [("video", v, args.video_id or v.stem) for v in vids]
    elif args.video:
        jobs = [("video", args.video, args.video_id or args.video.stem)]
    else:
        jobs = [("frames", args.frames_dir, args.video_id or args.frames_dir.name)]

    rc = 0
    for kind, ref, vid in jobs:
        try:
            if kind == "video":
                strips, srcs, w, h, native = decode_video_strips(
                    ref, args.video_fps, args.strip_px, args.strip_frac)
                if not strips:
                    print(f"{vid}: no frames decoded — skipping", file=sys.stderr)
                    rc = 1
                    continue
                print(f"{vid}: {len(strips)} sampled frames @ {args.video_fps}/s "
                      f"({w}x{h}, native {native:.1f} fps)")
                process_one(
                    video_id=vid, n=len(strips),
                    get_strip=lambda i, _s=strips: _s[i],
                    get_src=lambda i, _s=srcs: _s[i],
                    w=w, h=h, src_fps=native, out_dir=args.out_dir / vid,
                    ocr=ocr, choices=choices, mode_hints=mode_hints,
                    fuzz_mod=fuzz_mod, args=args)
            else:
                paths = sorted(
                    list(ref.glob("frame_*.png")) + list(ref.glob("frame_*.jpg")),
                    key=lambda p: src_idx_from_name(p.name) or 0)
                if not paths:
                    print(f"No frames in {ref}", file=sys.stderr)
                    rc = 1
                    continue
                probe = cv2.imread(str(paths[0]))
                h, w = probe.shape[:2]

                def _gs(i, _p=paths):
                    img = cv2.imread(str(_p[i]))
                    if img is None:
                        return None
                    sp = _strip_px(img.shape[0], args.strip_px, args.strip_frac)
                    return img[img.shape[0] - sp:, :]

                print(f"{vid}: {len(paths)} frames ({w}x{h})")
                process_one(
                    video_id=vid, n=len(paths),
                    get_strip=_gs,
                    get_src=lambda i, _p=paths: src_idx_from_name(_p[i].name),
                    w=w, h=h, src_fps=args.source_fps, out_dir=args.out_dir / vid,
                    ocr=ocr, choices=choices, mode_hints=mode_hints,
                    fuzz_mod=fuzz_mod, args=args)
        except Exception as e:
            print(f"FAILED {vid}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
