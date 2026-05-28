# Stage-1 OCR handoff — instrument-strip timeline (run on Mac, CPU)

Hand this file to Claude on the Mac. It explains the task, the setup, how to
run, and how to sanity-check. Everything here lives on the `pipeline-clean`
branch.

## Goal

Read the da Vinci **instrument strip** at the bottom of each surgery frame and
produce a per-arm **instrument timeline**: which instrument is mounted on each
of the 4 robot arms, over the whole video. Output is a `segments.csv` per video
plus a timeline plot. This is **stage 1** of a larger surgical-instrument
tracking pipeline; the GPU stages (Grounding DINO detection + SAM2/SAM3 mask
tracking) run on the HPC cluster, **not** on the Mac. On the Mac we only do OCR.

## Use PaddleOCR, not Qwen (validated 2026-05-28)

There are two OCR backends in `tools/`:
- `ocr_slots_timeline_paddle.py` — **PaddleOCR. Use this.** CPU, fast, no GPU.
- `ocr_slots_timeline.py` — Qwen3-VL-8B. Needs a GPU; **do not use on the Mac.**

We frame-verified that **Qwen hallucinates instruments on arm 3** (the
camera/empty arm) — it invents `vessel sealer extend` / `tip-up fenestrated
grasper` where the strip is actually empty. PaddleOCR reads the empty arm
correctly. Paddle is therefore both cheaper *and* more accurate for this cohort.
(Paddle's only weakness: at 480p it drops leading letters, e.g.
`FENESTRATED`→`ENESTRATE`; the script counters this with 3× upscaling + fuzzy
matching to a fixed 6-name vocabulary.)

## Setup (Mac)

```bash
python3 -m venv .venv-ocr && source .venv-ocr/bin/activate
pip install -r requirements-ocr.txt
```

This is the lean OCR-only dependency set — **not** the full `pyproject.toml`
(which pulls torch/sam2/hydra for the GPU pipeline and is not needed here).
PaddleOCR pulls `paddlepaddle` automatically; if it doesn't on Apple Silicon,
run `pip install paddlepaddle`. The script auto-detects the PaddleOCR 2.x vs 3.x
API, so whichever version pip installs is fine.

## Input expectations

- One directory per video, frames named `frame_<srcnum>.png` (zero-padded ok),
  where `<srcnum>` is the source frame index (used for the timeline's
  `start_src`/`end_src`).
- `--source-fps` = the fps the frames were extracted at. It only affects the
  `start_sec`/`end_sec` columns (seconds = src / source-fps); it does NOT change
  the segmentation. The 38-video pilot used 30.

## Run — three steps, per cohort

```bash
# 1) OCR every video -> results/ocr_paddle/<video>/segments.csv (+ timeline.json, qc.json, raw_reads.csv)
for V in /path/to/frames/*/ ; do
  vid=$(basename "$V")
  python tools/ocr_slots_timeline_paddle.py \
      --video-id "$vid" --frames-dir "$V" \
      --out-dir results/ocr_paddle --source-fps 30
done

# 2) Smooth out OCR flicker (merge same-config runs + fill blank reads)
python tools/smooth_ocr_timeline.py \
    --in-dir results/ocr_paddle --out-dir results/ocr_paddle_smoothed

# 3) Render a Gantt timeline PNG per video
python tools/plot_ocr_timeline.py \
    --in-dir results/ocr_paddle_smoothed --out-dir plots/
```

## Output: `segments.csv`

One row per **segment** = a maximal time-run where the `{arm→instrument}` map is
constant. Columns:

| column | meaning |
|---|---|
| `start_i`, `end_i` | loader frame indices (position in the sorted frame list) |
| `n_frames` | `end_i - start_i + 1` |
| `start_src`, `end_src` | source frame numbers from `frame_<N>.png` |
| `start_sec`, `end_sec` | `src / source_fps` (seconds) |
| `arm1`..`arm4` | instrument name per arm (blank = empty/unmounted) |

Also written: `timeline.json` (per-frame), `qc.json` (summary: n_segments,
distinct instruments, timing), and **`raw_reads.csv`** (the raw PaddleOCR text +
confidence per sample per slot — your debugging trail when a slot looks wrong).

## What good output looks like (sanity checks)

- The whip cohort uses a **stable ~3-instrument config**: arm1 `small grasping
  retractor`, arm2 `fenestrated bipolar forceps`, arm4 `vessel sealer extend`;
  **arm3 is usually the camera arm → empty**. Many videos are a single segment —
  that's correct, the tools don't change.
- Eyeball the plot: arm1/2/4 should be solid bars; arm3 mostly empty. If arm3 is
  full of short colored blips, that's noise — increase smoothing `--min-seconds`.
- If an arm shows the wrong/garbled instrument, open `raw_reads.csv` for that
  video and look at the raw text Paddle read for that slot/sample.

## Knobs (defaults are tuned for the pilot; adjust per cohort)

`ocr_slots_timeline_paddle.py`: `--stride 30` (OCR every Nth frame),
`--strip-frac 0.17` (strip height as fraction of frame), `--upscale 3.0`
(critical at low res), `--fuzz-threshold 70`, `--paddle-conf-floor 0.5`,
`--source-fps`.
`smooth_ocr_timeline.py`: `--min-seconds 5` (segments shorter than this are
absorbed as flicker), `--no-fill-blanks` to disable per-slot blank fill.

## Gotchas

- **Slot geometry** (`SLOT_FRACS_169` / `SLOT_FRACS_32` near the top of the
  Paddle script) is tuned for 16:9 (1080p) and 3:2 (480p) recordings. If the new
  49 videos have a different strip layout/aspect ratio, the per-arm crops will be
  wrong (names bleed between arms). Fix: crop a strip from one frame, eyeball
  where the 4 numbered boxes fall, and update the fractions.
- The 6-name vocabulary is `configs/cardiac_whip_vocabulary.json`. If the new
  cohort uses different instruments, add them there (canonical name + any
  on-screen UI variants).
- Do **not** switch to the Qwen backend to "improve" results — it hallucinates
  arm3 (see above).

## Files

| file | role |
|---|---|
| `tools/ocr_slots_timeline_paddle.py` | the OCR (PaddleOCR backend) |
| `tools/_ocr_vocab.py` | loads the instrument vocabulary |
| `tools/smooth_ocr_timeline.py` | flicker cleanup |
| `tools/plot_ocr_timeline.py` | Gantt timeline plot |
| `configs/cardiac_whip_vocabulary.json` | the 6 instruments + UI variants |
| `requirements-ocr.txt` | lean CPU deps for this workflow |
