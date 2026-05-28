"""Convert Tier-B FT labels (data/ft_labels_v1.jsonl) → an HF detection dataset.

Phase 4 prep, HF-native path. Reads the JSONL emitted by tools/extract_ft_labels.py,
the deterministic split (configs/splits/whip_ft_v1.yaml), and the derived vocab
(configs/cardiac_whip_vocab.json), and writes a dataset tools/ft_gd_train.py can
load to fine-tune HF grounding-dino-tiny:

    <out>/train.jsonl       one line per (video, frame) image in the train split
    <out>/val.jsonl         same for the val split
    <out>/categories.json   the class list + the exact text prompt, in vocab order

Test-split videos are NOT converted — they stay held out for tools/eval_detection.py
and tools/extract_kinematics.py.

Per-image JSONL line:
    {"image_path": "<abs path>", "width": W, "height": H,
     "objects": {"bbox_xyxy": [[x0,y0,x1,y1], ...],   # pixel coords, source image
                 "category_id": [int, ...],            # index into vocab order
                 "category":    [str, ...],            # canonical name
                 "obj_id":      [int, ...]}}           # SAM track id (traceability)

Why xyxy pixels (not normalized cxcywh): keep the on-disk format unambiguous and
human-checkable; the trainer normalizes to the model's expected cxcywh at collate
time, where it also has the image size.

`category_id` is the index of each class in configs/cardiac_whip_vocab.json's
"instruments" list — the SAME ordering pipeline/prompts/dino.py feeds GD as the
text query at inference, so the fine-tuned head's class space lines up with the
zero-shot baseline (ablation parity).

Frame→image mapping: extract_ft_labels writes frame_idx = the tracker's loader
index (mask filename stem). pipeline.io.frame_files_ordered maps that index back
to the source frame file — the single source of truth shared with the loader.

Run in .sam3_venv (imports pipeline.io / pipeline.db + reads frames with PIL).
Does NOT need transformers / any GD code installed.

Usage (on bp):
    source .sam3_venv/bin/activate
    python -m tools.ft_dataset_convert \
        --in    data/ft_labels_v1.jsonl \
        --split  configs/splits/whip_ft_v1.yaml \
        --vocab  configs/cardiac_whip_vocab.json \
        --out    data/ft_hf_v1
    python -m tools.ft_dataset_convert --dry-run   # stats only, no files written
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from pipeline.db import REPO_ROOT, connect
from pipeline.io import frame_files_ordered

DEFAULT_IN = REPO_ROOT / "data" / "ft_labels_v1.jsonl"
DEFAULT_SPLIT = REPO_ROOT / "configs" / "splits" / "whip_ft_v1.yaml"
DEFAULT_VOCAB = REPO_ROOT / "configs" / "cardiac_whip_vocab.json"
DEFAULT_OUT = REPO_ROOT / "data" / "ft_hf_v1"


def load_vocab(path: Path) -> tuple[dict[str, int], list[dict]]:
    """Return ({instrument_id: label_id}, ordered_categories).

    ordered_categories[i] = {"id": i, "instrument_id": ..., "name": canonical}.
    Order is the file's "instruments" list order — must match dino.py's query order.
    """
    data = json.loads(path.read_text())
    id_to_label: dict[str, int] = {}
    categories: list[dict] = []
    for label_id, inst in enumerate(data["instruments"]):
        instrument_id = inst["instrument_id"]
        name = inst.get("canonical") or instrument_id
        id_to_label[instrument_id] = label_id
        categories.append({"id": label_id, "instrument_id": instrument_id, "name": name})
    return id_to_label, categories


def load_split(path: Path) -> dict[str, str]:
    """Parse the make_split YAML → {video_id: split_name}. No pyyaml dependency.

    Mirrors tools/make_split.write_yaml: a `train:`/`val:`/`test:` header line
    followed by `  - <video_id>` items.
    """
    video_split: dict[str, str] = {}
    current: str | None = None
    sections = {"train", "val", "test"}
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith(":") and stripped[:-1] in sections:
            current = stripped[:-1]
            continue
        if current and stripped.startswith("- "):
            video_split[stripped[2:].strip()] = current
    return video_split


def frames_dir_for(conn, video_id: str) -> Path:
    row = conn.execute(
        "SELECT frames_dir FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None:
        raise SystemExit(f"video_id {video_id!r} not in manifest videos table")
    return Path(row["frames_dir"])


def build_split_lines(
    conn,
    grouped: dict[str, dict[int, list[dict]]],
    id_to_label: dict[str, int],
    categories: list[dict],
) -> tuple[list[dict], Counter, list[str]]:
    """Build per-image JSONL records for one split.

    `grouped` is {video_id: {frame_idx: [ft_record, ...]}}. Returns
    (lines, per_class_box_counts, warnings).
    """
    name_for = {c["id"]: c["name"] for c in categories}
    lines: list[dict] = []
    per_class: Counter = Counter()
    warnings: list[str] = []

    for video_id, frames in sorted(grouped.items()):
        fdir = frames_dir_for(conn, video_id)
        try:
            ordered = frame_files_ordered(fdir)
        except RuntimeError as e:
            warnings.append(f"{video_id}: {e} — skipped")
            continue
        size_cache: dict[Path, tuple[int, int]] = {}

        for frame_idx, recs in sorted(frames.items()):
            if frame_idx >= len(ordered):
                warnings.append(
                    f"{video_id}: frame_idx {frame_idx} >= {len(ordered)} frames on disk — skipped"
                )
                continue
            fpath = (fdir / ordered[frame_idx]).resolve()
            if fpath not in size_cache:
                with Image.open(fpath) as im:
                    size_cache[fpath] = im.size  # (width, height)
            w, h = size_cache[fpath]

            bbox_xyxy: list[list[float]] = []
            category_id: list[int] = []
            category: list[str] = []
            obj_id: list[int] = []
            for r in recs:
                label = id_to_label[r["instrument_id"]]
                x0, y0, x1, y1 = r["box_xyxy"]
                bbox_xyxy.append([float(x0), float(y0), float(x1), float(y1)])
                category_id.append(label)
                category.append(name_for[label])
                obj_id.append(int(r["obj_id"]))
                per_class[name_for[label]] += 1

            lines.append({
                "image_path": str(fpath),
                "width": w,
                "height": h,
                "objects": {
                    "bbox_xyxy": bbox_xyxy,
                    "category_id": category_id,
                    "category": category,
                    "obj_id": obj_id,
                },
            })
    return lines, per_class, warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN,
                   help=f"FT-labels JSONL from extract_ft_labels (default {DEFAULT_IN}).")
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT,
                   help=f"Split YAML from make_split (default {DEFAULT_SPLIT}).")
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB,
                   help=f"Vocab JSON from build_vocab (default {DEFAULT_VOCAB}).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output dataset dir (default {DEFAULT_OUT}).")
    p.add_argument("--db", help="Manifest path override.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute + print stats but write nothing.")
    args = p.parse_args(argv)

    for required in (args.inp, args.split, args.vocab):
        if not required.exists():
            print(f"FATAL: missing input {required}", file=sys.stderr)
            return 2

    id_to_label, categories = load_vocab(args.vocab)
    video_split = load_split(args.split)
    conn = connect(args.db)

    # Group FT records: split_name -> video_id -> frame_idx -> [records].
    grouped: dict[str, dict[str, dict[int, list[dict]]]] = {
        "train": defaultdict(lambda: defaultdict(list)),
        "val": defaultdict(lambda: defaultdict(list)),
    }
    n_total = n_skipped_split = n_skipped_class = 0
    with open(args.inp) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            n_total += 1
            r = json.loads(raw)
            sp = video_split.get(r["video_id"])
            if sp not in ("train", "val"):
                # test-split videos and any video absent from the split are
                # intentionally excluded from the FT corpus.
                n_skipped_split += 1
                continue
            if r["instrument_id"] not in id_to_label:
                # class dropped by build_vocab's frequency threshold — not trainable.
                n_skipped_class += 1
                continue
            grouped[sp][r["video_id"]][int(r["frame_idx"])].append(r)

    train_lines, train_pc, train_warn = build_split_lines(
        conn, grouped["train"], id_to_label, categories)
    val_lines, val_pc, val_warn = build_split_lines(
        conn, grouped["val"], id_to_label, categories)

    # The text prompt GD sees. Build it with the SAME helper the inference
    # detector uses so the FT training prompt is byte-identical to the prompt
    # at inference (pipeline/prompts/dino.py → _grounding_dino_detector). GD's
    # class head is text-token-conditioned — the loss matches object queries to
    # token spans of THIS string — so any train/inference prompt drift silently
    # misaligns the fine-tuned class space. _build_prompt is a classmethod that
    # imports nothing heavy (numpy/PIL only), so it runs fine here in .sam3_venv.
    from pipeline.prompts._grounding_dino_detector import GroundingDinoDetector
    prompt, _ = GroundingDinoDetector._build_prompt([c["name"] for c in categories])

    print(f"--- ft_dataset_convert  (vocab: {len(categories)} classes)", file=sys.stderr)
    print(f"  input records:     {n_total}", file=sys.stderr)
    print(f"  skipped (split):   {n_skipped_split}  (test / not-in-split videos)", file=sys.stderr)
    print(f"  skipped (class):   {n_skipped_class}  (below build_vocab threshold)", file=sys.stderr)
    print(f"  train images:      {len(train_lines)}  ({sum(train_pc.values())} boxes)", file=sys.stderr)
    print(f"  val images:        {len(val_lines)}  ({sum(val_pc.values())} boxes)", file=sys.stderr)
    print(f"  prompt:            {prompt!r}", file=sys.stderr)
    print("  per-class boxes (train / val):", file=sys.stderr)
    for c in categories:
        nm = c["name"]
        print(f"    {nm:30}  {train_pc.get(nm, 0):6} / {val_pc.get(nm, 0):<6}", file=sys.stderr)
    for w in (*train_warn, *val_warn):
        print(f"  ⚠ {w}", file=sys.stderr)

    if args.dry_run:
        print("\n[dry-run] no files written", file=sys.stderr)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "train.jsonl", "w") as f:
        for line in train_lines:
            f.write(json.dumps(line) + "\n")
    with open(args.out / "val.jsonl", "w") as f:
        for line in val_lines:
            f.write(json.dumps(line) + "\n")
    (args.out / "categories.json").write_text(json.dumps({
        "prompt": prompt,
        "categories": categories,
        "_meta": {
            "vocab": str(args.vocab),
            "split": str(args.split),
            "source": str(args.inp),
            "n_classes": len(categories),
            "n_train_images": len(train_lines),
            "n_val_images": len(val_lines),
            "bbox_format": "xyxy_pixels",
        },
    }, indent=2))
    print(f"\nWrote {args.out}/ (train.jsonl, val.jsonl, categories.json)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
