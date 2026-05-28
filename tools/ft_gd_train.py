"""Fine-tune HF grounding-dino-tiny on the cardiac-whip detection dataset.

Phase 4, HF-native path. Consumes the dataset written by tools/ft_dataset_convert.py
(data/ft_hf_v1/{train,val}.jsonl + categories.json) and fine-tunes
`IDEA-Research/grounding-dino-tiny`. Output is a standard HF model directory the
Stage-1 prompter (pipeline/prompts/dino.py) loads directly via its `checkpoint`
override — no checkpoint conversion, so the ablation differs only in weights.

How GroundingDINO training works (the non-obvious part):
  GD's "class head" is text-conditioned — it scores each object query against the
  TOKENS of the prompt, not a fixed-size class layer. So we do NOT resize any head
  or touch config.num_labels. Instead:
    * prompt   = categories.json's "prompt" — built by the inference detector's
                 own _build_prompt helper so training and inference tokenize the
                 SAME string (vocab order preserved).
    * per box: class_labels = index of its class in that prompt (== category_id)
               boxes        = normalized cxcywh in [0, 1]
  The bipartite (Hungarian) loss then matches predicted queries to these targets.
  This matches transformers' own GroundingDinoForObjectDetection loss test.

Augmentation (albumentations, train split only):
  * HorizontalFlip on by default — we classify instrument TYPE, which is
    mirror-invariant; flip adds appearance variety the 38-video corpus badly needs.
  * brightness/contrast + hue jitter on by default — surgical lighting varies.
  * VerticalFlip off by default (endoscope views aren't fully orientation-free);
    enable with --vflip as an experiment.
  Boxes ride along via bbox_params format="pascal_voc" (xyxy pixels).

  NOTE: the da Vinci strip-crop (so the model can't read instrument names off the
  overlay) is intentionally NOT done here — it would need the identical crop in
  dino.py at inference or train/test geometry diverges. Deferred as a paired change.

LoRA (--lora) is optional and, after training, merged back into the base weights so
the saved dir is a plain HF model. Default is full fine-tune (safer for a pilot;
LoRA target-module selection on GD is less battle-tested).

Run in the HF FT env (transformers + accelerate + albumentations, + peft for --lora):
    sbatch slurm/ft_gd_train.sh
or directly:
    python -m tools.ft_gd_train \
        --data-dir data/ft_hf_v1 \
        --output-dir checkpoints/gdino_whip_ft_v1 \
        --epochs 15 --lr 5e-5 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Match pipeline.io: tolerate GPFS "truncated" PNG false-positives.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transform(hflip: bool, vflip: bool, photometric: bool):
    """albumentations pipeline over (image, xyxy-pixel boxes). None if no-op."""
    import albumentations as A

    tfs = []
    if hflip:
        tfs.append(A.HorizontalFlip(p=0.5))
    if vflip:
        tfs.append(A.VerticalFlip(p=0.5))
    if photometric:
        tfs.append(A.RandomBrightnessContrast(p=0.3))
        tfs.append(A.HueSaturationValue(p=0.2))
    if not tfs:
        return None
    return A.Compose(
        tfs,
        bbox_params=A.BboxParams(
            format="pascal_voc",          # xyxy absolute pixels
            label_fields=["category"],
            min_visibility=0.2,           # drop boxes mostly augmented out of frame
        ),
    )


class DetDataset(Dataset):
    """One item per image: PIL image + xyxy-pixel boxes + class ids (vocab order)."""

    def __init__(self, jsonl_path: Path, transform=None):
        self.records = [
            json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line.strip()
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        boxes = [list(map(float, b)) for b in r["objects"]["bbox_xyxy"]]
        labels = list(map(int, r["objects"]["category_id"]))

        if self.transform is not None and boxes:
            arr = np.asarray(img)
            out = self.transform(image=arr, bboxes=boxes, category=labels)
            img = Image.fromarray(out["image"])
            boxes = [list(map(float, b)) for b in out["bboxes"]]
            labels = list(map(int, out["category"]))

        return {"image": img, "boxes_xyxy": boxes, "labels": labels,
                "width": img.width, "height": img.height}


def make_collate(processor, prompt: str):
    """Tokenize the shared prompt, batch images, and build GD's labels list.

    boxes -> normalized cxcywh (resize-invariant, so we don't have to track the
    image processor's internal resize).
    """
    def collate(batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        enc = processor(images=images, text=[prompt] * len(images),
                        return_tensors="pt", padding=True)
        labels = []
        for b in batch:
            w, h = float(b["width"]), float(b["height"])
            cxcywh = []
            for (x0, y0, x1, y1) in b["boxes_xyxy"]:
                cxcywh.append([((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h,
                               (x1 - x0) / w, (y1 - y0) / h])
            labels.append({
                "class_labels": torch.tensor(b["labels"], dtype=torch.long),
                "boxes": (torch.tensor(cxcywh, dtype=torch.float32)
                          if cxcywh else torch.zeros((0, 4), dtype=torch.float32)),
            })
        enc["labels"] = labels
        return enc

    return collate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=Path("data/ft_hf_v1"),
                   help="Dir with train.jsonl, val.jsonl, categories.json.")
    p.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny",
                   help="Base HF model id or local path.")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/gdino_whip_ft_v1"))
    p.add_argument("--epochs", type=float, default=15)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Peak LR (full FT default; try 1e-4 with --lora).")
    p.add_argument("--batch-size", type=int, default=2)
    # Smoke knobs: validate the training plumbing in minutes before a 4h run.
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap total optimizer steps (-1 = use --epochs). "
                        "Smoke test: --max-steps 50 should drive loss down quickly.")
    p.add_argument("--limit-train", type=int, default=0,
                   help="Use only the first N train images (0 = all). For a "
                        "single-/few-image overfit smoke test.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    # augmentation
    p.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=True,
                   help="Horizontal flip (default ON — instrument TYPE is mirror-invariant).")
    p.add_argument("--vflip", action=argparse.BooleanOptionalAction, default=False,
                   help="Vertical flip (default OFF — experimental).")
    p.add_argument("--photometric", action=argparse.BooleanOptionalAction, default=True,
                   help="Brightness/contrast + hue jitter (default ON).")
    # LoRA
    p.add_argument("--lora", action="store_true",
                   help="LoRA fine-tune (merged back into base weights on save). "
                        "Default is full fine-tune.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    args = p.parse_args(argv)

    set_seed(args.seed)

    cats = json.loads((args.data_dir / "categories.json").read_text())
    prompt = cats["prompt"]
    n_classes = len(cats["categories"])
    print(f"[ft_gd_train] {n_classes} classes; prompt={prompt!r}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    # Deliberately do NOT set id2label / num_labels: GD's classification is
    # text-conditioned (scores queries against prompt tokens), so resizing a
    # class head would corrupt the architecture. The prompt carries the classes.

    if args.lora:
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules="all-linear", bias="none",
        )
        model = get_peft_model(model, lconf)
        model.print_trainable_parameters()

    transform = build_transform(args.hflip, args.vflip, args.photometric)
    print(f"[ft_gd_train] augmentation: hflip={args.hflip} vflip={args.vflip} "
          f"photometric={args.photometric}")

    train_ds = DetDataset(args.data_dir / "train.jsonl", transform=transform)
    if args.limit_train > 0:
        train_ds.records = train_ds.records[:args.limit_train]
        print(f"[ft_gd_train] SMOKE: limiting train to {len(train_ds)} images")
    val_path = args.data_dir / "val.jsonl"
    val_ds = DetDataset(val_path, transform=None) if val_path.exists() and len(
        [l for l in val_path.read_text().splitlines() if l.strip()]) else None
    print(f"[ft_gd_train] train images: {len(train_ds)}; "
          f"val images: {len(val_ds) if val_ds else 0}")

    collate = make_collate(processor, prompt)

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,        # -1 = disabled (use epochs)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        # In smoke mode (--max-steps) skip eval — we're checking plumbing, and
        # evaluating the full val set every epoch would dominate runtime.
        save_strategy="no" if args.max_steps > 0 else "epoch",
        eval_strategy="epoch" if (val_ds and args.max_steps <= 0) else "no",
        save_total_limit=2,
        remove_unused_columns=False,   # keep our image/boxes columns for the collator
        dataloader_num_workers=args.num_workers,
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collate,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    # Save a plain HF model dir the prompter can from_pretrained directly.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final = model
    if args.lora:
        final = model.merge_and_unload()
    final.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"[ft_gd_train] saved fine-tuned model → {args.output_dir}")
    print("[ft_gd_train] point stage1_prompting.checkpoint at this dir for FT inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
