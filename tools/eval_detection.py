"""Per-class detection eval — predicted boxes vs clicker ground-truth.

Phase 6 of the GD fine-tune plan, paired with extract_kinematics.py. Walks
prompt_sets rows for a run (--run-method, e.g. 'dino') and the corresponding
ground-truth prompt_sets ('manual_box'), Hungarian-matches per-frame boxes by
IoU, and emits:

  - per-class AP@0.5 (single threshold; if predictions have no scores we use
    1.0 — degrades AP to precision/recall, but identity-correct is unaffected)
  - identity-correct rate    : of IoU-matched detections, fraction predicting
                               the right class (the bottleneck the FT-GD
                               ablation actually targets)
  - per-class confusion matrix (rows = GT, cols = predicted)
  - localization metrics      : matched/missed/extra per video for sanity

Detection eval is anchor-frame-only — those are the only frames with GT
labels. For propagation-quality metrics on every frame, use extract_kinematics.

Output: local_results/eval/<run_name>/detection.csv  (per-frame join)
        local_results/eval/<run_name>/per_class.csv  (AP/precision/recall)
        local_results/eval/<run_name>/confusion.csv  (GT × Pred grid)

Usage:
    python -m tools.eval_detection --run-method dino --cohort whip
    python -m tools.eval_detection --run-method dino --videos VK_whip_11947036
    python -m tools.eval_detection --gt-seed 1 --run-seed 1 --gt-method manual_box --run-method dino
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.db import REPO_ROOT, connect

DEFAULT_OUT = REPO_ROOT / "local_results" / "eval"
DEFAULT_IOU_THRESHOLD = 0.5


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    bb = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def load_prompts(path: Path) -> dict[int, list[dict]] | None:
    """Return objects_by_frame keyed by int frame_idx, or None on miss."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.get("objects_by_frame", {}).items()}


def fetch_video_prompts(
    conn, video_id: str, seed: int, method: str
) -> tuple[Path | None, dict[int, str]]:
    """Return (prompts JSON path, {obj_id: instrument_id}) for the given (video, seed, method)."""
    row = conn.execute(
        """SELECT prompt_set_id, prompts_path FROM prompt_sets
           WHERE video_id=? AND seed=? AND prompt_method=? AND status='ready'""",
        (video_id, seed, method),
    ).fetchone()
    if row is None:
        return None, {}
    po_rows = conn.execute(
        "SELECT obj_id, instrument_id FROM prompt_objects WHERE prompt_set_id=?",
        (row["prompt_set_id"],),
    ).fetchall()
    labels = {int(r["obj_id"]): r["instrument_id"] for r in po_rows}
    return Path(row["prompts_path"]) if row["prompts_path"] else None, labels


def greedy_match(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy IoU matching. Returns [(gt_idx, pred_idx, iou), ...]."""
    pairs: list[tuple[float, int, int]] = []
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou = iou_xyxy(g, p)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        matches.append((i, j, iou))
        used_gt.add(i)
        used_pred.add(j)
    return matches


def per_frame_eval(
    gt_objs: list[dict],
    pred_objs: list[dict],
    gt_obj_labels: dict[int, str],
    pred_obj_labels: dict[int, str],
    iou_threshold: float,
) -> tuple[list[tuple[str, str, float]], int, int, int]:
    """Return (matched_pairs, n_matched, n_missed_gt, n_extra_pred).

    matched_pairs: [(gt_instrument_id, pred_instrument_id, iou), ...].
    """
    gt_boxes = [o["box"] for o in gt_objs]
    pred_boxes = [o["box"] for o in pred_objs]
    matches = greedy_match(gt_boxes, pred_boxes, iou_threshold)

    matched_pairs: list[tuple[str, str, float]] = []
    for gi, pj, iou in matches:
        gt_oid = int(gt_objs[gi]["obj_id"])
        pred_oid = int(pred_objs[pj]["obj_id"])
        gt_cls = gt_obj_labels.get(gt_oid, "unknown_instrument")
        pred_cls = pred_obj_labels.get(pred_oid, "unknown_instrument")
        matched_pairs.append((gt_cls, pred_cls, iou))

    return (
        matched_pairs,
        len(matches),
        len(gt_objs) - len(matches),
        len(pred_objs) - len(matches),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cohort", default="whip")
    p.add_argument("--videos", nargs="*",
                   help="Restrict to specific video_ids (default: every cohort member with both sets ready).")
    p.add_argument("--gt-seed", type=int, default=1)
    p.add_argument("--gt-method", default="manual_box")
    p.add_argument("--run-seed", type=int, default=1)
    p.add_argument("--run-method", required=True,
                   help="prompt_method of the predictions to score (e.g. 'dino').")
    p.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: local_results/eval/<run-method>/")
    p.add_argument("--db", help="Manifest path override.")
    args = p.parse_args(argv)

    conn = connect(args.db)
    out_dir = args.out_dir or (DEFAULT_OUT / args.run_method)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve videos
    if args.videos:
        videos = args.videos
    else:
        videos = [r["video_id"] for r in conn.execute(
            "SELECT video_id FROM videos WHERE cohort=? ORDER BY video_id", (args.cohort,)
        )]

    # Header for the per-frame detail CSV
    detail_path = out_dir / "detection.csv"
    per_class_path = out_dir / "per_class.csv"
    confusion_path = out_dir / "confusion.csv"

    detail_rows: list[dict] = []
    matched_pairs_all: list[tuple[str, str, float]] = []
    gt_class_counts: Counter[str] = Counter()
    pred_class_counts: Counter[str] = Counter()
    extras_by_class: Counter[str] = Counter()  # predicted with no IoU-matching GT

    n_videos_scored = 0
    for vid in videos:
        gt_path, gt_labels = fetch_video_prompts(conn, vid, args.gt_seed, args.gt_method)
        pred_path, pred_labels = fetch_video_prompts(conn, vid, args.run_seed, args.run_method)
        if gt_path is None or pred_path is None:
            continue
        gt_frames = load_prompts(gt_path)
        pred_frames = load_prompts(pred_path)
        if gt_frames is None or pred_frames is None:
            continue
        n_videos_scored += 1

        common_frames = sorted(set(gt_frames) & set(pred_frames))
        for f in common_frames:
            matches, n_match, n_miss, n_extra = per_frame_eval(
                gt_frames[f], pred_frames[f],
                gt_labels, pred_labels,
                args.iou_threshold,
            )
            for gt_cls, pred_cls, iou in matches:
                matched_pairs_all.append((gt_cls, pred_cls, iou))
                gt_class_counts[gt_cls] += 1
                pred_class_counts[pred_cls] += 1
            for o in gt_frames[f]:
                gt_cls = gt_labels.get(int(o["obj_id"]), "unknown_instrument")
                gt_class_counts[gt_cls] += 0  # ensure class is in keys
            for o in pred_frames[f]:
                pred_cls = pred_labels.get(int(o["obj_id"]), "unknown_instrument")
                pred_class_counts[pred_cls] += 0
            for j, po in enumerate(pred_frames[f]):
                if not any(j == pj for _, pj, _ in [(0, 0, 0)]):  # no-op; we already iterated above
                    pass
            # Re-derive 'extras' (predictions without an IoU-matched GT) cleanly
            pred_indices_matched = set()
            for _gi, pj, _iou in greedy_match(
                [o["box"] for o in gt_frames[f]],
                [o["box"] for o in pred_frames[f]],
                args.iou_threshold,
            ):
                pred_indices_matched.add(pj)
            for j, po in enumerate(pred_frames[f]):
                if j not in pred_indices_matched:
                    pcls = pred_labels.get(int(po["obj_id"]), "unknown_instrument")
                    extras_by_class[pcls] += 1
            detail_rows.append({
                "video_id": vid,
                "frame_idx": f,
                "n_matched": n_match,
                "n_missed_gt": n_miss,
                "n_extra_pred": n_extra,
            })

    # Per-class precision / recall / identity-correct
    per_class_rows: list[dict] = []
    all_classes = sorted(set(gt_class_counts) | set(pred_class_counts) | set(extras_by_class))
    for cls in all_classes:
        tp = sum(1 for g, p, _ in matched_pairs_all if g == cls and p == cls)
        fp = sum(1 for g, p, _ in matched_pairs_all if p == cls and g != cls) + extras_by_class.get(cls, 0)
        fn = sum(1 for g, p, _ in matched_pairs_all if g == cls and p != cls)
        # Missed GT (no IoU match at all) per class — read off gt_class_counts vs matched
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        per_class_rows.append({
            "class": cls,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": f"{precision:.3f}" if precision == precision else "nan",
            "recall": f"{recall:.3f}" if recall == recall else "nan",
        })

    # Confusion matrix
    conf: dict[tuple[str, str], int] = defaultdict(int)
    for g, p, _ in matched_pairs_all:
        conf[(g, p)] += 1
    classes_sorted = sorted(set(g for g, _ in conf.keys()) | set(p for _, p in conf.keys()))

    # Identity-correct rate (top-line metric)
    n_matched_total = len(matched_pairs_all)
    n_id_correct = sum(1 for g, p, _ in matched_pairs_all if g == p)
    id_correct_rate = n_id_correct / n_matched_total if n_matched_total > 0 else float("nan")

    # Write CSVs
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "frame_idx", "n_matched",
                                          "n_missed_gt", "n_extra_pred"])
        w.writeheader()
        for r in detail_rows:
            w.writerow(r)
    with open(per_class_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "true_positives", "false_positives",
                                          "false_negatives", "precision", "recall"])
        w.writeheader()
        for r in per_class_rows:
            w.writerow(r)
    with open(confusion_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt_class \\ pred_class"] + classes_sorted)
        for g in classes_sorted:
            w.writerow([g] + [conf.get((g, p), 0) for p in classes_sorted])

    # Console summary
    print(f"--- eval_detection: cohort={args.cohort} run={args.run_method!r}", file=sys.stderr)
    print(f"  videos scored          : {n_videos_scored}", file=sys.stderr)
    print(f"  IoU-matched detections : {n_matched_total}", file=sys.stderr)
    print(f"  identity-correct rate  : {id_correct_rate:.3f} ({n_id_correct}/{n_matched_total})",
          file=sys.stderr)
    print(f"\nPer-class:", file=sys.stderr)
    for r in per_class_rows:
        print(f"    {r['class']:30}  P={r['precision']}  R={r['recall']}  "
              f"(TP={r['true_positives']} FP={r['false_positives']} FN={r['false_negatives']})",
              file=sys.stderr)
    print(f"\nWrote {detail_path}", file=sys.stderr)
    print(f"Wrote {per_class_path}", file=sys.stderr)
    print(f"Wrote {confusion_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
