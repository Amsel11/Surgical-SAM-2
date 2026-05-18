"""Tier A QC inventory + plots for one seed of inference results.

Reads log.json + the clicker prompts JSON for every video that has a
results/<vid>/seed_<N>/log.json available locally. Writes:
  - qc_tier_a.csv         per-(video, obj_id) row with metrics + status
  - qc_summary.md         human-readable summary
  - status_counts.png     bar chart of pass / review / fail
  - empty_rate_by_anc.png box plot showing the anchor-count effect
  - per_video_status.png  stacked bar — each video's pass/review/fail mix
  - empty_vs_length.png   scatter — does empty rate scale with video length?

Usage:
  python tools/qc_tier_a.py --seed 1 \\
        --previews-dir local_results/previews \\
        --prompts-dir prompts \\
        --db local_manifest.db \\
        --out-dir local_results/qc/seed_1
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_instruments(db_path: Path, seed: int) -> dict[tuple[str, int], str]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    out = {}
    for r in con.execute(
        """
        SELECT ps.video_id, po.obj_id, po.instrument_id
        FROM prompt_objects po
        JOIN prompt_sets ps ON ps.prompt_set_id = po.prompt_set_id
        WHERE ps.seed = ?
        """,
        (seed,),
    ):
        out[(r["video_id"], r["obj_id"])] = r["instrument_id"]
    return out


def compute_inventory(
    previews_dir: Path,
    prompts_dir: Path,
    instruments: dict,
    seed: int,
    mask_scan_csv: Path | None = None,
) -> pd.DataFrame:
    """Walk previews_dir, join with the matching seed's prompts JSON, return
    a long-form DataFrame with one row per (video, obj_id).

    If `mask_scan_csv` is provided (output of tools/scan_masks.py), we add
    the *honest* in-anchor-span empty rate that distinguishes tracking
    failure (empty between two anchors that both had this obj) from
    ambiguous absence (empty before first anchor or after last anchor).
    Status thresholds switch to `in_span_rate` whenever available — that's
    the metric you want; the raw `empty_rate_overall` stays in the CSV
    only for comparison.

    Expected layout (seed-batch as top level):
        previews_dir/
            <video_id>/
                log.json
                preview_small.mp4
    """
    # Pre-load mask-scan rows keyed by (video_id, obj_id) for fast join.
    scan_map: dict[tuple[str, int], dict] = {}
    if mask_scan_csv and mask_scan_csv.exists():
        scan_df = pd.read_csv(mask_scan_csv)
        for _, r in scan_df.iterrows():
            scan_map[(r["video_id"], int(r["obj_id"]))] = r.to_dict()

    rows = []
    for vdir in sorted(previews_dir.iterdir()):
        if not vdir.is_dir():
            continue
        log_p = vdir / "log.json"
        if not log_p.exists():
            continue
        log = json.loads(log_p.read_text())
        vid = vdir.name
        n_frames = log.get("n_frames", 0)
        res = log.get("resolution", [0, 0])
        frame_area = res[0] * res[1] if res else 0
        mean_area = log.get("mean_mask_area_px", {})
        empty_counts = log.get("frames_with_empty_mask", {})

        # Anchor count per obj from the prompts JSON.
        pj_path = prompts_dir / f"{vid}_seed{seed}_manual_box.json"
        obj_anchors: dict[int, set[int]] = defaultdict(set)
        if pj_path.exists():
            pj = json.loads(pj_path.read_text())
            for f, objs in pj.get("objects_by_frame", {}).items():
                for o in objs:
                    obj_anchors[o["obj_id"]].add(int(f))

        obj_ids = sorted({int(k) for k in mean_area} | obj_anchors.keys())
        for obj_id in obj_ids:
            n_anch = len(obj_anchors.get(obj_id, set()))
            empty = empty_counts.get(str(obj_id), 0)
            mean = mean_area.get(str(obj_id), 0.0)
            empty_rate_overall = empty / n_frames if n_frames else 0.0
            area_frac = mean / frame_area if frame_area else 0.0

            # Pull mask-scan fields if available.
            scan = scan_map.get((vid, obj_id))
            in_span_rate = None
            empty_in_span = empty_pre_span = empty_post_span = None
            span_length = None
            if scan is not None:
                empty_in_span = int(scan.get("empty_in_span") or 0)
                empty_pre_span = int(scan.get("empty_pre_span") or 0)
                empty_post_span = int(scan.get("empty_post_span") or 0)
                span_length = int(scan.get("span_length") or 0)
                if scan.get("in_span_rate") not in (None, ""):
                    in_span_rate = float(scan["in_span_rate"])

            # Status: prefer in_span_rate when available — it isolates
            # tracking drift from legitimate absence.
            if in_span_rate is not None:
                if in_span_rate > 0.30 or n_anch < 2 or area_frac > 0.30:
                    status = "FAIL"
                elif in_span_rate > 0.10 or n_anch < 3:
                    status = "review"
                else:
                    status = "pass"
            else:
                if empty_rate_overall > 0.30 or n_anch < 2 or area_frac > 0.30:
                    status = "FAIL"
                elif empty_rate_overall > 0.10 or n_anch < 3:
                    status = "review"
                else:
                    status = "pass"

            rows.append({
                "video_id": vid,
                "obj_id": obj_id,
                "instrument_id": instruments.get((vid, obj_id), "?"),
                "n_frames": n_frames,
                "n_anchors": n_anch,
                "empty_count": empty,
                "empty_rate_overall": round(empty_rate_overall, 4),
                # mask-scan fields (None if scan not present)
                "empty_in_span": empty_in_span,
                "empty_pre_span": empty_pre_span,
                "empty_post_span": empty_post_span,
                "span_length": span_length,
                "in_span_rate": round(in_span_rate, 4) if in_span_rate is not None else None,
                "mean_mask_area_px": round(mean, 1),
                "area_fraction": round(area_frac, 4),
                "status": status,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STATUS_ORDER = ["pass", "review", "FAIL"]
STATUS_COLORS = {"pass": "#2ca02c", "review": "#ff7f0e", "FAIL": "#d62728"}


def plot_status_counts(df: pd.DataFrame, out: Path) -> None:
    counts = df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[STATUS_COLORS[s] for s in counts.index])
    ax.set_title(f"Tier A QC status — {len(df)} (video, obj) pairs")
    ax.set_ylabel("count")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, str(val),
                ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_empty_rate_by_anc(df: pd.DataFrame, out: Path) -> None:
    """Box plot of empty_rate grouped by anchor count. Proves the anc=1 problem."""
    # Prefer in_span_rate when available; otherwise empty_rate_overall.
    rate_col = "in_span_rate" if df["in_span_rate"].notna().any() else "empty_rate_overall"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = sorted(df["n_anchors"].unique())
    data = [df[df["n_anchors"] == g][rate_col].dropna().values * 100 for g in groups]
    bp = ax.boxplot(data, labels=[f"n_anchors={g}\n(n={len(d)})" for g, d in zip(groups, data)],
                     patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c72b0")
        patch.set_alpha(0.6)
    # threshold lines
    ax.axhline(10, ls="--", color="#ff7f0e", alpha=0.5, label="review threshold (10%)")
    ax.axhline(30, ls="--", color="#d62728", alpha=0.5, label="fail threshold (30%)")
    ax.set_ylabel(f"{rate_col.replace('_', ' ')} (%)")
    title_metric = "in-anchor-span empty rate" if rate_col == "in_span_rate" else "raw empty rate"
    ax.set_title(f"{title_metric} vs number of anchor frames clicked")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-2, 100)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_per_video_status(df: pd.DataFrame, out: Path) -> None:
    """Stacked bar: each video's mix of pass/review/fail across its objects."""
    pivot = (df.groupby(["video_id", "status"]).size()
               .unstack(fill_value=0)
               .reindex(columns=STATUS_ORDER, fill_value=0))
    # Order videos by FAIL count desc, then review desc
    pivot = pivot.sort_values(by=["FAIL", "review"], ascending=[False, False])
    fig, ax = plt.subplots(figsize=(11, 8))
    bottom = [0] * len(pivot)
    for status in STATUS_ORDER:
        ax.barh(pivot.index, pivot[status], left=bottom,
                color=STATUS_COLORS[status], label=status)
        bottom = [b + v for b, v in zip(bottom, pivot[status])]
    ax.set_xlabel("number of objects")
    ax.set_title("Per-video QC mix (sorted by FAIL count)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def video_level_verdict(statuses: list[str]) -> str:
    """Strict rule: a video passes iff EVERY object in it passes.
    Any FAIL drops it to FAIL; otherwise pass+review only -> review."""
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if all(s == "pass" for s in statuses):
        return "pass"
    return "review"


def plot_video_status_strict(df: pd.DataFrame, out: Path) -> None:
    """Per-video verdict under the strict rule (all objects must pass)."""
    by_video = df.groupby("video_id")["status"].apply(list)
    verdicts = by_video.apply(video_level_verdict)
    counts = verdicts.value_counts().reindex(STATUS_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[STATUS_COLORS[s] for s in counts.index])
    ax.set_title(f"Per-video verdict (strict: ALL objects must pass)\n"
                 f"{len(verdicts)} videos total")
    ax.set_ylabel("number of videos")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, str(val),
                ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return verdicts


def plot_empty_vs_length(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for status in STATUS_ORDER:
        sub = df[df["status"] == status]
        rate_col = "in_span_rate" if df["in_span_rate"].notna().any() else "empty_rate_overall"
        ax.scatter(sub["n_frames"], sub[rate_col].fillna(0) * 100,
                   c=STATUS_COLORS[status], label=status, alpha=0.7, s=40)
    ax.axhline(10, ls="--", color="#ff7f0e", alpha=0.3)
    ax.axhline(30, ls="--", color="#d62728", alpha=0.3)
    ax.set_xlabel("video length (frames)")
    ax.set_ylabel("empty-frame rate (%)")
    ax.set_title("Empty-frame rate vs video length, by status")
    ax.legend()
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_summary_md(df: pd.DataFrame, out: Path, seed: int, verdicts) -> None:
    counts = df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    vcounts = verdicts.value_counts().reindex(STATUS_ORDER, fill_value=0)
    n_videos = df["video_id"].nunique()
    n_pairs = len(df)
    pass_pct = 100 * counts["pass"] / n_pairs if n_pairs else 0
    vpass_pct = 100 * vcounts["pass"] / n_videos if n_videos else 0
    out.write_text(f"""# Tier A QC — seed {seed}

Auto-generated by `tools/qc_tier_a.py`.

## Headline (per (video, obj) pair)
- **pairs evaluated:** {n_pairs} across {n_videos} videos
- **pass:** {counts['pass']} ({pass_pct:.1f}%)
- **review:** {counts['review']}
- **FAIL:** {counts['FAIL']}

## Per-video verdict (strict: all objects must pass)
- **pass:** {vcounts['pass']} videos ({vpass_pct:.1f}%)
- **review:** {vcounts['review']}
- **FAIL:** {vcounts['FAIL']}

## Thresholds (v1)
**Per (video, obj):**
- **FAIL** if `empty_rate > 0.30` OR `n_anchors < 2` OR `area_fraction > 0.30`
- **review** if `empty_rate > 0.10` OR `n_anchors < 3`
- **pass** otherwise

**Per video (strict):**
- **pass** iff every object in that video passes
- **FAIL** if any object FAILs
- **review** otherwise (mix of pass + review)

## Source data
- log.json files from `local_results/previews/<video>/seed_{seed}/`
- prompts JSONs from `prompts/<video>_seed{seed}_manual_box.json`
- instrument labels joined from `local_manifest.db` (seed={seed} prompt_sets)

## Outputs in this directory
- `qc_tier_a.csv` — long-form table, one row per (video, obj_id)
- `qc_video_verdicts_strict.csv` — one row per video, strict verdict
- `status_counts.png` — per-(video, obj) pass/review/FAIL counts
- `video_status_strict.png` — per-video pass/review/FAIL counts (strict)
- `empty_rate_by_anc.png` — box plot showing how empty-rate scales with anchor count
- `per_video_status_per_obj.png` — stacked bar per video (bar width = # objects)
- `empty_vs_length.png` — scatter empty-rate vs video length
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--previews-dir", type=Path, default=None,
                    help="default: local_results/seed_<N>/previews")
    ap.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    ap.add_argument("--db", type=Path, default=Path("local_manifest.db"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: local_results/seed_<N>/qc")
    ap.add_argument("--mask-scan-csv", type=Path, default=None,
                    help="Output of tools/scan_masks.py for this seed. When "
                         "given, status uses in_span_rate (much more honest). "
                         "default: <out-dir>/mask_scan.csv if present.")
    args = ap.parse_args(argv)

    seed_root = Path("local_results") / f"seed_{args.seed}"
    previews_dir = args.previews_dir or (seed_root / "previews")
    out_dir = args.out_dir or (seed_root / "qc")
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_scan_csv = args.mask_scan_csv or (out_dir / "mask_scan.csv")

    instruments = load_instruments(args.db, args.seed)
    df = compute_inventory(previews_dir, args.prompts_dir, instruments,
                           args.seed, mask_scan_csv if mask_scan_csv.exists() else None)
    if df.empty:
        print(f"No results for seed={args.seed} under {previews_dir}. "
              "Run inference first.")
        return 1

    csv_path = out_dir / "qc_tier_a.csv"
    df.to_csv(csv_path, index=False)

    plot_status_counts(df, out_dir / "status_counts.png")
    plot_empty_rate_by_anc(df, out_dir / "empty_rate_by_anc.png")
    plot_per_video_status(df, out_dir / "per_video_status_per_obj.png")
    plot_empty_vs_length(df, out_dir / "empty_vs_length.png")
    verdicts = plot_video_status_strict(df, out_dir / "video_status_strict.png")

    # Save the per-video verdict to CSV too — useful for downstream filtering.
    verdicts.to_frame(name="video_status").reset_index().to_csv(
        out_dir / "qc_video_verdicts_strict.csv", index=False
    )
    write_summary_md(df, out_dir / "qc_summary.md", args.seed, verdicts)

    counts = df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    metric = "in_span_rate" if df["in_span_rate"].notna().any() else "empty_rate_overall (no mask scan)"
    print(f"Tier A QC for seed={args.seed}: {len(df)} pairs across "
          f"{df['video_id'].nunique()} videos  [metric: {metric}]")
    print(f"  pass:   {counts['pass']}")
    print(f"  review: {counts['review']}")
    print(f"  FAIL:   {counts['FAIL']}")
    print(f"  Outputs -> {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
