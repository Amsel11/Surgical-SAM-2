"""Render an OCR arm-slot timeline (segments.csv) as a Gantt-style chart.

Four horizontal lanes (arm1..arm4); each segment draws a colored bar in every
arm that holds an instrument, spanning its [start_sec, end_sec]. Color encodes
the instrument; the camera arm is gray. Gives an at-a-glance view of which
instrument is on which arm over the whole procedure, and where swaps happen.

Usage:
  python tools/plot_ocr_timeline.py --in-csv results/.../DC.../segments.csv --out-png dc.png
  python tools/plot_ocr_timeline.py --in-dir results/ocr_slots_timeline --out-dir plots/
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

ARMS = (1, 2, 3, 4)
_PALETTE = plt.get_cmap("tab10").colors


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "start_sec": float(r["start_sec"]) if r["start_sec"] else 0.0,
                "end_sec":   float(r["end_sec"]) if r["end_sec"] else 0.0,
                "arms": {a: (r.get(f"arm{a}") or "").strip() for a in ARMS},
            })
    return rows


def _color_map(rows: list[dict]) -> dict[str, tuple]:
    names = []
    for r in rows:
        for a in ARMS:
            v = r["arms"][a]
            if v and v not in names:
                names.append(v)
    cmap: dict[str, tuple] = {}
    ci = 0
    for n in names:
        if n.lower() == "camera":
            cmap[n] = (0.6, 0.6, 0.6)
        else:
            cmap[n] = _PALETTE[ci % len(_PALETTE)]
            ci += 1
    return cmap


def plot_file(in_csv: Path, out_png: Path, video_id: str | None = None) -> None:
    rows = load(in_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmap = _color_map(rows)
    t_max = max((r["end_sec"] for r in rows), default=1.0)

    fig, ax = plt.subplots(figsize=(14, 3.2))
    for r in rows:
        x0, w = r["start_sec"], max(r["end_sec"] - r["start_sec"], t_max * 0.001)
        for a in ARMS:
            v = r["arms"][a]
            if not v:
                continue
            y = 4 - a  # arm1 on top
            ax.broken_barh([(x0, w)], (y + 0.1, 0.8),
                           facecolors=cmap[v], edgecolors="white", linewidth=0.4)

    ax.set_yticks([3.5, 2.5, 1.5, 0.5])
    ax.set_yticklabels(["arm1", "arm2", "arm3", "arm4"])
    ax.set_ylim(0, 4)
    ax.set_xlim(0, t_max * 1.02)
    ax.set_xlabel("time (s)")
    ax.set_title(video_id or in_csv.parent.name)
    ax.grid(axis="x", alpha=0.3)
    handles = [mpatches.Patch(color=c, label=n) for n, c in cmap.items()]
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
                  ncol=min(len(handles), 4), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}  ({len(rows)} segments)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-csv", type=Path)
    ap.add_argument("--out-png", type=Path)
    ap.add_argument("--in-dir", type=Path, help="Root of <video>/segments.csv")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.in_csv:
        out = args.out_png or args.in_csv.with_suffix(".png")
        plot_file(args.in_csv, out, args.in_csv.parent.name)
        return 0
    if args.in_dir:
        for d in sorted(args.in_dir.iterdir()):
            seg = d / "segments.csv"
            if seg.exists():
                plot_file(seg, args.out_dir / f"{d.name}.png", d.name)
        return 0
    ap.error("provide --in-csv/--out-png or --in-dir/--out-dir")


if __name__ == "__main__":
    raise SystemExit(main())
