"""Derive the empirical cardiac-whip vocabulary from labelled prompt_objects.

Phase 0c of the GD fine-tune plan. Runs after the Phase 0b relabel pass
(clicker --edit). For every instrument_id appearing on >= MIN_VIDEOS whip
videos under (seed, prompt_method), emits a canonical-name entry in
configs/cardiac_whip_vocab.json.

The output feeds:
  - pipeline.prompts.dino       : text queries to Grounding DINO
  - tools.ft_dataset_convert    : class label set for Open-GroundingDino FT
  - tools.eval_detection        : the universe of classes for AP / ID confusion

Frequency threshold (default 2) drops one-off mis-clicks without requiring
the operator to delete them from the manifest. ui_variants start empty;
populate by hand once the zero-shot dino baseline is in and we see which
phrasings GD's text encoder responds to.

Usage:
    python -m tools.build_vocab                                       # defaults: whip, seed=1, manual_box
    python -m tools.build_vocab --cohort whip --min-videos 2
    python -m tools.build_vocab --dry-run                             # print payload, don't write

On bp:
    ssh bp 'cd /gpfs/data/oermannlab/users/schula12/Surgical-SAM-2 \\
            && source .sam3_venv/bin/activate \\
            && python -m tools.build_vocab'
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from pipeline.db import REPO_ROOT, connect

INSTRUMENTS_JSON = REPO_ROOT / "pipeline" / "instruments.json"
DEFAULT_OUT = REPO_ROOT / "configs" / "cardiac_whip_vocab.json"
DEFAULT_MIN_VIDEOS = 2

VOCAB_SQL = """
SELECT po.instrument_id, v.video_id
FROM prompt_objects po
JOIN prompt_sets ps ON ps.prompt_set_id = po.prompt_set_id
JOIN videos v       ON v.video_id        = ps.video_id
WHERE v.cohort           = ?
  AND ps.status          = 'ready'
  AND ps.seed            = ?
  AND ps.prompt_method   = ?
  AND po.instrument_id  != 'unknown_instrument'
"""


def load_instrument_metadata(path: Path = INSTRUMENTS_JSON) -> dict[str, dict]:
    """Return {instrument_id: {display_name, category, ...}} from instruments.json."""
    data = json.loads(Path(path).read_text())
    return {e["instrument_id"]: e for e in data["instruments"]}


def collect_per_video(
    conn, cohort: str, seed: int, prompt_method: str
) -> dict[str, set[str]]:
    """Return {instrument_id: set(video_id)} for non-unknown labelled instruments."""
    per_inst: dict[str, set[str]] = defaultdict(set)
    for row in conn.execute(VOCAB_SQL, (cohort, seed, prompt_method)):
        per_inst[row["instrument_id"]].add(row["video_id"])
    return dict(per_inst)


def build_vocab(
    per_video: dict[str, set[str]],
    instrument_meta: dict[str, dict],
    min_videos: int,
) -> dict:
    """Apply the frequency threshold; assemble the vocab JSON payload.

    Sort surviving entries by descending video count, then instrument_id
    alphabetical — gives a deterministic, scannable file.
    """
    surviving: list[tuple[str, int]] = sorted(
        ((iid, len(vids)) for iid, vids in per_video.items() if len(vids) >= min_videos),
        key=lambda t: (-t[1], t[0]),
    )

    entries: list[dict] = []
    missing: list[str] = []
    for iid, _ in surviving:
        meta = instrument_meta.get(iid)
        if meta is None:
            missing.append(iid)
            continue
        entries.append({
            "instrument_id": iid,
            "canonical": meta["display_name"],
            "ui_variants": [],
        })

    all_source_videos: set[str] = set()
    for iid, _ in surviving:
        all_source_videos.update(per_video.get(iid, set()))

    return {
        "instruments": entries,
        "_meta": {
            "n_classes": len(entries),
            "source_videos": len(all_source_videos),
            "frequency_threshold": min_videos,
            "dropped_below_threshold": sorted(
                iid for iid, vids in per_video.items() if len(vids) < min_videos
            ),
            "missing_from_instruments_json": missing,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", help="Manifest path (default: $SURGSAM_MANIFEST or repo/manifest.db)")
    p.add_argument("--cohort", default="whip")
    p.add_argument("--seed", type=int, default=1,
                   help="prompt_sets.seed to read labels from (default 1)")
    p.add_argument("--method", default="manual_box",
                   help="prompt_sets.prompt_method to read from (default 'manual_box')")
    p.add_argument("--min-videos", type=int, default=DEFAULT_MIN_VIDEOS,
                   help=f"Minimum supporting videos per class (default {DEFAULT_MIN_VIDEOS})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output vocab JSON (default: configs/cardiac_whip_vocab.json)")
    p.add_argument("--instruments-json", type=Path, default=INSTRUMENTS_JSON,
                   help="Canonical instrument names (default: pipeline/instruments.json)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the payload to stdout instead of writing.")
    args = p.parse_args(argv)

    instrument_meta = load_instrument_metadata(args.instruments_json)
    conn = connect(args.db)
    per_video = collect_per_video(conn, args.cohort, args.seed, args.method)
    if not per_video:
        print(
            f"No non-unknown labelled prompt_objects for cohort={args.cohort!r} "
            f"seed={args.seed} method={args.method!r}. "
            "Did you run the Phase 0b relabel pass? Try: "
            f"python -m tools.audit_labels --cohort {args.cohort}",
            file=sys.stderr,
        )
        return 2

    payload = build_vocab(per_video, instrument_meta, args.min_videos)

    print(f"--- build_vocab: cohort={args.cohort} seed={args.seed} method={args.method}",
          file=sys.stderr)
    print(f"  candidates           : {len(per_video)}", file=sys.stderr)
    print(f"  surviving threshold  : {payload['_meta']['n_classes']} "
          f"(>= {args.min_videos} videos)", file=sys.stderr)
    print(f"  source videos        : {payload['_meta']['source_videos']}", file=sys.stderr)
    if payload['_meta']['dropped_below_threshold']:
        print(f"  dropped below thresh : {payload['_meta']['dropped_below_threshold']}",
              file=sys.stderr)
    if payload['_meta']['missing_from_instruments_json']:
        print(
            "  ⚠ instrument_ids NOT IN instruments.json (skipped from vocab): "
            + str(payload['_meta']['missing_from_instruments_json']),
            file=sys.stderr,
        )
    for e in payload["instruments"]:
        n_vids = len(per_video[e["instrument_id"]])
        print(f"    {e['instrument_id']:35} {n_vids:3} videos  ({e['canonical']})",
              file=sys.stderr)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
