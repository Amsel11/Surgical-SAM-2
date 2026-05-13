"""SurgSAM-2 manifest CLI.

Usage examples:
    python -m pipeline.cli init                         # create DB + seed vocab
    python -m pipeline.cli status                       # progress summary
    python -m pipeline.cli list-pending                 # what still needs running
    python -m pipeline.cli scan-videos /gpfs/.../frames # register video frames dirs

Why a module CLI (`python -m pipeline.cli`) instead of a script: import path is
deterministic regardless of cwd, and we can later add `entry_points` in
pyproject.toml to expose it as plain `pipeline status`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .db import REPO_ROOT, connect, db_path, transaction


def cmd_init(args: argparse.Namespace) -> int:
    """Create the DB (idempotent) and seed the instrument vocabulary."""
    conn = connect(args.db)
    print(f"DB initialized at {db_path() if not args.db else args.db}")

    vocab_path = REPO_ROOT / "pipeline" / "instruments.json"
    with open(vocab_path) as f:
        vocab = json.load(f)

    with transaction(conn):
        for inst in vocab["instruments"]:
            conn.execute(
                """
                INSERT INTO instruments (instrument_id, display_name, category, notes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(instrument_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    category     = excluded.category,
                    notes        = excluded.notes
                """,
                (inst["instrument_id"], inst["display_name"], inst.get("category"), inst.get("notes")),
            )
    n = conn.execute("SELECT COUNT(*) FROM instruments").fetchone()[0]
    print(f"Seeded {n} instruments from {vocab_path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print top-level progress summary."""
    conn = connect(args.db)

    counts = {
        "videos":           conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
        "instruments":      conn.execute("SELECT COUNT(*) FROM instruments").fetchone()[0],
        "prompt_sets":      conn.execute("SELECT COUNT(*) FROM prompt_sets").fetchone()[0],
        "runs_pending":     conn.execute("SELECT COUNT(*) FROM inference_runs WHERE status='pending'").fetchone()[0],
        "runs_queued":      conn.execute("SELECT COUNT(*) FROM inference_runs WHERE status='queued'").fetchone()[0],
        "runs_running":     conn.execute("SELECT COUNT(*) FROM inference_runs WHERE status='running'").fetchone()[0],
        "runs_done":        conn.execute("SELECT COUNT(*) FROM inference_runs WHERE status='done'").fetchone()[0],
        "runs_failed":      conn.execute("SELECT COUNT(*) FROM inference_runs WHERE status='failed'").fetchone()[0],
        "evaluations":      conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0],
    }
    width = max(len(k) for k in counts)
    print(f"manifest: {db_path() if not args.db else args.db}")
    print("-" * 40)
    for k, v in counts.items():
        print(f"  {k:<{width}}  {v}")
    return 0


def cmd_list_pending(args: argparse.Namespace) -> int:
    """List runs that are not done. Optionally filter by model."""
    conn = connect(args.db)
    sql = """
        SELECT v.video_id, ps.seed, ps.prompt_method, r.model, r.status
        FROM inference_runs r
        JOIN prompt_sets ps ON ps.prompt_set_id = r.prompt_set_id
        JOIN videos v       ON v.video_id = ps.video_id
        WHERE r.status != 'done'
    """
    params: tuple = ()
    if args.model:
        sql += " AND r.model = ?"
        params = (args.model,)
    sql += " ORDER BY v.video_id, ps.seed, ps.prompt_method, r.model"
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("No pending runs.")
        return 0
    print(f"{'video_id':<32} {'seed':<5} {'method':<14} {'model':<16} status")
    for r in rows:
        print(f"{r['video_id']:<32} {r['seed']:<5} {r['prompt_method']:<14} {r['model']:<16} {r['status']}")
    print(f"\nTotal: {len(rows)} pending")
    return 0


def cmd_ingest_existing(args: argparse.Namespace) -> int:
    """Backfill prompts/*.json and results/<vid>/seed_*/log.json into the manifest.

    Legacy data shape: every existing prompt set is treated as manual_click+seed=1,
    every existing result as model=surgsam2. No instrument labels exist yet, so
    every obj_id gets 'unknown_instrument'. This is a one-shot bootstrap; new runs
    go through the proper flow.
    """
    prompts_dir = Path(args.prompts_dir)
    results_dir = Path(args.results_dir)
    frames_base = Path(args.frames_base)
    conn = connect(args.db)

    inserted_videos = 0
    inserted_prompt_sets = 0
    inserted_runs = 0
    inserted_objects = 0

    with transaction(conn):
        for pjson in sorted(prompts_dir.glob("*.json")):
            with open(pjson) as f:
                data = json.load(f)
            video_id = data["video"]
            n_frames = data.get("n_frames")

            frames_dir = frames_base / video_id
            cur = conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, frames_dir, n_frames, cohort) VALUES (?, ?, ?, 'whip')",
                (video_id, str(frames_dir), n_frames),
            )
            inserted_videos += cur.rowcount

            obj_ids = set()
            for frame_objs in data.get("objects_by_frame", {}).values():
                for o in frame_objs:
                    obj_ids.add(o["obj_id"])

            cur = conn.execute(
                """
                INSERT OR IGNORE INTO prompt_sets
                    (video_id, seed, prompt_method, prompts_path, n_objects, n_prompt_frames, status, created_by)
                VALUES (?, 1, 'manual_click', ?, ?, ?, 'ready', 'legacy-ingest')
                """,
                (video_id, str(pjson), len(obj_ids), len(data.get("prompt_frames", []))),
            )
            if cur.rowcount:
                inserted_prompt_sets += 1
                ps_id = cur.lastrowid
            else:
                ps_id = conn.execute(
                    "SELECT prompt_set_id FROM prompt_sets WHERE video_id=? AND seed=1 AND prompt_method='manual_click'",
                    (video_id,),
                ).fetchone()["prompt_set_id"]

            for obj_id in sorted(obj_ids):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO prompt_objects (prompt_set_id, obj_id, instrument_id) VALUES (?, ?, 'unknown_instrument')",
                    (ps_id, obj_id),
                )
                inserted_objects += cur.rowcount

            log_path = results_dir / video_id / "seed_1" / "log.json"
            if log_path.exists():
                with open(log_path) as f:
                    log = json.load(f)
                mean_area = log.get("mean_mask_area_px") or {}
                avg_area = sum(mean_area.values()) / len(mean_area) if mean_area else None
                empty_count = len(log.get("frames_with_empty_mask") or {})

                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO inference_runs
                        (prompt_set_id, model, checkpoint_path, results_dir, status,
                         finished_at, wall_seconds, mean_mask_area_px, frames_with_empty_mask)
                    VALUES (?, 'surgsam2', ?, ?, 'done', ?, ?, ?, ?)
                    """,
                    (
                        ps_id,
                        log.get("checkpoint"),
                        str(log_path.parent),
                        log.get("finished_at"),
                        log.get("wall_seconds"),
                        avg_area,
                        empty_count,
                    ),
                )
                inserted_runs += cur.rowcount

    print(f"Ingested:")
    print(f"  videos        +{inserted_videos}")
    print(f"  prompt_sets   +{inserted_prompt_sets}")
    print(f"  prompt_objs   +{inserted_objects}")
    print(f"  inference_runs +{inserted_runs}")
    return 0


def cmd_scan_videos(args: argparse.Namespace) -> int:
    """Walk a frames root and register video subdirs.

    Each immediate subdir of `path` is treated as one video, with name = video_id.
    Skips dirs matching --skip glob patterns or with fewer than --min-frames frames.
    """
    import fnmatch

    root = Path(args.path)
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1
    skip_patterns = [s.strip() for s in (args.skip or "").split(",") if s.strip()]

    conn = connect(args.db)
    added = updated = skipped = 0
    with transaction(conn):
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            video_id = child.name
            if any(fnmatch.fnmatch(video_id, pat) for pat in skip_patterns):
                print(f"  skip {video_id} (matches --skip)")
                skipped += 1
                continue
            n_frames = sum(1 for _ in child.glob("*.png")) + sum(1 for _ in child.glob("*.jpg"))
            if n_frames < args.min_frames:
                print(f"  skip {video_id} ({n_frames} frames < --min-frames {args.min_frames})")
                skipped += 1
                continue
            cur = conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, frames_dir, n_frames, cohort) VALUES (?, ?, ?, ?)",
                (video_id, str(child), n_frames, args.cohort),
            )
            if cur.rowcount:
                added += 1
            else:
                conn.execute(
                    "UPDATE videos SET n_frames = ?, frames_dir = ? WHERE video_id = ?",
                    (n_frames, str(child), video_id),
                )
                updated += 1
    print(f"Scanned {root}: {added} new, {updated} updated, {skipped} skipped")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipeline", description="SurgSAM-2 manifest CLI")
    p.add_argument("--db", help="Override manifest path (default: $SURGSAM_MANIFEST or repo/manifest.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Create DB and seed instrument vocab")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("status", help="Top-level progress summary")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("list-pending", help="List runs not yet done")
    sp.add_argument("--model", help="Filter to one model")
    sp.set_defaults(func=cmd_list_pending)

    sp = sub.add_parser("ingest-existing", help="Backfill legacy prompts/*.json + results into manifest")
    sp.add_argument("--prompts-dir", required=True, help="Dir containing prompts/<video>.json")
    sp.add_argument("--results-dir", required=True, help="Dir containing results/<video>/seed_*/log.json")
    sp.add_argument("--frames-base", required=True, help="Base path where frames live (per-video subdirs)")
    sp.set_defaults(func=cmd_ingest_existing)

    sp = sub.add_parser("scan-videos", help="Register frames-dir subdirs as videos")
    sp.add_argument("path", help="Frames root, e.g. /gpfs/.../frames_attempt2")
    sp.add_argument("--cohort", default="whip")
    sp.add_argument("--skip", help="Comma-separated glob patterns to exclude (e.g. 'JD_whip_13838502,480fullPJ4x_*')")
    sp.add_argument("--min-frames", type=int, default=10, help="Skip dirs with fewer than this many frames")
    sp.set_defaults(func=cmd_scan_videos)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
