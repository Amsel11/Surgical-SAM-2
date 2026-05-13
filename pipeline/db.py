"""SQLite connection helpers for the SurgSAM-2 manifest.

Why WAL mode: SLURM array tasks finish and write to inference_runs concurrently.
WAL (Write-Ahead Logging) lets one writer and many readers coexist without
blocking. Default rollback-journal mode would serialize everything and
occasionally deadlock under burst writes.

Why foreign_keys=ON: SQLite does NOT enforce FK constraints by default — you
have to opt in per-connection. Catches bugs like inserting an inference_run
that references a non-existent prompt_set.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# Override with $SURGSAM_MANIFEST=/path/to/manifest.db for testing.
DEFAULT_DB_PATH = Path(os.environ.get("SURGSAM_MANIFEST", REPO_ROOT / "manifest.db"))


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the manifest. Creates schema on first call. Safe to call repeatedly."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    first_time = not path.exists()

    # timeout=30: if another writer holds the lock, wait up to 30s before raising.
    # SLURM jobs occasionally collide; this avoids spurious failures.
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")  # WAL+NORMAL is the standard safe combo

    if first_time:
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())

    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Explicit transaction. Use for multi-statement writes that must be atomic."""
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def db_path() -> Path:
    return DEFAULT_DB_PATH
