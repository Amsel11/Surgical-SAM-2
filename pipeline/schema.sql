-- SurgSAM-2 experiment manifest
-- One SQLite file is the source of truth for what's been clicked, run, and evaluated.
-- Lives on bigpurple GPFS (writable by SLURM jobs). Snapshots get exported to CSV/MD for git.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- videos: one row per source video. Static after creation.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS videos (
    video_id          TEXT PRIMARY KEY,            -- e.g. DC_whip_11609423
    frames_dir        TEXT NOT NULL,               -- /gpfs/.../frames_attempt2/<video_id>
    n_frames          INTEGER,                     -- populated by `pipeline scan`
    fps               REAL    DEFAULT 1.0,         -- extraction rate
    raw_video_path    TEXT,                        -- original .mp4 if known
    cohort            TEXT    DEFAULT 'whip',      -- 'whip', 'endovis18', etc.
    notes             TEXT,
    added_at          TEXT    DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- instruments: the controlled vocabulary. Seed from DaVinci Xi catalog.
-- Solves per-class DICE and human cross-video labeling consistency.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS instruments (
    instrument_id     TEXT PRIMARY KEY,            -- 'monopolar_curved_scissors' (snake_case slug)
    display_name      TEXT NOT NULL,               -- 'Monopolar Curved Scissors'
    category          TEXT,                        -- 'energy' | 'grasper' | 'needle_driver' | 'retractor' | 'other'
    reference_img     TEXT,                        -- path to a catalog photo (optional)
    aliases           TEXT,                        -- JSON array of alt names, optional
    notes             TEXT
);

-- ---------------------------------------------------------------------------
-- prompt_sets: one row per (video × seed × prompt_method).
-- Represents "a set of prompts ready to be fed to ANY SAM model".
-- The UNIQUE constraint prevents accidentally re-creating the same seed twice.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_sets (
    prompt_set_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id          TEXT NOT NULL REFERENCES videos(video_id),
    seed              INTEGER NOT NULL,            -- 1..N for variance estimation
    prompt_method     TEXT NOT NULL,               -- 'manual_click' | 'yolo' | 'dino' | 'gt_box'
    prompts_path      TEXT,                        -- prompts/<video>_seed<k>_<method>.json
    n_objects         INTEGER,
    n_prompt_frames   INTEGER,                     -- 1 or 3 typically
    status            TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'ready' | 'failed'
    created_at        TEXT    DEFAULT (datetime('now')),
    created_by        TEXT,                        -- 'annelene' or 'yolov8' etc.
    notes             TEXT,
    UNIQUE(video_id, seed, prompt_method)
);

CREATE INDEX IF NOT EXISTS idx_prompt_sets_video  ON prompt_sets(video_id);
CREATE INDEX IF NOT EXISTS idx_prompt_sets_status ON prompt_sets(status);

-- ---------------------------------------------------------------------------
-- prompt_objects: per-object instrument labels inside a prompt set.
-- This is what makes per-class evaluation possible later.
-- obj_id matches the integer object id SAM2 uses internally.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_objects (
    prompt_set_id     INTEGER NOT NULL REFERENCES prompt_sets(prompt_set_id) ON DELETE CASCADE,
    obj_id            INTEGER NOT NULL,            -- 1, 2, 3, ...
    instrument_id     TEXT NOT NULL REFERENCES instruments(instrument_id),
    cutout_img        TEXT,                        -- saved RGBA cutout for the click UI reference strip
    PRIMARY KEY (prompt_set_id, obj_id)
);

-- ---------------------------------------------------------------------------
-- inference_runs: one row per (prompt_set × model). SLURM jobs UPSERT here.
-- Idempotency: UNIQUE constraint + status='done' check before running.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS inference_runs (
    run_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_set_id           INTEGER NOT NULL REFERENCES prompt_sets(prompt_set_id),
    model                   TEXT NOT NULL,        -- 'sam2_oob' | 'surgsam2' | 'surgsam2_whip' | 'sam3' | 'sam3_whip'
    checkpoint_path         TEXT,
    results_dir             TEXT,                  -- results/<vid>/seed_<k>_<model>/
    status                  TEXT NOT NULL DEFAULT 'pending', -- 'pending'|'queued'|'running'|'done'|'failed'
    slurm_job_id            TEXT,
    slurm_array_task_id     INTEGER,
    started_at              TEXT,
    finished_at             TEXT,
    wall_seconds            REAL,
    mean_mask_area_px       REAL,                  -- from log.json
    frames_with_empty_mask  INTEGER,               -- from log.json
    error_message           TEXT,
    UNIQUE(prompt_set_id, model)
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON inference_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_model  ON inference_runs(model);

-- ---------------------------------------------------------------------------
-- evaluations: DICE/IoU per (run, object). Filled in when GT exists.
-- The join to prompt_objects gives per-instrument metrics.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluations (
    run_id            INTEGER NOT NULL REFERENCES inference_runs(run_id),
    obj_id            INTEGER NOT NULL,
    dice              REAL,
    iou               REAL,
    n_gt_frames       INTEGER,                     -- how many GT frames available for this obj
    evaluated_at      TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, obj_id)
);

-- ---------------------------------------------------------------------------
-- Views: convenience for the common "what needs doing" questions.
-- ---------------------------------------------------------------------------

-- All (video, seed, method, model) combos that should exist, with status.
-- A planned ablation = INSERT rows here with status='pending'.
CREATE VIEW IF NOT EXISTS v_run_status AS
SELECT
    v.video_id,
    ps.seed,
    ps.prompt_method,
    r.model,
    r.status         AS run_status,
    ps.status        AS prompt_status,
    r.results_dir,
    r.wall_seconds,
    r.frames_with_empty_mask
FROM prompt_sets ps
JOIN videos v ON v.video_id = ps.video_id
LEFT JOIN inference_runs r ON r.prompt_set_id = ps.prompt_set_id;

-- Per-video progress summary.
CREATE VIEW IF NOT EXISTS v_video_progress AS
SELECT
    v.video_id,
    COUNT(DISTINCT ps.prompt_set_id)                           AS n_prompt_sets,
    SUM(CASE WHEN r.status='done'   THEN 1 ELSE 0 END)         AS n_runs_done,
    SUM(CASE WHEN r.status='failed' THEN 1 ELSE 0 END)         AS n_runs_failed,
    SUM(CASE WHEN r.status IN ('pending','queued','running')
                                    THEN 1 ELSE 0 END)         AS n_runs_pending
FROM videos v
LEFT JOIN prompt_sets ps    ON ps.video_id = v.video_id
LEFT JOIN inference_runs r  ON r.prompt_set_id = ps.prompt_set_id
GROUP BY v.video_id;
