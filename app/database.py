import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import get_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(get_settings().database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    get_settings().data_dir.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, url TEXT NOT NULL, status TEXT NOT NULL,
                step TEXT NOT NULL, progress_message TEXT NOT NULL, error TEXT,
                provider TEXT NOT NULL, asr_model TEXT NOT NULL,
                source_language_code TEXT NOT NULL, source_language TEXT NOT NULL,
                target_language TEXT NOT NULL, glossary TEXT,
                burn_subtitles INTEGER NOT NULL, dub INTEGER NOT NULL,
                tts_voice TEXT NOT NULL, original_audio_volume REAL NOT NULL,
                received_bytes INTEGER, sha256 TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)").fetchall()}
        if "subtitle_source" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN subtitle_source TEXT NOT NULL DEFAULT 'asr'")
        if "diarize" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN diarize INTEGER NOT NULL DEFAULT 0")
        if "min_speakers" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN min_speakers INTEGER")
        if "max_speakers" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN max_speakers INTEGER")
        if "tts_secondary_voice" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN tts_secondary_voice TEXT NOT NULL DEFAULT 'vi-VN-NamMinhNeural'")
        if "voice_overrides" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN voice_overrides TEXT NOT NULL DEFAULT '{}'")
        if "narrator_mode" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN narrator_mode INTEGER NOT NULL DEFAULT 1")
        if "hide_source_subtitles" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN hide_source_subtitles INTEGER NOT NULL DEFAULT 1")
        if "received_bytes" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN received_bytes INTEGER")
        if "sha256" not in columns:
            db.execute("ALTER TABLE jobs ADD COLUMN sha256 TEXT")
        db.execute(
            """CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, progress_message TEXT NOT NULL,
                error TEXT, burn_subtitles INTEGER NOT NULL, dub INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS batch_jobs (
                batch_id TEXT NOT NULL, job_id TEXT NOT NULL UNIQUE,
                position INTEGER NOT NULL, filename TEXT NOT NULL,
                PRIMARY KEY (batch_id, position),
                FOREIGN KEY (batch_id) REFERENCES batches(id),
                FOREIGN KEY (job_id) REFERENCES jobs(id)
            )"""
        )
        batch_columns = {row["name"] for row in db.execute("PRAGMA table_info(batches)").fetchall()}
        if "channel_name" not in batch_columns:
            db.execute("ALTER TABLE batches ADD COLUMN channel_name TEXT NOT NULL DEFAULT ''")
        if "watermark_opacity" not in batch_columns:
            db.execute("ALTER TABLE batches ADD COLUMN watermark_opacity REAL NOT NULL DEFAULT 0.58")
        if "output_filename" not in batch_columns:
            db.execute("ALTER TABLE batches ADD COLUMN output_filename TEXT NOT NULL DEFAULT ''")
        if "expected_episodes" not in batch_columns:
            db.execute("ALTER TABLE batches ADD COLUMN expected_episodes INTEGER NOT NULL DEFAULT 0")
        if "download_confirmed_at" not in batch_columns:
            db.execute("ALTER TABLE batches ADD COLUMN download_confirmed_at TEXT")
        if "download_confirmed_bytes" not in batch_columns:
            db.execute("ALTER TABLE batches ADD COLUMN download_confirmed_bytes INTEGER")


def create_job(values: dict[str, Any]) -> dict[str, Any]:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    with connect() as db:
        db.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", tuple(values.values()))
    return get_job(values["id"])  # type: ignore[return-value]


def get_job(job_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def update_job(job_id: str, **values: Any) -> None:
    values["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in values)
    with connect() as db:
        db.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*values.values(), job_id))


def delete_job(job_id: str) -> None:
    with connect() as db:
        db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def create_batch(values: dict[str, Any]) -> dict[str, Any]:
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    with connect() as db:
        db.execute(f"INSERT INTO batches ({columns}) VALUES ({placeholders})", tuple(values.values()))
    return get_batch(values["id"])  # type: ignore[return-value]


def add_batch_job(batch_id: str, job_id: str, position: int, filename: str) -> None:
    with connect() as db:
        db.execute(
            "INSERT INTO batch_jobs (batch_id, job_id, position, filename) VALUES (?, ?, ?, ?)",
            (batch_id, job_id, position, filename),
        )


def get_batch(batch_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    return dict(row) if row else None


def get_batch_jobs(batch_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """SELECT bj.batch_id, bj.position, bj.filename, j.* FROM batch_jobs bj
               JOIN jobs j ON j.id = bj.job_id WHERE bj.batch_id = ? ORDER BY bj.position""",
            (batch_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def remove_batch_job(batch_id: str, position: int) -> dict[str, Any] | None:
    """Detach one episode and delete its job row atomically."""
    with connect() as db:
        row = db.execute(
            "SELECT job_id FROM batch_jobs WHERE batch_id = ? AND position = ?",
            (batch_id, position),
        ).fetchone()
        if not row:
            return None
        job_id = str(row["job_id"])
        db.execute("DELETE FROM batch_jobs WHERE batch_id = ? AND position = ?", (batch_id, position))
        db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return {"id": job_id}


def list_batches_by_status(statuses: tuple[str, ...]) -> list[dict[str, Any]]:
    if not statuses:
        return []
    placeholders = ",".join("?" for _ in statuses)
    with connect() as db:
        rows = db.execute(
            f"SELECT * FROM batches WHERE status IN ({placeholders}) ORDER BY created_at", statuses,
        ).fetchall()
    return [dict(row) for row in rows]


def get_job_batch(job_id: str) -> str | None:
    with connect() as db:
        row = db.execute("SELECT batch_id FROM batch_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return str(row["batch_id"]) if row else None


def update_batch(batch_id: str, **values: Any) -> None:
    values["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in values)
    with connect() as db:
        db.execute(f"UPDATE batches SET {assignments} WHERE id = ?", (*values.values(), batch_id))


def recover_interrupted_jobs() -> None:
    with connect() as db:
        now = utc_now()
        db.execute(
            "UPDATE jobs SET status='queued', step='queued', progress_message=?, error=NULL, updated_at=? "
            "WHERE status='running'",
            ("Server restarted; job queued again", now),
        )
