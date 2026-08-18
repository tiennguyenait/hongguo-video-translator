"""Folder-batch orchestration and final episode concatenation."""

from __future__ import annotations

import re
import logging
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

from . import media
from .config import get_settings
from .database import (
    add_batch_job, create_batch, get_batch, get_batch_jobs, get_job_batch,
    list_batches_by_status, update_batch, utc_now,
)

logger = logging.getLogger(__name__)
_combine_lock = threading.Lock()
_active_combines: set[str] = set()


class InsufficientStorageError(RuntimeError):
    pass


def storage_status() -> dict[str, int | bool]:
    settings = get_settings()
    usage = shutil.disk_usage(settings.data_dir)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "warning": usage.free < settings.storage_warning_bytes,
        "accepting_uploads": usage.free >= settings.storage_reserve_bytes,
    }


def require_upload_capacity(additional_bytes: int = 0) -> None:
    settings = get_settings()
    free = int(storage_status()["free_bytes"])
    required = settings.storage_reserve_bytes + max(0, additional_bytes)
    if free < required:
        raise InsufficientStorageError(
            f"Insufficient server storage: {free / (1024**3):.1f} GiB free; "
            f"at least {required / (1024**3):.1f} GiB is required"
        )


def estimated_combine_bytes(inputs: list[Path]) -> int:
    return round(sum(path.stat().st_size for path in inputs) * get_settings().combine_size_multiplier)


def natural_filename_key(filename: str) -> list[tuple[int, Any]]:
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", Path(filename).name)
    ]


def create_folder_batch(
    burn_subtitles: bool, dub: bool, channel_name: str = "", watermark_opacity: float = 0.58,
    expected_episodes: int = 0,
) -> dict[str, Any]:
    batch_id, now = str(uuid.uuid4()), utc_now()
    batch_directory(batch_id).mkdir(parents=True)
    return create_batch({
        "id": batch_id, "status": "uploading", "progress_message": "Uploading folder videos",
        "error": None, "burn_subtitles": int(burn_subtitles), "dub": int(dub),
        "channel_name": channel_name.strip(), "watermark_opacity": watermark_opacity,
        "output_filename": "", "expected_episodes": expected_episodes,
        "created_at": now, "updated_at": now,
    })


def save_batch_logo(batch_id: str, stream: BinaryIO) -> Path:
    raw = batch_directory(batch_id) / ".logo.uploading"
    total = 0
    with raw.open("wb") as target:
        while chunk := stream.read(256 * 1024):
            total += len(chunk)
            if total > 5 * 1024 * 1024:
                raw.unlink(missing_ok=True)
                raise ValueError("Channel logo exceeds the 5 MiB limit")
            target.write(chunk)
    if total < 64:
        raw.unlink(missing_ok=True)
        raise ValueError("Channel logo is empty")
    output = batch_directory(batch_id) / "logo.png"
    try:
        with Image.open(raw) as image:
            image.verify()
        with Image.open(raw) as image:
            converted = image.convert("RGBA")
            converted.thumbnail((512, 512), Image.Resampling.LANCZOS)
            converted.save(output, format="PNG", optimize=True)
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise ValueError("Channel logo must be a valid PNG, JPG, or WEBP image") from exc
    finally:
        raw.unlink(missing_ok=True)
    return output


def attach_job(batch_id: str, job_id: str, position: int, filename: str) -> None:
    add_batch_job(batch_id, job_id, position, Path(filename).name)


def batch_directory(batch_id: str) -> Path:
    return get_settings().data_dir / "batches" / batch_id


def combined_filename(batch: dict[str, Any]) -> str:
    if batch.get("output_filename"):
        return str(batch["output_filename"])
    if batch["dub"]:
        return "combined-vi-dubbed.mp4"
    if batch["burn_subtitles"]:
        return "combined-vi-burned.mp4"
    return "combined-source.mp4"


def _unique_output_filename(batch: dict[str, Any], episodes: list[dict[str, Any]]) -> str:
    """Build a stable, human-readable name without trusting an uploaded path."""
    try:
        stamp = datetime.fromisoformat(str(batch["created_at"])).strftime("%Y%m%d-%H%M%S")
    except (KeyError, TypeError, ValueError):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if len(episodes) == 1:
        label = Path(str(episodes[0].get("filename") or "video")).stem
    elif batch["dub"]:
        label = "combined-vi-dubbed"
    elif batch["burn_subtitles"]:
        label = "combined-vi-burned"
    else:
        label = "combined-source"
    label = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", label).strip(" .-") or "video"
    return f"{stamp}_{label}.mp4"


def cleanup_combined_episode_media(episodes: list[dict[str, Any]]) -> list[str]:
    """After a verified combine, retain metadata but discard per-episode media."""
    removed: list[str] = []
    jobs_root = get_settings().jobs_dir.resolve()
    for episode in episodes:
        directory = (jobs_root / str(episode["id"])).resolve()
        if directory.parent != jobs_root or not directory.is_dir() or directory.is_symlink():
            continue
        for path in directory.iterdir():
            if path.is_symlink():
                continue
            if path.is_file() and path.suffix.lower() in {".mp4", ".wav", ".mp3", ".mkv", ".webm"}:
                path.unlink()
                removed.append(str(path.relative_to(jobs_root)))
            elif path.is_dir() and path.name == "tts":
                shutil.rmtree(path)
                removed.append(str(path.relative_to(jobs_root)) + "/")
    return removed


def finalize_batch_for_job(job_id: str) -> None:
    batch_id = get_job_batch(job_id)
    if not batch_id:
        return
    batch, episodes = get_batch(batch_id), get_batch_jobs(batch_id)
    if not batch or not episodes:
        return
    if batch["status"] in {"uploading", "done"}:
        return
    failed = [item for item in episodes if item["status"] in {"failed", "needs_review"}]
    active = [item for item in episodes if item["status"] in {"queued", "running"}]
    completed = sum(item["status"] == "done" for item in episodes)
    if active:
        detail = None
        if failed:
            first = failed[0]
            detail = f"{first['filename']}: {first.get('error') or first['status']}"
        update_batch(
            batch_id, status="running",
            progress_message=(
                f"Completed {completed}/{len(episodes)}; {len(failed)} need attention; "
                f"continuing episode processing"
            ),
            error=detail,
        )
        return
    if failed:
        first = failed[0]
        update_batch(
            batch_id, status="failed",
            progress_message=f"Episode {first['position']} failed",
            error=f"{first['filename']}: {first.get('error') or first['status']}",
        )
        return
    if completed < len(episodes):
        update_batch(
            batch_id, status="running",
            progress_message=f"Completed episode {completed}/{len(episodes)}; processing sequentially",
            error=None,
        )
        return
    with _combine_lock:
        if batch_id in _active_combines:
            return
        _active_combines.add(batch_id)
    output_name = str(batch.get("output_filename") or _unique_output_filename(batch, episodes))
    source_name = "vi-dubbed.mp4" if batch["dub"] else "vi-burned.mp4" if batch["burn_subtitles"] else "source.mp4"
    inputs = [get_settings().jobs_dir / item["id"] / source_name for item in episodes]
    directory = batch_directory(batch_id)
    output = directory / output_name
    try:
        missing = [str(path) for path in inputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"Combined inputs are missing: {missing[:3]}")
        required = estimated_combine_bytes(inputs) + get_settings().storage_reserve_bytes
        free = int(storage_status()["free_bytes"])
        if free < required:
            update_batch(
                batch_id, status="waiting_for_space", output_filename=output_name,
                progress_message=(
                    f"Waiting for disk space before combining {len(episodes)} episodes: "
                    f"{free / (1024**3):.1f} GiB free, {required / (1024**3):.1f} GiB required"
                ), error=None,
            )
            return
        update_batch(
            batch_id, status="combining", output_filename=output_name,
            progress_message=f"Combining {len(episodes)} translated episodes", error=None,
        )
        batch["output_filename"] = output_name
        logo = directory / "logo.png"
        branded = bool(batch.get("channel_name") and logo.is_file())
        media.concat_videos(
            inputs, output, logo if branded else None,
            batch.get("channel_name", ""), batch.get("watermark_opacity", 0.58),
        )
        update_batch(
            batch_id, status="done", output_filename=output.name,
            progress_message=f"Combined {len(episodes)} episodes successfully in one encode pass", error=None,
        )
        removed = cleanup_combined_episode_media(episodes)
        if removed:
            logger.info("batch=%s storage cleanup removed %d episode media paths", batch_id, len(removed))
    except Exception as exc:
        update_batch(batch_id, status="failed", progress_message="Combining episodes failed", error=str(exc))
    finally:
        with _combine_lock:
            _active_combines.discard(batch_id)


def recover_interrupted_batches() -> None:
    """Resume final encodes lost to a restart and retry batches after space is freed."""
    for batch in list_batches_by_status(("combining", "waiting_for_space")):
        episodes = get_batch_jobs(batch["id"])
        if not episodes:
            continue
        try:
            finalize_batch_for_job(episodes[-1]["id"])
        except Exception:
            logger.exception("Cannot recover batch combine %s", batch["id"])


def start_batch_recovery_monitor(interval_seconds: int = 60) -> threading.Thread:
    def monitor() -> None:
        while True:
            recover_interrupted_batches()
            time.sleep(interval_seconds)

    thread = threading.Thread(target=monitor, name="hongguo-batch-recovery", daemon=True)
    thread.start()
    return thread
