"""Folder-batch orchestration and final episode concatenation."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

from . import media
from .config import get_settings
from .database import (
    add_batch_job, create_batch, get_batch, get_batch_jobs, get_job_batch, update_batch, utc_now,
)


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


def finalize_batch_for_job(job_id: str) -> None:
    batch_id = get_job_batch(job_id)
    if not batch_id:
        return
    batch, episodes = get_batch(batch_id), get_batch_jobs(batch_id)
    if not batch or not episodes:
        return
    if batch["status"] == "uploading":
        return
    failed = [item for item in episodes if item["status"] in {"failed", "needs_review"}]
    if failed:
        first = failed[0]
        update_batch(
            batch_id, status="failed",
            progress_message=f"Episode {first['position']} failed",
            error=f"{first['filename']}: {first.get('error') or first['status']}",
        )
        return
    completed = sum(item["status"] == "done" for item in episodes)
    if completed < len(episodes):
        update_batch(
            batch_id, status="running",
            progress_message=f"Completed episode {completed}/{len(episodes)}; processing sequentially",
            error=None,
        )
        return
    update_batch(batch_id, status="combining", progress_message=f"Combining {len(episodes)} translated episodes", error=None)
    source_name = "vi-dubbed.mp4" if batch["dub"] else "vi-burned.mp4" if batch["burn_subtitles"] else "source.mp4"
    inputs = [get_settings().jobs_dir / item["id"] / source_name for item in episodes]
    directory = batch_directory(batch_id)
    output = directory / combined_filename(batch)
    try:
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
    except Exception as exc:
        update_batch(batch_id, status="failed", progress_message="Combining episodes failed", error=str(exc))
