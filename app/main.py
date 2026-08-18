import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError
from fastapi.responses import FileResponse

from . import batching
from .database import delete_job, get_batch, get_batch_jobs, get_job, init_db, list_jobs, recover_interrupted_jobs, remove_batch_job, update_batch
from .jobs import apply_subtitle_review, job_directory, remove_job_files, safe_job_file, serialize_job, worker
from .models import JobStatus
from .schemas import BatchDownloadAck, BatchResponse, JobCreate, JobResponse, JobReview
from .subtitle import read_srt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recover_interrupted_jobs()
    if os.getenv("HONGGUO_DISABLE_WORKER", "").strip().lower() not in {"1", "true", "yes"}:
        worker.start()
    batching.start_batch_recovery_monitor()
    yield
    worker.stop()


app = FastAPI(title="Hongguo Video Translator", version="2.0.0", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def serialize_batch(row: dict) -> dict:
    batch_id = row["id"]
    output_name = batching.combined_filename(row)
    output_path = batching.batch_directory(batch_id) / output_name
    episodes = get_batch_jobs(batch_id)
    return {
        "id": batch_id, "status": row["status"], "progress_message": row["progress_message"],
        "error": row["error"], "created_at": row["created_at"], "updated_at": row["updated_at"],
        # FFmpeg creates the destination before it has finalized the MP4.  An
        # existing file is therefore not evidence that it is safe to stream.
        "output": (
            f"/api/batches/{batch_id}/files/{output_name}"
            if row["status"] == "done" and output_path.is_file()
            else None
        ),
        "download_confirmed_at": row.get("download_confirmed_at"),
        "download_confirmed_bytes": row.get("download_confirmed_bytes"),
        "eligible_for_cleanup": bool(row.get("download_confirmed_at") and row["status"] == "done"),
        "episodes": [
            {
                "batch_id": item["batch_id"], "position": item["position"],
                "filename": item["filename"], "received_bytes": item.get("received_bytes"),
                "sha256": item.get("sha256"), "job_id": item["id"],
                "job": serialize_job(item),
            }
            for item in episodes
        ],
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/samples/review-local-demo.mp3", include_in_schema=False)
def local_voice_demo():
    return FileResponse(
        STATIC_DIR / "samples" / "review-local-demo.mp3",
        filename="review-local-demo.mp3",
        media_type="audio/mpeg",
    )


@app.get("/samples/ngoc-huyen-clone-demo.mp3", include_in_schema=False)
def cloned_voice_demo():
    return FileResponse(
        STATIC_DIR / "samples" / "ngoc-huyen-clone-demo.mp3",
        filename="ngoc-huyen-clone-demo.mp3",
        media_type="audio/mpeg",
    )


@app.post("/api/jobs", response_model=JobResponse, status_code=201)
def create_job(request: JobCreate):
    try:
        batching.require_upload_capacity()
    except batching.InsufficientStorageError as exc:
        raise HTTPException(507, str(exc)) from exc
    return serialize_job(worker.submit(request))


@app.post("/api/jobs/upload", response_model=JobResponse, status_code=201)
def create_upload_job(video: UploadFile = File(...), options: str = Form("{}")):
    try:
        batching.require_upload_capacity(int(video.size or 0))
        payload = json.loads(options)
        if not isinstance(payload, dict):
            raise ValueError("Upload options must be a JSON object")
        payload.pop("url", None)
        request = JobCreate(url="https://upload.local/source.mp4", **payload)
        return serialize_job(worker.submit_upload(request, video.filename or "video.mp4", video.file))
    except batching.InsufficientStorageError as exc:
        raise HTTPException(507, str(exc)) from exc
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        message = str(exc)
        status = 413 if "5 GiB" in message else 400
        raise HTTPException(status, message) from exc
    finally:
        video.file.close()


@app.post("/api/batches/upload", response_model=BatchResponse, status_code=201)
def create_folder_upload(
    videos: list[UploadFile] = File(...),
    options: str = Form("{}"),
    logo: UploadFile | None = File(default=None),
):
    mp4_videos = [item for item in videos if Path(item.filename or "").suffix.lower() == ".mp4"]
    if not mp4_videos or len(mp4_videos) > 200:
        raise HTTPException(400, "Select between 1 and 200 MP4 files")
    batch = None
    try:
        batching.require_upload_capacity(sum(int(video.size or 0) for video in mp4_videos))
        payload = json.loads(options)
        if not isinstance(payload, dict):
            raise ValueError("Upload options must be a JSON object")
        payload.pop("url", None)
        channel_name = str(payload.pop("channel_name", "")).strip()
        watermark_opacity = float(payload.pop("watermark_opacity", 0.58))
        if len(channel_name) > 80:
            raise ValueError("Channel name may contain at most 80 characters")
        if not 0.15 <= watermark_opacity <= 0.85:
            raise ValueError("Watermark opacity must be between 0.15 and 0.85")
        if bool(channel_name) != bool(logo and logo.filename):
            raise ValueError("Provide both channel name and channel logo, or leave both empty")
        request = JobCreate(url="https://upload.local/source.mp4", **payload)
        ordered = sorted(mp4_videos, key=lambda item: batching.natural_filename_key(item.filename or ""))
        batch = batching.create_folder_batch(
            request.burn_subtitles, request.dub, channel_name, watermark_opacity,
        )
        if logo and logo.filename:
            batching.save_batch_logo(batch["id"], logo.file)
        for position, video in enumerate(ordered, 1):
            row = worker.submit_upload(
                request, video.filename or f"episode-{position}.mp4", video.file, enqueue=False,
            )
            batching.attach_job(batch["id"], row["id"], position, video.filename or f"episode-{position}.mp4")
            video.file.close()
        update_batch(batch["id"], status="queued", progress_message=f"Queued {len(ordered)} episodes in filename order", error=None)
        for episode in get_batch_jobs(batch["id"]):
            worker.enqueue(episode["id"])
        batching.finalize_batch_for_job(row["id"])
        return serialize_batch(get_batch(batch["id"]))
    except batching.InsufficientStorageError as exc:
        raise HTTPException(507, str(exc)) from exc
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        if batch:
            update_batch(batch["id"], status="failed", progress_message="Folder upload failed", error=str(exc))
        message = str(exc)
        raise HTTPException(413 if "5 GiB" in message else 400, message) from exc
    finally:
        for video in videos:
            video.file.close()
        if logo:
            logo.file.close()


@app.post("/api/batches/{batch_id}/positions/{position}/replace", response_model=BatchResponse)
def replace_batch_position(
    batch_id: str, position: int, video: UploadFile = File(...), options: str = Form("{}"),
):
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    if batch["status"] != "uploading":
        raise HTTPException(409, "Only an uploading batch can replace an episode")
    expected = int(batch.get("expected_episodes") or 0)
    if position < 1 or position > expected:
        raise HTTPException(400, "Position is outside the batch range")
    old = next((item for item in get_batch_jobs(batch_id) if item["position"] == position), None)
    if old and old["status"] == "running":
        raise HTTPException(409, "Episode is currently processing; retry replacement after it stops")
    try:
        batching.require_upload_capacity(int(video.size or 0))
        payload = json.loads(options)
        if not isinstance(payload, dict):
            raise ValueError("Upload options must be a JSON object")
        payload.pop("url", None); payload.pop("channel_name", None); payload.pop("watermark_opacity", None)
        request = JobCreate(url="https://upload.local/source.mp4", **payload)
        # Validate and persist the replacement before removing the old receipt.
        row = worker.submit_upload(request, video.filename or f"episode-{position}.mp4", video.file, enqueue=False)
        if old:
            removed = remove_batch_job(batch_id, position)
            if removed:
                import shutil
                shutil.rmtree(job_directory(removed["id"]), ignore_errors=True)
        batching.attach_job(batch_id, row["id"], position, video.filename or f"episode-{position}.mp4")
        return serialize_batch(get_batch(batch_id))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(413 if "5 GiB" in str(exc) else 400, str(exc)) from exc
    finally:
        video.file.close()


@app.post("/api/batches/start", response_model=BatchResponse, status_code=201)
def start_chunked_batch(
    expected_episodes: int = Form(...), options: str = Form("{}"),
    logo: UploadFile | None = File(default=None),
):
    if not 1 <= expected_episodes <= 200:
        raise HTTPException(400, "Expected episode count must be between 1 and 200")
    batch = None
    try:
        batching.require_upload_capacity()
        payload = json.loads(options)
        if not isinstance(payload, dict):
            raise ValueError("Upload options must be a JSON object")
        payload.pop("url", None)
        channel_name = str(payload.pop("channel_name", "")).strip()
        opacity = float(payload.pop("watermark_opacity", 0.58))
        if len(channel_name) > 80 or not 0.15 <= opacity <= 0.85:
            raise ValueError("Invalid channel name or watermark opacity")
        if bool(channel_name) != bool(logo and logo.filename):
            raise ValueError("Provide both channel name and channel logo, or leave both empty")
        request = JobCreate(url="https://upload.local/source.mp4", **payload)
        batch = batching.create_folder_batch(
            request.burn_subtitles, request.dub, channel_name, opacity, expected_episodes,
        )
        if logo and logo.filename:
            batching.save_batch_logo(batch["id"], logo.file)
        return serialize_batch(get_batch(batch["id"]))
    except batching.InsufficientStorageError as exc:
        raise HTTPException(507, str(exc)) from exc
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        if batch:
            update_batch(batch["id"], status="failed", progress_message="Batch setup failed", error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    finally:
        if logo:
            logo.file.close()


@app.post("/api/batches/{batch_id}/chunks", response_model=BatchResponse)
def upload_batch_chunk(
    batch_id: str, videos: list[UploadFile] = File(...),
    positions: str = Form(...), options: str = Form("{}"),
):
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    if batch["status"] != "uploading":
        raise HTTPException(409, "Batch is no longer accepting uploads")
    try:
        batching.require_upload_capacity(sum(int(video.size or 0) for video in videos))
        requested_positions = json.loads(positions)
        if (not isinstance(requested_positions, list) or len(requested_positions) != len(videos)
                or not all(isinstance(item, int) for item in requested_positions)):
            raise ValueError("Chunk positions must contain one integer per video")
        expected = int(batch.get("expected_episodes") or 0)
        if len(set(requested_positions)) != len(requested_positions) or any(item < 1 or item > expected for item in requested_positions):
            raise ValueError("Chunk positions are duplicated or outside the batch range")
        existing = {item["position"] for item in get_batch_jobs(batch_id)}
        if existing.intersection(requested_positions):
            if set(requested_positions).issubset(existing):
                return serialize_batch(get_batch(batch_id))
            raise ValueError("Chunk overlaps positions already uploaded")
        payload = json.loads(options)
        if not isinstance(payload, dict):
            raise ValueError("Upload options must be a JSON object")
        payload.pop("url", None); payload.pop("channel_name", None); payload.pop("watermark_opacity", None)
        request = JobCreate(url="https://upload.local/source.mp4", **payload)
        invalid = [position for position, video in zip(requested_positions, videos) if Path(video.filename or "").suffix.lower() != ".mp4"]
        if invalid:
            raise ValueError(f"Positions are not MP4 files: {invalid}")
        for position, video in zip(requested_positions, videos):
            row = worker.submit_upload(
                request, video.filename or f"episode-{position}.mp4", video.file, enqueue=False,
            )
            batching.attach_job(batch_id, row["id"], position, video.filename or f"episode-{position}.mp4")
        uploaded = len(get_batch_jobs(batch_id))
        update_batch(batch_id, progress_message=f"Uploaded {uploaded}/{expected} episodes in resumable chunks", error=None)
        return serialize_batch(get_batch(batch_id))
    except batching.InsufficientStorageError as exc:
        raise HTTPException(507, str(exc)) from exc
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(413 if "5 GiB" in str(exc) else 400, str(exc)) from exc
    finally:
        for video in videos:
            video.file.close()


@app.post("/api/batches/{batch_id}/finish", response_model=BatchResponse)
def finish_chunked_batch(batch_id: str):
    batch = get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    episodes = get_batch_jobs(batch_id)
    expected = int(batch.get("expected_episodes") or 0)
    if len(episodes) != expected:
        raise HTTPException(409, f"Uploaded {len(episodes)}/{expected} episodes")
    missing_receipts = [item["position"] for item in episodes if not item.get("received_bytes") or not item.get("sha256")]
    if missing_receipts:
        raise HTTPException(409, f"Upload receipts missing for positions: {missing_receipts}")
    update_batch(batch_id, status="queued", progress_message=f"Uploaded all {expected} episodes; processing sequentially", error=None)
    for episode in episodes:
        worker.enqueue(episode["id"])
    batching.finalize_batch_for_job(episodes[-1]["id"])
    return serialize_batch(get_batch(batch_id))


@app.get("/api/batches/{batch_id}", response_model=BatchResponse)
def batch_detail(batch_id: str):
    row = get_batch(batch_id)
    if not row:
        raise HTTPException(404, "Batch not found")
    return serialize_batch(row)


@app.get("/api/batches/{batch_id}/files/{filename}")
def batch_file(batch_id: str, filename: str):
    row = get_batch(batch_id)
    if not row:
        raise HTTPException(404, "Batch not found")
    expected = batching.combined_filename(row)
    if filename != expected:
        raise HTTPException(400, "Invalid batch filename")
    if row["status"] != "done":
        raise HTTPException(409, "Combined video is still being finalized")
    path = batching.batch_directory(batch_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Combined video is not ready")
    return FileResponse(path, filename=filename, media_type="video/mp4")


@app.post("/api/batches/{batch_id}/download-complete", response_model=BatchResponse)
def confirm_batch_download(batch_id: str, request: BatchDownloadAck):
    row = get_batch(batch_id)
    if not row:
        raise HTTPException(404, "Batch not found")
    if row["status"] != "done":
        raise HTTPException(409, "Batch output is not finalized")
    filename = batching.combined_filename(row)
    path = batching.batch_directory(batch_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Combined video is not ready")
    actual_size = path.stat().st_size
    if request.size_bytes != actual_size:
        raise HTTPException(409, f"Downloaded size mismatch: local={request.size_bytes}, server={actual_size}")
    from .database import utc_now
    update_batch(batch_id, download_confirmed_at=utc_now(), download_confirmed_bytes=actual_size)
    # The local client has verified the exact byte count. Keeping another
    # multi-gigabyte server copy only blocks subsequent uploads.
    path.unlink()
    return serialize_batch(get_batch(batch_id))


@app.get("/api/storage")
def storage_detail():
    return batching.storage_status()


@app.get("/api/jobs", response_model=list[JobResponse])
def recent_jobs(limit: int = Query(default=50, ge=1, le=200)):
    return [serialize_job(row) for row in list_jobs(limit)]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def job_detail(job_id: str):
    row = get_job(job_id)
    if not row:
        raise HTTPException(404, "Job not found")
    return serialize_job(row)


@app.get("/api/jobs/{job_id}/subtitles")
def job_subtitles(job_id: str):
    if not get_job(job_id):
        raise HTTPException(404, "Job not found")
    source_path, translated_path = job_directory(job_id) / "source.srt", job_directory(job_id) / "vi.srt"
    if not source_path.is_file() or not translated_path.is_file():
        raise HTTPException(404, "Subtitles are not ready")
    source = {cue.index: cue for cue in read_srt(source_path)}
    translated = {cue.index: cue for cue in read_srt(translated_path)}
    return [{
        "id": item_id, "start": str(source[item_id].start), "end": str(source[item_id].end),
        "source": source[item_id].content, "translation": translated.get(item_id).content if item_id in translated else "",
    } for item_id in source]


@app.patch("/api/jobs/{job_id}/subtitles", status_code=202)
def review_subtitles(job_id: str, request: JobReview):
    row = get_job(job_id)
    if not row:
        raise HTTPException(404, "Job not found")
    if row["status"] not in {JobStatus.DONE, JobStatus.NEEDS_REVIEW} or worker.is_active(job_id):
        raise HTTPException(409, "Job must be finished before subtitles can be edited")
    try:
        apply_subtitle_review(job_id, {item.id: item.text for item in request.translations})
        worker.retry(job_id, "Human edits saved; rendering affected outputs from checkpoint")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}/files/{filename}")
def job_file(job_id: str, filename: str):
    if not get_job(job_id):
        raise HTTPException(404, "Job not found")
    try:
        path = safe_job_file(job_id, filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename, media_type="application/octet-stream")


@app.delete("/api/jobs/{job_id}", status_code=204)
def remove_job(job_id: str):
    row = get_job(job_id)
    if not row:
        raise HTTPException(404, "Job not found")
    if row["status"] in {JobStatus.RUNNING, JobStatus.QUEUED} or worker.is_active(job_id):
        raise HTTPException(409, "A queued or running job cannot be deleted")
    remove_job_files(job_id)
    delete_job(job_id)
