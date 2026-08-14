import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import ValidationError
from fastapi.responses import FileResponse

from . import batching
from .database import delete_job, get_batch, get_batch_jobs, get_job, init_db, list_jobs, recover_interrupted_jobs, update_batch
from .jobs import apply_subtitle_review, job_directory, remove_job_files, safe_job_file, serialize_job, worker
from .models import JobStatus
from .schemas import BatchResponse, JobCreate, JobResponse, JobReview
from .subtitle import read_srt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recover_interrupted_jobs()
    worker.start()
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
        "output": f"/api/batches/{batch_id}/files/{output_name}" if output_path.is_file() else None,
        "episodes": [
            {"position": item["position"], "filename": item["filename"], "job": serialize_job(item)}
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
    return serialize_job(worker.submit(request))


@app.post("/api/jobs/upload", response_model=JobResponse, status_code=201)
def create_upload_job(video: UploadFile = File(...), options: str = Form("{}")):
    try:
        payload = json.loads(options)
        if not isinstance(payload, dict):
            raise ValueError("Upload options must be a JSON object")
        payload.pop("url", None)
        request = JobCreate(url="https://upload.local/source.mp4", **payload)
        return serialize_job(worker.submit_upload(request, video.filename or "video.mp4", video.file))
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
    if not videos or len(videos) > 200:
        raise HTTPException(400, "Select between 1 and 200 MP4 files")
    batch = None
    try:
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
        ordered = sorted(videos, key=lambda item: batching.natural_filename_key(item.filename or ""))
        if any(Path(item.filename or "").suffix.lower() != ".mp4" for item in ordered):
            raise ValueError("Folder may contain MP4 files only")
        batch = batching.create_folder_batch(
            request.burn_subtitles, request.dub, channel_name, watermark_opacity,
        )
        if logo and logo.filename:
            batching.save_batch_logo(batch["id"], logo.file)
        for position, video in enumerate(ordered, 1):
            row = worker.submit_upload(request, video.filename or f"episode-{position}.mp4", video.file)
            batching.attach_job(batch["id"], row["id"], position, video.filename or f"episode-{position}.mp4")
            video.file.close()
        update_batch(batch["id"], status="queued", progress_message=f"Queued {len(ordered)} episodes in filename order", error=None)
        batching.finalize_batch_for_job(row["id"])
        return serialize_batch(get_batch(batch["id"]))
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
    path = batching.batch_directory(batch_id) / filename
    if not path.is_file():
        raise HTTPException(404, "Combined video is not ready")
    return FileResponse(path, filename=filename, media_type="video/mp4")


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
