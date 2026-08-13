import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .database import delete_job, get_job, init_db, list_jobs, recover_interrupted_jobs
from .jobs import remove_job_files, safe_job_file, serialize_job, worker
from .models import JobStatus
from .schemas import JobCreate, JobResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    recover_interrupted_jobs()
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="Hongguo Video Translator", version="1.0.0", lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / "static"


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


@app.get("/api/jobs", response_model=list[JobResponse])
def recent_jobs(limit: int = Query(default=50, ge=1, le=200)):
    return [serialize_job(row) for row in list_jobs(limit)]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def job_detail(job_id: str):
    row = get_job(job_id)
    if not row:
        raise HTTPException(404, "Job not found")
    return serialize_job(row)


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
