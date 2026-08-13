import logging
import json
import queue
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from . import downloader, media, speech_pipeline, translator, tts
from .config import get_settings
from .database import create_job as db_create_job
from .database import get_job, list_jobs, update_job, utc_now
from .models import JobStatus, JobStep
from .schemas import JobCreate
from .subtitle import read_srt, write_srt

logger = logging.getLogger(__name__)
OUTPUT_NAMES = {
    "video": "source.mp4", "source_srt": "source.srt", "translated_srt": "vi.srt",
    "burned_video": "vi-burned.mp4", "dubbed_video": "vi-dubbed.mp4",
    "speaker_report": "voice-profiles.json",
}


def job_directory(job_id: str) -> Path:
    return get_settings().jobs_dir / job_id


def safe_job_file(job_id: str, filename: str) -> Path:
    root = job_directory(job_id).resolve()
    candidate = (root / filename).resolve()
    if candidate.parent != root or filename not in set(OUTPUT_NAMES.values()) | {"vi-dub.wav"}:
        raise ValueError("Invalid job filename")
    return candidate


def serialize_job(row: dict[str, Any]) -> dict[str, Any]:
    job_id = row["id"]
    outputs = {}
    for key, filename in OUTPUT_NAMES.items():
        path = job_directory(job_id) / filename
        outputs[key] = f"/api/jobs/{job_id}/files/{filename}" if path.is_file() else None
    return {
        "id": job_id, "url": row["url"], "status": row["status"], "step": row["step"],
        "progress_message": row["progress_message"], "error": row["error"],
        "provider": row["provider"], "asr_model": row["asr_model"],
        "diarize": bool(row.get("diarize", 0)),
        "burn_subtitles": bool(row["burn_subtitles"]), "dub": bool(row["dub"]),
        "created_at": row["created_at"], "updated_at": row["updated_at"], "outputs": outputs,
    }


class JobWorker:
    def __init__(self) -> None:
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.active_job_id: str | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="hongguo-job-worker", daemon=True)
        self.thread.start()
        for row in reversed(list_jobs(1000)):
            if row["status"] == JobStatus.QUEUED:
                self.queue.put(row["id"])

    def stop(self) -> None:
        self.stop_event.set()
        self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=10)

    def submit(self, request: JobCreate) -> dict[str, Any]:
        job_id, now = str(uuid.uuid4()), utc_now()
        values = {
            "id": job_id, "url": request.url, "status": JobStatus.QUEUED, "step": JobStep.QUEUED,
            "progress_message": "Waiting for worker", "error": None, "provider": request.provider,
            "asr_model": request.asr_model, "source_language_code": request.source_language_code,
            "diarize": int(request.diarize), "min_speakers": request.min_speakers, "max_speakers": request.max_speakers,
            "source_language": request.source_language, "target_language": request.target_language,
            "glossary": request.glossary, "burn_subtitles": int(request.burn_subtitles), "dub": int(request.dub),
            "narrator_mode": int(request.narrator_mode),
            "tts_voice": request.tts_voice, "tts_secondary_voice": request.tts_secondary_voice,
            "voice_overrides": json.dumps(request.voice_overrides),
            "original_audio_volume": request.original_audio_volume,
            "created_at": now, "updated_at": now,
        }
        job_directory(job_id).mkdir(parents=True)
        row = db_create_job(values)
        self.queue.put(job_id)
        return row

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return self.active_job_id == job_id

    def _progress(self, job_id: str, step: JobStep, message: str) -> None:
        logger.info("job=%s step=%s %s", job_id, step, message)
        update_job(job_id, status=JobStatus.RUNNING, step=step, progress_message=message)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            job_id = self.queue.get()
            if job_id is None:
                self.queue.task_done()
                break
            with self._lock:
                self.active_job_id = job_id
            try:
                row = get_job(job_id)
                if row and row["status"] == JobStatus.QUEUED:
                    self._process(row)
            except Exception as exc:
                logger.exception("Job %s failed", job_id)
                update_job(job_id, status=JobStatus.FAILED, progress_message="Job failed", error=str(exc))
            finally:
                with self._lock:
                    self.active_job_id = None
                self.queue.task_done()

    def _process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        directory = job_directory(job_id)
        self._progress(job_id, JobStep.DOWNLOADING, "Downloading video")

        def download_progress(info):
            if info.get("status") == "downloading" and info.get("_percent_str"):
                self._progress(job_id, JobStep.DOWNLOADING, f"Downloading {info['_percent_str'].strip()}")

        video = downloader.download_video(job["url"], directory, download_progress)
        source_srt = directory / "source.srt"
        speakers_path = directory / "speakers.json"
        if source_srt.is_file():
            self._progress(job_id, JobStep.TRANSCRIBING, "Resuming from existing aligned transcript")
            subtitles = read_srt(source_srt)
            speakers = {int(key): value for key, value in json.loads(speakers_path.read_text()).items()} if speakers_path.is_file() else {}
        else:
            self._progress(job_id, JobStep.TRANSCRIBING, f"WhisperX aligned transcription with {job['asr_model']}")
            subtitles, speakers = speech_pipeline.transcribe_aligned(
                video, source_srt, job["asr_model"], job["source_language_code"], bool(job.get("diarize")),
                job.get("min_speakers"), job.get("max_speakers"),
                lambda message: self._progress(job_id, JobStep.TRANSCRIBING, message),
            )
        self._progress(job_id, JobStep.TRANSLATING, f"Translating {len(subtitles)} cues via {job['provider']}")
        translated = translator.translate_subtitles(
            subtitles, job["provider"], job["source_language"], job["target_language"], job["glossary"],
            lambda message: self._progress(job_id, JobStep.TRANSLATING, message),
            speakers,
        )
        translated_srt = directory / "vi.srt"
        write_srt(translated_srt, translated)
        if job["burn_subtitles"]:
            self._progress(job_id, JobStep.BURNING, "Burning Vietnamese subtitles into video")
            media.burn_subtitles(video, translated_srt, directory / "vi-burned.mp4")
        if job["dub"]:
            self._progress(job_id, JobStep.DUBBING, "Creating Vietnamese basic dub")
            dub_video_base = directory / "vi-burned.mp4" if job["burn_subtitles"] else video
            tts.create_dub(
                dub_video_base, translated, directory, job["tts_voice"], job["original_audio_volume"],
                lambda message: self._progress(job_id, JobStep.DUBBING, message),
                speakers, job.get("tts_secondary_voice", "vi-VN-NamMinhNeural"),
                json.loads(job.get("voice_overrides") or "{}"),
                bool(job.get("narrator_mode", 1)),
            )
        update_job(job_id, status=JobStatus.DONE, step=JobStep.DONE, progress_message="All requested outputs are ready", error=None)


worker = JobWorker()


def remove_job_files(job_id: str) -> None:
    directory = job_directory(job_id)
    if directory.is_symlink():
        raise ValueError("Refusing to remove a symlinked job directory")
    if directory.exists():
        shutil.rmtree(directory)
