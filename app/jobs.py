import logging
import json
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, BinaryIO

import srt

from . import batching, dialogue_master, downloader, media, qa, speech_pipeline, translator, tts
from .artifacts import ArtifactManifest, stable_hash
from .artifacts import atomic_write_json
from .dialogue import build_dialogue_units, repair_fragment_speakers
from .config import get_settings
from .database import create_job as db_create_job
from .database import get_job, list_jobs, update_job, utc_now
from .models import JobStatus, JobStep
from .schemas import JobCreate
from .subtitle import read_srt, write_srt
from .source_subtitle_mask import SubtitleRegion, detect_source_subtitle_regions

logger = logging.getLogger(__name__)
OUTPUT_NAMES = {
    "video": "source.mp4", "source_srt": "source.srt", "translated_srt": "vi.srt",
    "burned_video": "vi-burned.mp4", "dubbed_video": "vi-dubbed.mp4",
    "speaker_report": "voice-profiles.json",
    "qa_report": "qa-report.json",
    "subtitle_regions": "subtitle-regions.json",
    "subtitle_layout": "subtitle-layout.json",
}
UPLOAD_EXTENSIONS = {".mp4"}


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
        "burn_subtitles": bool(row["burn_subtitles"]), "hide_source_subtitles": bool(row.get("hide_source_subtitles", 0)), "dub": bool(row["dub"]),
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

    def _submit(self, request: JobCreate, job_id: str | None = None) -> dict[str, Any]:
        job_id, now = job_id or str(uuid.uuid4()), utc_now()
        values = {
            "id": job_id, "url": request.url, "status": JobStatus.QUEUED, "step": JobStep.QUEUED,
            "progress_message": "Waiting for worker", "error": None, "provider": request.provider,
            "asr_model": request.asr_model, "source_language_code": request.source_language_code,
            "diarize": int(request.diarize), "min_speakers": request.min_speakers, "max_speakers": request.max_speakers,
            "source_language": request.source_language, "target_language": request.target_language,
            "glossary": request.glossary, "burn_subtitles": int(request.burn_subtitles), "dub": int(request.dub),
            "hide_source_subtitles": int(request.hide_source_subtitles),
            "narrator_mode": int(request.narrator_mode),
            "tts_voice": request.tts_voice, "tts_secondary_voice": request.tts_secondary_voice,
            "voice_overrides": json.dumps(request.voice_overrides),
            "original_audio_volume": request.original_audio_volume,
            "created_at": now, "updated_at": now,
        }
        job_directory(job_id).mkdir(parents=True, exist_ok=True)
        row = db_create_job(values)
        self.queue.put(job_id)
        return row

    def submit(self, request: JobCreate) -> dict[str, Any]:
        return self._submit(request)

    def submit_upload(self, request: JobCreate, filename: str, stream: BinaryIO) -> dict[str, Any]:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in UPLOAD_EXTENSIONS:
            raise ValueError("Only MP4 uploads are supported")
        job_id = str(uuid.uuid4())
        directory = job_directory(job_id)
        directory.mkdir(parents=True)
        temporary = directory / ".source.uploading"
        output = directory / "source.mp4"
        total = 0
        try:
            with temporary.open("wb") as target:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > get_settings().max_upload_bytes:
                        raise ValueError("Uploaded video exceeds the 5 GiB limit")
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if total < 1024:
                raise ValueError("Uploaded video is empty or truncated")
            os.replace(temporary, output)
            duration = media.probe_duration(output)
            if duration <= 0:
                raise ValueError("Uploaded MP4 has no readable video duration")
            upload_request = request.model_copy(update={"url": f"upload://{Path(filename).name}"})
            return self._submit(upload_request, job_id)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return self.active_job_id == job_id

    def retry(self, job_id: str, message: str = "Queued for render from checkpoint") -> None:
        row = get_job(job_id)
        if not row or row["status"] not in {JobStatus.DONE, JobStatus.FAILED, JobStatus.NEEDS_REVIEW}:
            raise ValueError("Only a completed, failed, or review job can be retried")
        update_job(job_id, status=JobStatus.QUEUED, step=JobStep.QUEUED, progress_message=message, error=None)
        self.queue.put(job_id)

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
                try:
                    batching.finalize_batch_for_job(job_id)
                except Exception:
                    logger.exception("Cannot update folder batch for job %s", job_id)
                with self._lock:
                    self.active_job_id = None
                self.queue.task_done()

    def _process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        directory = job_directory(job_id)
        manifest = ArtifactManifest(directory)
        uploaded = str(job["url"]).startswith("upload://")
        self._progress(job_id, JobStep.DOWNLOADING, "Validating uploaded video" if uploaded else "Downloading video")

        def download_progress(info):
            if info.get("status") == "downloading" and info.get("_percent_str"):
                self._progress(job_id, JobStep.DOWNLOADING, f"Downloading {info['_percent_str'].strip()}")

        video = directory / "source.mp4" if uploaded else downloader.download_video(job["url"], directory, download_progress)
        if uploaded and (not video.is_file() or video.stat().st_size < 1024):
            raise RuntimeError("Uploaded source.mp4 is missing or truncated")
        manifest.complete("download", stable_hash({"url": job["url"]}), [video], {"bytes": video.stat().st_size})
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
        manifest.complete(
            "transcript", stable_hash({"video_bytes": video.stat().st_size, "model": job["asr_model"], "language": job["source_language_code"], "diarize": bool(job.get("diarize"))}),
            [source_srt], {"cues": len(subtitles)},
        )
        speakers = repair_fragment_speakers(subtitles, speakers)
        speakers_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
        units = build_dialogue_units(subtitles, speakers)
        (directory / "dialogue-units.json").write_text(json.dumps([unit.to_dict() for unit in units], ensure_ascii=False, indent=2), encoding="utf-8")
        self._progress(job_id, JobStep.TRANSLATING, f"Translating {len(subtitles)} cues via {job['provider']}")
        translated_srt = directory / "vi.srt"
        translation_fingerprint = stable_hash({"source": [(cue.index, cue.content) for cue in subtitles], "provider": job["provider"], "target": job["target_language"], "glossary": job["glossary"], "pipeline": "context-v2"})
        if manifest.valid("translation", translation_fingerprint, [translated_srt, directory / "vi-final.json"]):
            self._progress(job_id, JobStep.TRANSLATING, "Resuming from context-edited Vietnamese translation")
            translated = read_srt(translated_srt)
        else:
            translated = translator.translate_subtitles(
                subtitles, job["provider"], job["source_language"], job["target_language"], job["glossary"],
                lambda message: self._progress(job_id, JobStep.TRANSLATING, message),
                speakers, directory,
            )
            write_srt(translated_srt, translated)
            manifest.complete("translation", translation_fingerprint, [translated_srt, directory / "vi-final.json"], {"cues": len(translated)})
        # The final translation draft is stable even after vi.srt is redistributed
        # into original display-line slots by the dialogue master.
        draft_payload = json.loads((directory / "vi-final.json").read_text(encoding="utf-8"))
        draft_mapping = {int(item["id"]): str(item["text"]) for item in draft_payload}
        draft = [srt.Subtitle(cue.index, cue.start, cue.end, draft_mapping.get(cue.index, cue.content)) for cue in translated]
        regions: list[SubtitleRegion] = []
        regions_path = directory / "subtitle-regions.json"
        mask_fingerprint = stable_hash({"video_bytes": video.stat().st_size, "detector": "visual-tracks-v2"})
        if job["burn_subtitles"] and job.get("hide_source_subtitles", 0):
            if manifest.valid("source_subtitle_mask", mask_fingerprint, [regions_path]):
                payload = json.loads(regions_path.read_text(encoding="utf-8"))
                regions = [SubtitleRegion(**item) for item in payload.get("regions", [])]
                self._progress(job_id, JobStep.DETECTING_SUBTITLES, "Resuming detected source subtitle mask")
            else:
                self._progress(job_id, JobStep.DETECTING_SUBTITLES, "Automatically locating burned-in source subtitles")
                regions = detect_source_subtitle_regions(video, subtitles, regions_path)
                manifest.complete("source_subtitle_mask", mask_fingerprint, [regions_path], {"regions": len(regions)})
        box_widths = {}
        for cue in draft:
            start, end = cue.start.total_seconds(), cue.end.total_seconds()
            matching = sorted(regions, key=lambda region: max(0.0, min(end, region.end)-max(start, region.start)), reverse=True)
            if matching:
                box_widths[cue.index] = matching[0].width
        master_path = directory / "dialogue-master.json"
        master_fingerprint = stable_hash({
            "draft": [(cue.index, cue.content) for cue in draft], "source": [(cue.index, cue.content) for cue in subtitles],
            "speakers": speakers, "provider": job["provider"], "boxes": box_widths, "pipeline": "display-lines-v3",
        })
        if manifest.valid("dialogue_master", master_fingerprint, [translated_srt, master_path]):
            translated = read_srt(translated_srt)
            master_payload = json.loads(master_path.read_text(encoding="utf-8"))
            self._progress(job_id, JobStep.TRANSLATING, "Resuming validated dialogue master")
        else:
            self._progress(job_id, JobStep.TRANSLATING, "Reflowing complete dialogue into original display lines")
            translated, master_utterances, master_warning = dialogue_master.build_dialogue_master(
                draft, subtitles, speakers, job["provider"], box_widths,
            )
            write_srt(translated_srt, translated)
            master_payload = {
                "warning": master_warning, "utterances": [item.to_dict() for item in master_utterances],
                "display_lines": [{"id": cue.index, "text": cue.content} for cue in translated],
            }
            atomic_write_json(master_path, master_payload)
            manifest.complete("dialogue_master", master_fingerprint, [translated_srt, master_path], {"utterances": len(master_utterances)})
        if job["burn_subtitles"]:
            burned = directory / "vi-burned.mp4"
            burn_fingerprint = stable_hash({"video_bytes": video.stat().st_size, "srt": translated_srt.read_text(encoding="utf-8"), "style": "adaptive-ass-v10-h264-crf23", "mask": [region.to_dict() for region in regions]})
            expected_burn_files = [burned] + ([directory / "vi.ass", directory / "subtitle-layout.json"] if regions else [])
            if manifest.valid("burn", burn_fingerprint, expected_burn_files):
                self._progress(job_id, JobStep.BURNING, "Resuming from burned subtitle video")
            else:
                self._progress(job_id, JobStep.BURNING, "Burning Vietnamese subtitles into video")
                media.burn_subtitles(video, translated_srt, burned, regions)
                manifest.complete("burn", burn_fingerprint, expected_burn_files)
        if job["dub"]:
            dub_video_base = directory / "vi-burned.mp4" if job["burn_subtitles"] else video
            dubbed = directory / "vi-dubbed.mp4"
            base_stat = dub_video_base.stat()
            dub_fingerprint = stable_hash({"translation": [(cue.index, cue.content) for cue in translated], "master": master_payload, "video_base": [base_stat.st_size, base_stat.st_mtime_ns], "voice": job["tts_voice"], "original_audio_volume": job["original_audio_volume"], "mix": "sidechain-v4-48k-aac128", "prosody": "conservative-v1", "timing": "atempo-v2-1.35"})
            if manifest.valid("dub", dub_fingerprint, [dubbed, directory / "speech-plan.json", directory / "prosody-plan.json", directory / "tts-timing.json"]):
                self._progress(job_id, JobStep.DUBBING, "Resuming from cached Vietnamese dub")
            else:
                self._progress(job_id, JobStep.DUBBING, "Creating Vietnamese natural dub")
                tts.create_dub(
                    dub_video_base, translated, directory, job["tts_voice"], job["original_audio_volume"],
                    lambda message: self._progress(job_id, JobStep.DUBBING, message),
                    speakers, job.get("tts_secondary_voice", "vi-VN-NamMinhNeural"),
                    json.loads(job.get("voice_overrides") or "{}"),
                    bool(job.get("narrator_mode", 1)),
                    job["provider"],
                    master_payload.get("utterances"),
                )
                manifest.complete("dub", dub_fingerprint, [dubbed, directory / "speech-plan.json", directory / "prosody-plan.json", directory / "tts-timing.json"])
        self._progress(job_id, JobStep.QA, "Running automatic content, timing, and media QA")
        final_video = directory / "vi-dubbed.mp4" if job["dub"] else directory / "vi-burned.mp4" if job["burn_subtitles"] else video
        report = qa.validate_job(directory, translated, [cue.index for cue in subtitles], final_video)
        manifest.complete("qa", stable_hash({"translation": [(cue.index, cue.content) for cue in translated], "master": master_payload, "output_bytes": final_video.stat().st_size}), [directory / "qa-report.json"], report["summary"])
        status = JobStatus.NEEDS_REVIEW if report["summary"]["error"] else JobStatus.DONE
        message = f"QA complete: {report['summary']['pass']} passed, {report['summary']['warning']} warnings, {report['summary']['error']} errors"
        update_job(job_id, status=status, step=JobStep.DONE, progress_message=message, error=None)


worker = JobWorker()


def apply_subtitle_review(job_id: str, edits: dict[int, str]) -> None:
    directory = job_directory(job_id)
    translated_path = directory / "vi.srt"
    if not translated_path.is_file():
        raise ValueError("Vietnamese subtitles are not available")
    subtitles = read_srt(translated_path)
    known = {cue.index for cue in subtitles}
    unknown = sorted(set(edits) - known)
    if unknown:
        raise ValueError(f"Unknown subtitle ids: {unknown}")
    for cue in subtitles:
        if cue.index in edits:
            cue.content = edits[cue.index].strip()
    write_srt(translated_path, subtitles)
    atomic_write_json(directory / "vi-final.json", [{"id": cue.index, "text": cue.content} for cue in subtitles])
    manifest = ArtifactManifest(directory)
    translation = manifest.data.get("artifacts", {}).get("translation")
    if translation:
        manifest.complete("translation", translation["fingerprint"], [translated_path, directory / "vi-final.json"], {"cues": len(subtitles), "human_reviewed": True})
        manifest.invalidate_after(["download", "transcript", "source_subtitle_mask", "translation", "dialogue_master", "burn", "dub", "qa"], "translation")
    for filename in ("dialogue-master.json", "vi-burned.mp4", "vi-dubbed.mp4", "vi.ass", "subtitle-layout.json", "qa-report.json"):
        (directory / filename).unlink(missing_ok=True)


def remove_job_files(job_id: str) -> None:
    directory = job_directory(job_id)
    if directory.is_symlink():
        raise ValueError("Refusing to remove a symlinked job directory")
    if directory.exists():
        shutil.rmtree(directory)
