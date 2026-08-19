import logging
import hashlib
import json
import os
import queue
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable

import srt

from . import batching, dialogue_master, downloader, media, ocr_subtitles, qa, speech_pipeline, translator, tts
from .artifacts import ArtifactManifest, stable_hash
from .artifacts import atomic_write_json
from .dialogue import build_dialogue_units, repair_fragment_speakers
from .config import get_settings
from .database import create_job as db_create_job
from .database import get_batch, get_job, get_job_batch, list_jobs, update_job, utc_now
from .models import JobStatus, JobStep
from .schemas import JobCreate
from .subtitle import read_srt, write_srt
from .source_subtitle_mask import SubtitleRegion, detect_source_subtitle_regions

logger = logging.getLogger(__name__)
OUTPUT_NAMES = {
    "source_srt": "source.srt", "translated_srt": "vi.srt",
    "burned_video": "vi-burned.mp4", "dubbed_video": "vi-dubbed.mp4",
    "speaker_report": "voice-profiles.json",
    "qa_report": "qa-report.json",
    "subtitle_regions": "subtitle-regions.json",
    "subtitle_layout": "subtitle-layout.json",
    "aligned_srt": "vi-aligned.srt",
    "ai_usage": "ai-usage.json",
}
UPLOAD_EXTENSIONS = {".mp4"}


def _extract_low_confidence_ocr_indices(ocr_report: dict, low_score_threshold: float | None = None) -> list[int]:
    threshold = (
        float(low_score_threshold)
        if low_score_threshold is not None
        else float(ocr_subtitles.LOW_SCORE_RECOGNITION_THRESHOLD)
    )
    items = ocr_report.get("items", [])
    low_ids: list[int] = []
    for index, item in enumerate(items, 1):
        score = float(item.get("score", 1.0))
        if score < threshold:
            low_ids.append(index)
    return low_ids


def _interval_overlap_seconds(
    start_a_s: float, end_a_s: float, start_b_s: float, end_b_s: float,
) -> float:
    intersection = max(0.0, min(end_a_s, end_b_s) - max(start_a_s, start_b_s))
    if intersection <= 0:
        return 0.0
    span = max(0.001, end_a_s - start_a_s, end_b_s - start_b_s)
    return intersection / span


def _replace_ocr_with_asr_when_needed(
    subtitles: list[srt.Subtitle],
    asr_subtitles: list[srt.Subtitle],
    low_confidence_ids: list[int],
) -> tuple[list[srt.Subtitle], int]:
    if not subtitles or not asr_subtitles or not low_confidence_ids:
        return subtitles, 0
    low_confidence_ids = set(low_confidence_ids)
    asr_by_index = list(enumerate(asr_subtitles, 1))
    replaced = 0
    updated: list[srt.Subtitle] = []
    for cue in subtitles:
        if cue.index not in low_confidence_ids:
            updated.append(cue)
            continue
        source_start = cue.start.total_seconds()
        source_end = cue.end.total_seconds()
        best_score = 0.0
        best_text: str | None = None
        for _, asr_cue in asr_by_index:
            overlap = _interval_overlap_seconds(
                source_start, source_end, asr_cue.start.total_seconds(), asr_cue.end.total_seconds(),
            )
            if overlap > best_score:
                best_score = overlap
                best_text = asr_cue.content.strip()
        if best_text and best_score >= 0.25:
            cue = srt.Subtitle(
                index=cue.index,
                start=cue.start,
                end=cue.end,
                content=best_text,
            )
            replaced += 1
        updated.append(cue)
    return updated, replaced


def _asr_rescue_low_ocr_segments(
    directory: Path,
    video: Path,
    subtitles: list[srt.Subtitle],
    ocr_report_path: Path,
    source_language_code: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[srt.Subtitle], dict[int, str]]:
    if not ocr_report_path.is_file():
        return subtitles, {}
    report = json.loads(ocr_report_path.read_text(encoding="utf-8"))
    low_confidence_ids = _extract_low_confidence_ocr_indices(report)
    if not low_confidence_ids:
        return subtitles, {}
    asr_srt = directory / "source_asr_fallback.srt"
    try:
        if progress:
            progress(
                f"Fallback ASR for {len(low_confidence_ids)} low-confidence OCR cues (source IDs: {low_confidence_ids[:8]}..."
            )
        # Use the user's preferred ASR model for rescues so alignment remains stable.
        # Qwen-ASR can be slower but gives stronger correction on noisy subtitles.
        asr_subtitles, asr_speakers = speech_pipeline.transcribe_aligned(
            video, asr_srt, "qwen3-asr-1.7b", source_language_code, True, 1, 6, progress,
        )
    except Exception as exc:
        if progress:
            progress(f"ASR rescue fallback skipped due to ASR failure: {exc}")
        return subtitles, {}
    repaired, replaced_count = _replace_ocr_with_asr_when_needed(subtitles, asr_subtitles, low_confidence_ids)
    if progress:
        progress(f"Repaired {replaced_count} OCR cues from ASR fallback")
    return repaired, asr_speakers if replaced_count else {}


def _recover_speakers_from_overlapping_asr(
    directory: Path,
    video: Path,
    subtitles: list[srt.Subtitle],
    asr_model: str,
    source_language_code: str,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    progress: Callable[[str], None] | None = None,
) -> dict[int, str]:
    if not subtitles:
        return {}
    asr_srt = directory / "source_asr_for_speakers.srt"
    asr_subtitles, asr_speakers = speech_pipeline.transcribe_aligned(
        video, asr_srt, asr_model, source_language_code, diarize, min_speakers, max_speakers, progress,
    )
    if not asr_subtitles:
        return {}
    labels = {label for label in asr_speakers.values() if label}
    if len(labels) < 2:
        return {}
    asr_items = [(index + 1, cue, asr_speakers.get(index + 1)) for index, cue in enumerate(asr_subtitles)]
    recovered: dict[int, str] = {}
    for cue in subtitles:
        start = cue.start.total_seconds()
        end = cue.end.total_seconds()
        best_score = 0.0
        best_label = None
        for _, asr_cue, label in asr_items:
            if not label:
                continue
            score = _interval_overlap_seconds(start, end, asr_cue.start.total_seconds(), asr_cue.end.total_seconds())
            if score > best_score:
                best_score = score
                best_label = label
        if best_label and best_score >= 0.18:
            recovered[cue.index] = best_label
    return recovered


def cleanup_completed_job_media(directory: Path, final_video: Path) -> list[str]:
    """Keep only the deliverable video; remove source and render scratch media."""
    removed: list[str] = []
    for name in ("source.mp4", "vi-burned.mp4", "speaker-analysis.wav", "vi-dub.wav"):
        path = directory / name
        if path == final_video or not path.exists():
            continue
        path.unlink()
        removed.append(name)
    tts_dir = directory / "tts"
    if tts_dir.is_dir():
        shutil.rmtree(tts_dir)
        removed.append("tts/")
    return removed


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
        "provider": row["provider"],
        "translation_draft_provider": row.get("translation_draft_provider", row["provider"]),
        "translation_refine_provider": row.get("translation_refine_provider", row["provider"]),
        "asr_model": row["asr_model"],
        "received_bytes": row.get("received_bytes"), "sha256": row.get("sha256"),
        "retry_count": int(row.get("retry_count") or 0),
        "diarize": bool(row.get("diarize", 0)),
        "burn_subtitles": bool(row["burn_subtitles"]), "hide_source_subtitles": bool(row.get("hide_source_subtitles", 0)), "dub": bool(row["dub"]),
        "speaker_gender_profile": row.get("speaker_gender_profile", "auto"),
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
            batch_id = get_job_batch(row["id"])
            batch = get_batch(batch_id) if batch_id else None
            if row["status"] == JobStatus.QUEUED and (not batch or batch["status"] != "uploading"):
                self.queue.put(row["id"])
            elif row["status"] == JobStatus.FAILED and self._is_retryable_job(row):
                self.queue.put(row["id"])

    def stop(self) -> None:
        self.stop_event.set()
        self.queue.put(None)
        if self.thread:
            self.thread.join(timeout=10)

    def _submit(
        self, request: JobCreate, job_id: str | None = None, *,
        received_bytes: int | None = None, sha256: str | None = None, enqueue: bool = True,
    ) -> dict[str, Any]:
        job_id, now = job_id or str(uuid.uuid4()), utc_now()
        resolved_refine_provider = request.translation_refine_provider
        if resolved_refine_provider == "auto":
            resolved_refine_provider = request.provider
        values = {
            "id": job_id, "url": request.url, "status": JobStatus.QUEUED, "step": JobStep.QUEUED,
            "progress_message": "Waiting for worker", "error": None, "provider": request.provider,
            "translation_draft_provider": request.translation_draft_provider,
            "translation_refine_provider": resolved_refine_provider,
            "asr_model": request.asr_model, "source_language_code": request.source_language_code,
            "diarize": int(request.diarize), "min_speakers": request.min_speakers, "max_speakers": request.max_speakers,
            "source_language": request.source_language, "target_language": request.target_language,
            "glossary": request.glossary, "burn_subtitles": int(request.burn_subtitles), "dub": int(request.dub),
            "hide_source_subtitles": int(request.hide_source_subtitles),
            "speaker_gender_profile": request.speaker_gender_profile,
            "narrator_mode": 1,
            "tts_voice": request.tts_voice, "tts_secondary_voice": request.tts_voice,
            "voice_overrides": "{}",
            "original_audio_volume": request.original_audio_volume,
            "retry_count": 0,
            "received_bytes": received_bytes, "sha256": sha256,
            "created_at": now, "updated_at": now,
        }
        job_directory(job_id).mkdir(parents=True, exist_ok=True)
        row = db_create_job(values)
        if enqueue:
            self.queue.put(job_id)
        return row

    def submit(self, request: JobCreate) -> dict[str, Any]:
        return self._submit(request)

    def submit_upload(
        self, request: JobCreate, filename: str, stream: BinaryIO, *, enqueue: bool = True,
    ) -> dict[str, Any]:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in UPLOAD_EXTENSIONS:
            raise ValueError("Only MP4 uploads are supported")
        job_id = str(uuid.uuid4())
        directory = job_directory(job_id)
        directory.mkdir(parents=True)
        temporary = directory / ".source.uploading"
        output = directory / "source.mp4"
        total = 0
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as target:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > get_settings().max_upload_bytes:
                        raise ValueError("Uploaded video exceeds the 5 GiB limit")
                    digest.update(chunk)
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
            return self._submit(
                upload_request, job_id, received_bytes=total,
                sha256=digest.hexdigest(), enqueue=enqueue,
            )
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def enqueue(self, job_id: str) -> None:
        self.queue.put(job_id)

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return self.active_job_id == job_id

    def _max_job_retries(self) -> int:
        value = int(get_settings().job_auto_retries)
        return value if value >= 0 else 0

    def _is_retryable_job(self, row: dict[str, Any]) -> bool:
        if row["status"] != JobStatus.FAILED:
            return False
        max_retries = self._max_job_retries()
        if max_retries <= 0:
            return False
        retry_count = int(row.get("retry_count") or 0)
        return retry_count < max_retries

    def _mark_job_for_retry(self, job_id: str, error: str) -> bool:
        row = get_job(job_id)
        if not row:
            return False
        max_retries = self._max_job_retries()
        if max_retries <= 0:
            return False
        current = int(row.get("retry_count") or 0)
        if current >= max_retries:
            return False
        attempt = current + 1
        update_job(
            job_id,
            status=JobStatus.QUEUED,
            step=JobStep.QUEUED,
            progress_message=f"Retrying {attempt}/{max_retries} after transient failure",
            error=error,
            retry_count=attempt,
        )
        self.queue.put(job_id)
        return True

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
                message = str(exc)
                if not self._mark_job_for_retry(job_id, message):
                    update_job(job_id, status=JobStatus.FAILED, progress_message="Job failed", error=message)
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
            if not speakers:
                try:
                    recovered = _recover_speakers_from_overlapping_asr(
                        directory, video, subtitles, job["asr_model"], job["source_language_code"],
                        bool(job.get("diarize")), job.get("min_speakers"), job.get("max_speakers"),
                        lambda message: self._progress(job_id, JobStep.TRANSCRIBING, message),
                    )
                except Exception as exc:
                    logger.warning("job=%s failed to recover speakers from ASR overlap: %s", job_id, exc)
                else:
                    if recovered:
                        speakers = recovered
                        speakers_path.write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
                        self._progress(
                            job_id, JobStep.TRANSCRIBING,
                            f"Recovered {len(recovered)} speaker labels from overlap-aligned ASR",
                        )
        else:
            subtitles, speakers = [], {}
            if str(job["source_language_code"]).lower().startswith("zh"):
                self._progress(job_id, JobStep.TRANSCRIBING, "Reading burned-in Chinese subtitles with RapidOCR")
                try:
                    ocr = ocr_subtitles.extract_burned_subtitles(
                        video, source_srt, directory / "ocr-report.json",
                    )
                    if ocr:
                        subtitles, _ = ocr
                        report = json.loads((directory / "ocr-report.json").read_text(encoding="utf-8"))
                        low_confidence = _extract_low_confidence_ocr_indices(report)
                        if low_confidence:
                            self._progress(
                                job_id, JobStep.TRANSCRIBING,
                                f"OCR produced {len(subtitles)} cues, {len(low_confidence)} low-confidence entries",
                            )
                            subtitles, rescue_speakers = _asr_rescue_low_ocr_segments(
                                directory, video, subtitles, directory / "ocr-report.json", job["source_language_code"],
                                lambda msg: self._progress(job_id, JobStep.TRANSCRIBING, msg),
                            )
                            if rescue_speakers:
                                speakers.update(rescue_speakers)
                            write_srt(source_srt, subtitles)
                        self._progress(job_id, JobStep.TRANSCRIBING, f"RapidOCR produced {len(subtitles)} source subtitle cues")
                except Exception as exc:
                    logger.warning("job=%s OCR subtitle extraction skipped: %s", job_id, exc)
        if not subtitles:
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
        draft_provider = str(job.get("translation_draft_provider", job["provider"])).lower().strip()
        if draft_provider not in {"openai", "gemini", "deepseek"}:
            draft_provider = str(job["translation_refine_provider"]).lower().strip() if str(job["translation_refine_provider"]).lower().strip() in {"openai", "gemini", "deepseek"} else job["provider"]
        refine_provider = str(job["translation_refine_provider"]).lower().strip()
        if refine_provider == "auto":
            refine_provider = job["provider"]
        if refine_provider not in {"openai", "gemini", "deepseek"}:
            refine_provider = draft_provider

        self._progress(job_id, JobStep.TRANSLATING, f"Translating {len(subtitles)} cues via {draft_provider}")
        translated_srt = directory / "vi.srt"
        translation_fingerprint = stable_hash({
            "source": [(cue.index, cue.content) for cue in subtitles], "speakers": speakers,
            "draft_provider": draft_provider, "refine_provider": refine_provider, "provider": job["provider"],
            "target": job["target_language"], "glossary": job["glossary"],
            "speaker_gender_profile": job.get("speaker_gender_profile", "auto"),
            "pipeline": "romance-scene-batch-v6-speaker-aware",
        })
        if manifest.valid("translation", translation_fingerprint, [translated_srt, directory / "vi-final.json"]):
            self._progress(job_id, JobStep.TRANSLATING, "Resuming from context-edited Vietnamese translation")
            resumed_mapping = {
                int(item["id"]): str(item["text"])
                for item in json.loads((directory / "vi-final.json").read_text(encoding="utf-8"))
            }
            translated = [
                srt.Subtitle(cue.index, cue.start, cue.end, resumed_mapping.get(cue.index, ""))
                for cue in subtitles
            ]
        else:
            translator.start_ai_usage(directory, draft_provider)
            translated = translator.translate_subtitles(
                subtitles,
                job["provider"],
                job["source_language"],
                job["target_language"],
                job["glossary"],
                lambda message: self._progress(job_id, JobStep.TRANSLATING, message),
                speakers=speakers,
                job_dir=directory,
                speaker_gender_profile=job.get("speaker_gender_profile", "auto"),
                draft_provider=draft_provider,
                refine_provider=refine_provider,
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
            "speakers": speakers, "provider": refine_provider, "boxes": box_widths, "pipeline": "display-lines-v7-speaker-aware",
        })
        if manifest.valid("dialogue_master", master_fingerprint, [translated_srt, master_path]):
            master_payload = json.loads(master_path.read_text(encoding="utf-8"))
            display_mapping = {
                int(item["id"]): str(item.get("text", ""))
                for item in master_payload.get("display_lines", [])
            }
            translated = [
                srt.Subtitle(cue.index, cue.start, cue.end, display_mapping.get(cue.index, ""))
                for cue in subtitles
            ]
            self._progress(job_id, JobStep.TRANSLATING, "Resuming validated dialogue master")
        else:
            self._progress(job_id, JobStep.TRANSLATING, "Reflowing complete dialogue into original display lines")
            translated, master_utterances, master_warning = dialogue_master.build_dialogue_master(
                draft, subtitles, speakers, refine_provider, box_widths,
            )
            write_srt(translated_srt, translated)
            master_payload = {
                "warning": master_warning, "utterances": [item.to_dict() for item in master_utterances],
                "display_lines": [{"id": cue.index, "text": cue.content} for cue in translated],
            }
            atomic_write_json(master_path, master_payload)
            manifest.complete("dialogue_master", master_fingerprint, [translated_srt, master_path], {"utterances": len(master_utterances)})
        # When dubbing is enabled, subtitles are burned only after the real TTS
        # duration has retimed each display line. Burn-only jobs retain the
        # original source-aligned SRT behavior.
        if job["burn_subtitles"] and not job["dub"]:
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
            dub_video_base = video
            dubbed = directory / "vi-dubbed.mp4"
            base_stat = dub_video_base.stat()
            dub_fingerprint = stable_hash({"translation": [(cue.index, cue.content) for cue in translated], "master": master_payload, "video_base": [base_stat.st_size, base_stat.st_mtime_ns], "voice_policy": "single-narrator-v1", "voice": job["tts_voice"], "original_audio_volume": job["original_audio_volume"], "mix": "review-master-v6-minus13.5lufs-lra2.5-tp1", "prosody": "local-conservative-v2", "timing": "punctuated-sentence-v6-source-anchored-retimed-display", "burn": bool(job["burn_subtitles"]), "mask": [region.to_dict() for region in regions]})
            dub_outputs = [dubbed, directory / "speech-plan.json", directory / "prosody-plan.json", directory / "tts-timing.json", directory / "vi-aligned.srt"]
            if manifest.valid("dub", dub_fingerprint, dub_outputs):
                self._progress(job_id, JobStep.DUBBING, "Resuming from cached Vietnamese dub")
            else:
                self._progress(job_id, JobStep.DUBBING, "Creating Vietnamese natural dub")
                tts.create_dub(
                    dub_video_base, translated, directory, job["tts_voice"], job["original_audio_volume"],
                    lambda message: self._progress(job_id, JobStep.DUBBING, message),
                    speakers, refine_provider,
                    master_payload.get("utterances"),
                )
                if job["burn_subtitles"]:
                    self._progress(job_id, JobStep.BURNING, "Burning voice-aligned Vietnamese subtitles")
                    aligned_burned = directory / "vi-dubbed-aligned.mp4"
                    media.burn_subtitles(dubbed, directory / "vi-aligned.srt", aligned_burned, regions)
                    aligned_burned.replace(dubbed)
                manifest.complete("dub", dub_fingerprint, dub_outputs)
        self._progress(job_id, JobStep.QA, "Running automatic content, timing, and media QA")
        final_video = directory / "vi-dubbed.mp4" if job["dub"] else directory / "vi-burned.mp4" if job["burn_subtitles"] else video
        # Normalize the serialized subtitle artifacts from the authoritative
        # in-memory cues before QA. This self-heals files written by older code
        # that renumbered ids after an intentionally empty ASR-noise cue.
        write_srt(translated_srt, translated)
        if job["dub"] and (directory / "tts-timing.json").is_file():
            timing = json.loads((directory / "tts-timing.json").read_text(encoding="utf-8"))
            write_srt(directory / "vi-aligned.srt", tts.retime_subtitles_to_tts(translated, timing))
        report = qa.validate_job(directory, translated, [cue.index for cue in subtitles], final_video)
        manifest.complete("qa", stable_hash({"translation": [(cue.index, cue.content) for cue in translated], "master": master_payload, "output_bytes": final_video.stat().st_size}), [directory / "qa-report.json"], report["summary"])
        status = JobStatus.NEEDS_REVIEW if report["summary"]["error"] else JobStatus.DONE
        # A review job must retain its source for rerendering. A successful
        # translated job keeps only the actual deliverable; uploaded originals
        # and per-utterance WAV files otherwise accumulate indefinitely.
        if status == JobStatus.DONE and final_video != video:
            removed = cleanup_completed_job_media(directory, final_video)
            if removed:
                logger.info("job=%s storage cleanup removed=%s", job_id, removed)
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
