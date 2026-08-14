import asyncio
import json
import re
import subprocess
import shutil
from pathlib import Path

import edge_tts
import srt
from pydub import AudioSegment

from .media import mux_delayed_clips, probe_duration
from .config import get_settings
from .voice_profiles import classify_speaker_voices
from .artifacts import atomic_write_json, stable_hash
from .speech_plan import build_speech_plans


def _fit_audio_to_window(audio: AudioSegment, target_ms: int, work_path: Path) -> AudioSegment:
    """Pitch-preserving time stretch, bounded to keep the voice natural."""
    if target_ms <= 0 or not audio:
        return audio
    tempo = len(audio) / target_ms
    # Narration may finish before the visual window; that silence is natural.
    # Slowing it down to fill every millisecond destroys emphasis and cadence.
    if tempo <= 1.03:
        return audio
    tempo = min(1.22, tempo)
    input_path = work_path.with_suffix(".timing-input.wav")
    output_path = work_path.with_suffix(".timing.wav")
    audio.export(input_path, format="wav")
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(input_path), "-filter:a", f"atempo={tempo:.5f}", str(output_path)],
        text=True, capture_output=True,
    )
    input_path.unlink(missing_ok=True)
    if result.returncode:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot fit TTS timing: {result.stderr.strip()}")
    fitted = AudioSegment.from_file(output_path)
    output_path.unlink(missing_ok=True)
    return fitted.fade_in(20).fade_out(35)


async def _synthesize(cue: srt.Subtitle, voice: str, path: Path, narrator_mode: bool = False) -> None:
    last_error = None
    for attempt in range(1, 4):
        try:
            path.unlink(missing_ok=True)
            clean_text = re.sub(r"\s+", " ", cue.content).strip()
            await asyncio.wait_for(
                edge_tts.Communicate(
                    clean_text, voice,
                    rate="+4%" if narrator_mode else "+10%",
                    pitch="-8Hz" if narrator_mode else "+0Hz",
                ).save(str(path)),
                timeout=45,
            )
            if path.is_file() and path.stat().st_size >= 512:
                return
            raise RuntimeError("Edge TTS returned an empty or truncated MP3")
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                await asyncio.sleep(attempt * 2)
    raise RuntimeError(f"Edge TTS failed for cue {cue.index} after 3 attempts: {last_error}")


def build_utterances(subtitles: list[srt.Subtitle], speakers: dict[int, str]) -> list[tuple[srt.Subtitle, str | None]]:
    """Join adjacent cues from one speaker so TTS keeps sentence-level prosody."""
    utterances: list[tuple[srt.Subtitle, str | None]] = []
    current: list[srt.Subtitle] = []
    current_speaker = None
    for cue in subtitles:
        speaker = speakers.get(cue.index)
        previous_is_continuation = bool(current and re.search(r'(?:\.{3,}|…+)\s*$', current[-1].content.strip()))
        previous_is_complete = bool(
            current and not previous_is_continuation
            and re.search(r'[.!?。！？]["”’)]?\s*$', current[-1].content.strip())
        )
        can_join = bool(
            current
            and speaker == current_speaker
            and not previous_is_complete
            and (cue.start - current[-1].end).total_seconds() <= 0.55
            and (cue.end - current[0].start).total_seconds() <= (10.0 if previous_is_continuation else 6.5)
            and sum(len(item.content) for item in current) + len(cue.content) <= 90
        )
        if current and not can_join:
            content = " ".join(item.content for item in current)
            content = re.sub(r"\s*(?:\.{3,}|…+)\s*", " ", content).strip()
            utterances.append((srt.Subtitle(index=current[0].index, start=current[0].start, end=current[-1].end, content=content), current_speaker))
            current = []
        if not current:
            current_speaker = speaker
        current.append(cue)
    if current:
        content = " ".join(item.content for item in current)
        content = re.sub(r"\s*(?:\.{3,}|…+)\s*", " ", content).strip()
        utterances.append((srt.Subtitle(index=current[0].index, start=current[0].start, end=current[-1].end, content=content), current_speaker))
    return utterances


def _synthesize_local_batch(items: list[dict], voice: str, clips_dir: Path) -> None:
    settings = get_settings()
    if not settings.vieneu_python.is_file() or not settings.vieneu_runner.is_file():
        raise RuntimeError("VieNeu local TTS is not installed")
    local_voice = voice if not voice.startswith("vi-VN-") else "Ngọc Linh"
    manifest = clips_dir / "vieneu-input.json"
    reference_stamp = None
    if settings.narrator_reference.is_file():
        stat = settings.narrator_reference.stat()
        reference_stamp = [stat.st_size, stat.st_mtime_ns]
    cache_keys: dict[int, str] = {}
    missing: list[dict] = []
    for item in items:
        key = stable_hash({"text": item["text"], "voice": voice, "reference": reference_stamp, "engine": "vieneu-v3-turbo"})
        cache_keys[item["id"]] = key
        cached = settings.tts_cache_dir / f"{key}.wav"
        output = clips_dir / f"{item['id']:06d}.wav"
        if cached.is_file() and cached.stat().st_size > 512:
            shutil.copy2(cached, output)
        else:
            missing.append(item)
    if not missing:
        return
    manifest.write_text(json.dumps(missing, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [
            str(settings.vieneu_python), str(settings.vieneu_runner), str(manifest), str(clips_dir), local_voice,
            str(settings.narrator_reference) if settings.narrator_reference.is_file() else "",
        ],
        text=True, capture_output=True, timeout=1800,
    )
    if result.returncode:
        raise RuntimeError(f"VieNeu local TTS failed: {result.stderr.strip()[-3000:]}")
    missing_ids = [item["id"] for item in items if not (clips_dir / f"{item['id']:06d}.wav").is_file()]
    if missing_ids:
        raise RuntimeError(f"VieNeu did not create clips: {missing_ids}")
    for item in missing:
        output = clips_dir / f"{item['id']:06d}.wav"
        cached = settings.tts_cache_dir / f"{cache_keys[item['id']]}.wav"
        if not cached.exists():
            shutil.copy2(output, cached)


def create_dub(
    video: Path, subtitles: list[srt.Subtitle], job_dir: Path, voice: str,
    original_audio_volume: float, progress=None, speakers: dict[int, str] | None = None,
    secondary_voice: str = "vi-VN-NamMinhNeural",
    voice_overrides: dict[str, str] | None = None,
    narrator_mode: bool = True,
) -> Path:
    clips_dir = job_dir / "tts"
    clips_dir.mkdir(exist_ok=True)
    # Remove legacy full-length PCM timelines from pre-FFmpeg jobs.
    (job_dir / "vi-dub.wav").unlink(missing_ok=True)
    video_duration_ms = int(probe_duration(video) * 1000)
    speakers = speakers or {}
    if narrator_mode:
        profiles = {
            label: {"mode": "single_narrator", "voice": voice, "overridden": False}
            for label in sorted(set(speakers.values()))
        }
        (job_dir / "voice-profiles.json").write_text(
            __import__("json").dumps(profiles or {"NARRATOR": {"mode": "single_narrator", "voice": voice}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        profiles = classify_speaker_voices(video, subtitles, speakers, voice, secondary_voice, voice_overrides)
    utterances = build_utterances(subtitles, speakers)
    plans = build_speech_plans(utterances, video_duration_ms)
    atomic_write_json(job_dir / "speech-plan.json", [plan.to_dict() for plan in plans])
    if narrator_mode:
        _synthesize_local_batch([{"id": plan.id, "text": plan.spoken_text} for plan in plans], voice, clips_dir)
    fitted_dir = clips_dir / "fitted"
    fitted_dir.mkdir(exist_ok=True)
    delayed_clips: list[tuple[Path, int]] = []
    timing_report: list[dict] = []
    for number, (cue, speaker) in enumerate(utterances, 1):
        plan = plans[number - 1]
        clip_path = clips_dir / f"{cue.index:06d}.{'wav' if narrator_mode else 'mp3'}"
        cue_voice = profiles.get(speaker, {}).get("voice", voice)
        if not narrator_mode:
            asyncio.run(_synthesize(cue, cue_voice, clip_path, narrator_mode))
        audio = AudioSegment.from_file(clip_path)
        original_ms = len(audio)
        target_window = max(plan.target_duration_ms, min(plan.hard_deadline_ms, round(plan.target_duration_ms * 1.18)))
        audio = _fit_audio_to_window(audio, target_window, clip_path)
        fitted_path = fitted_dir / f"{cue.index:06d}.wav"
        audio.set_frame_rate(48000).set_channels(1).set_sample_width(2).export(fitted_path, format="wav")
        delayed_clips.append((fitted_path, plan.start_ms))
        timing_report.append({
            "id": plan.id, "predicted_ms": plan.predicted_duration_ms, "original_ms": original_ms,
            "fitted_ms": len(audio), "target_ms": plan.target_duration_ms, "hard_deadline_ms": plan.hard_deadline_ms,
            "tempo": round(original_ms / max(1, len(audio)), 4), "overflow_ms": max(0, len(audio) - plan.hard_deadline_ms),
        })
        if progress and (number == len(utterances) or number % 5 == 0):
            progress(f"Synthesized {number}/{len(utterances)} natural utterances")
    atomic_write_json(job_dir / "tts-timing.json", timing_report)
    output = job_dir / "vi-dubbed.mp4"
    mux_delayed_clips(video, delayed_clips, output, original_audio_volume)
    return output
