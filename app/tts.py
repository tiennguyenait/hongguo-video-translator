import asyncio
import json
import re
import subprocess
from pathlib import Path

import edge_tts
import srt
from pydub import AudioSegment

from .media import mux_delayed_clips, probe_duration
from .config import get_settings
from .voice_profiles import classify_speaker_voices


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


def _synthesize_local_batch(utterances: list[tuple[srt.Subtitle, str | None]], voice: str, clips_dir: Path) -> None:
    settings = get_settings()
    if not settings.vieneu_python.is_file() or not settings.vieneu_runner.is_file():
        raise RuntimeError("VieNeu local TTS is not installed")
    local_voice = voice if not voice.startswith("vi-VN-") else "Ngọc Linh"
    manifest = clips_dir / "vieneu-input.json"
    manifest.write_text(
        json.dumps(
            [{"id": cue.index, "text": cue.content} for cue, _ in utterances],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            str(settings.vieneu_python), str(settings.vieneu_runner), str(manifest), str(clips_dir), local_voice,
            str(settings.narrator_reference) if settings.narrator_reference.is_file() else "",
        ],
        text=True, capture_output=True, timeout=1800,
    )
    if result.returncode:
        raise RuntimeError(f"VieNeu local TTS failed: {result.stderr.strip()[-3000:]}")
    missing = [cue.index for cue, _ in utterances if not (clips_dir / f"{cue.index:06d}.wav").is_file()]
    if missing:
        raise RuntimeError(f"VieNeu did not create clips: {missing}")


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
    if narrator_mode:
        _synthesize_local_batch(utterances, voice, clips_dir)
    fitted_dir = clips_dir / "fitted"
    fitted_dir.mkdir(exist_ok=True)
    delayed_clips: list[tuple[Path, int]] = []
    for number, (cue, speaker) in enumerate(utterances, 1):
        clip_path = clips_dir / f"{cue.index:06d}.{'wav' if narrator_mode else 'mp3'}"
        cue_voice = profiles.get(speaker, {}).get("voice", voice)
        if not narrator_mode:
            asyncio.run(_synthesize(cue, cue_voice, clip_path, narrator_mode))
        audio = AudioSegment.from_file(clip_path)
        next_start = int(utterances[number][0].start.total_seconds() * 1000) if number < len(utterances) else video_duration_ms
        available_ms = max(1, next_start - int(cue.start.total_seconds() * 1000))
        speech_window_ms = max(250, int((cue.end - cue.start).total_seconds() * 1000))
        audio = _fit_audio_to_window(audio, min(available_ms, speech_window_ms), clip_path)
        fitted_path = fitted_dir / f"{cue.index:06d}.wav"
        audio.set_frame_rate(48000).set_channels(1).set_sample_width(2).export(fitted_path, format="wav")
        delayed_clips.append((fitted_path, int(cue.start.total_seconds() * 1000)))
        if progress and (number == len(utterances) or number % 5 == 0):
            progress(f"Synthesized {number}/{len(utterances)} natural utterances")
    output = job_dir / "vi-dubbed.mp4"
    mux_delayed_clips(video, delayed_clips, output, original_audio_volume)
    return output
