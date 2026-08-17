import asyncio
import json
import re
import subprocess
import shutil
from pathlib import Path

import edge_tts
import srt
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from .media import mux_delayed_clips, probe_duration
from .config import get_settings
from .voice_profiles import classify_speaker_voices
from .artifacts import atomic_write_json, stable_hash
from .speech_plan import build_speech_plans, predict_duration_ms, punctuation_pause_after_ms
from .prosody import PROSODY_VERSION, plan_prosody

MAX_NATURAL_TEMPO = 1.18


def sentence_aligned_utterances(
    subtitles: list[srt.Subtitle], speakers: dict[int, str],
) -> list[tuple[srt.Subtitle, str | None, list[int]]]:
    """Join display lines until real punctuation completes the spoken phrase.

    SRT line wrapping is visual, not linguistic.  Synthesizing every display
    line separately makes the voice stop between fragments such as
    ``Nếu tiền bối`` / ``muốn hại ta,``.  Keep the original first/last timing,
    but send the complete punctuated phrase to TTS in one inference call.
    """
    if not subtitles:
        return []
    result: list[tuple[srt.Subtitle, str | None, list[int]]] = []
    current: list[srt.Subtitle] = []
    for cue in subtitles:
        if current:
            previous_complete = bool(re.search(r'[.!?…]["”’)]?\s*$', current[-1].content.strip()))
            # SRT timing gaps and visual line wraps are not linguistic stops.
            # Only punctuation in the actual dialogue may end a spoken phrase.
            if previous_complete:
                first, last = current[0], current[-1]
                result.append((
                    srt.Subtitle(first.index, first.start, last.end, " ".join(item.content.strip() for item in current)),
                    speakers.get(first.index), [item.index for item in current],
                ))
                current = []
        current.append(cue)
    first, last = current[0], current[-1]
    result.append((
        srt.Subtitle(first.index, first.start, last.end, " ".join(item.content.strip() for item in current)),
        speakers.get(first.index), [item.index for item in current],
    ))
    return result


def retime_subtitles_to_tts(
    subtitles: list[srt.Subtitle], timing: list[dict],
) -> list[srt.Subtitle]:
    """Place display-line boundaries inside the audio that actually speaks them.

    Each sentence is synthesized naturally as one clip. Its constituent visual
    lines are then distributed across the measured clip duration using spoken
    syllables and punctuation as weights. This keeps the text change close to
    the words being heard instead of retaining unrelated source-ASR timings.
    """
    expected_ids = [cue.index for cue in subtitles]
    timed_ids = [item_id for item in timing for item_id in item.get("cue_ids", [])]
    if timed_ids != expected_ids:
        raise ValueError("TTS timing must cover every subtitle id exactly once and in order")
    by_id = {cue.index: cue for cue in subtitles}
    aligned: list[srt.Subtitle] = []
    from datetime import timedelta

    for phrase in timing:
        cue_ids = [int(value) for value in phrase["cue_ids"]]
        phrase_start_ms = int(phrase["scheduled_start_ms"])
        phrase_duration_ms = max(len(cue_ids), int(phrase["fitted_ms"]))
        cues = [by_id[item_id] for item_id in cue_ids]
        weights = [max(1, predict_duration_ms(cue.content)) for cue in cues]
        total_weight = sum(weights)
        elapsed_weight = 0
        phrase_aligned = []
        for index, (cue, weight) in enumerate(zip(cues, weights, strict=True)):
            start_ms = phrase_start_ms + round(phrase_duration_ms * elapsed_weight / total_weight)
            elapsed_weight += weight
            end_ms = (
                phrase_start_ms + phrase_duration_ms
                if index == len(cues) - 1
                else phrase_start_ms + round(phrase_duration_ms * elapsed_weight / total_weight)
            )
            end_ms = max(start_ms + 1, end_ms)
            phrase_aligned.append({"id": cue.index, "start_ms": start_ms, "end_ms": end_ms})
            aligned.append(srt.Subtitle(
                cue.index, timedelta(milliseconds=start_ms), timedelta(milliseconds=end_ms), cue.content,
            ))
        phrase["aligned_cues"] = phrase_aligned
    return aligned


def _trim_edge_silence(audio: AudioSegment) -> AudioSegment:
    """Remove only long silent padding added by TTS, preserving natural breaths."""
    if not audio or audio.dBFS == float("-inf"):
        return audio
    ranges = detect_nonsilent(
        audio, min_silence_len=120, silence_thresh=max(-48.0, audio.dBFS - 28.0), seek_step=5,
    )
    if not ranges:
        return audio
    start = max(0, ranges[0][0] - 40)
    end = min(len(audio), ranges[-1][1] + 60)
    return audio[start:end]


def _available_speech_window_ms(plan) -> int:
    """Use real silence before the next utterance while reserving punctuation pauses."""
    deadline_window = plan.hard_deadline_ms - plan.pause_before_ms - plan.pause_after_ms
    return max(plan.target_duration_ms, deadline_window)


def _fit_audio_to_window(audio: AudioSegment, target_ms: int, work_path: Path) -> AudioSegment:
    """Pitch-preserving time stretch, bounded to keep the voice natural."""
    if target_ms <= 0 or not audio:
        return audio
    tempo = len(audio) / target_ms
    # Narration may finish before the visual window; that silence is natural.
    # Slowing it down to fill every millisecond destroys emphasis and cadence.
    if tempo <= 1.03:
        return audio
    tempo = min(MAX_NATURAL_TEMPO, tempo)
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
        key = stable_hash({"text": item["text"], "style": item.get("style", "doc_truyen"), "voice": voice,
                           "reference": reference_stamp, "engine": "vieneu-v3-turbo-pause-v1", "prosody": PROSODY_VERSION})
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
    provider: str = "deepseek",
    master_utterances: list[dict] | None = None,
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
    # Display-line boundaries must not create artificial TTS pauses. Build
    # complete phrases from the final punctuated SRT, without trusting the
    # dialogue master's punctuation-normalized full_text.
    sentence_units = sentence_aligned_utterances(subtitles, speakers)
    utterances = [(cue, speaker) for cue, speaker, _ in sentence_units]
    plans = build_speech_plans(utterances, video_duration_ms)
    prosody_warning = None
    if narrator_mode:
        plans, prosody_warning = plan_prosody(plans, provider)
    for plan in plans:
        # Provider direction may add a longer dramatic pause, but it must not
        # erase the deterministic breathing room required by terminal punctuation.
        plan.pause_after_ms = max(plan.pause_after_ms, punctuation_pause_after_ms(plan.spoken_text))
    atomic_write_json(job_dir / "prosody-plan.json", {
        "version": PROSODY_VERSION, "warning": prosody_warning,
        "items": [plan.to_dict() for plan in plans],
    })
    atomic_write_json(job_dir / "speech-plan.json", [plan.to_dict() for plan in plans])
    if narrator_mode:
        _synthesize_local_batch([{"id": plan.id, "text": plan.spoken_text, "style": plan.style} for plan in plans], voice, clips_dir)
    fitted_dir = clips_dir / "fitted"
    fitted_dir.mkdir(exist_ok=True)
    delayed_clips: list[tuple[Path, int]] = []
    timing_report: list[dict] = []
    timeline_cursor_ms = 0
    for number, (cue, speaker) in enumerate(utterances, 1):
        plan = plans[number - 1]
        cue_ids = sentence_units[number - 1][2]
        clip_path = clips_dir / f"{cue.index:06d}.{'wav' if narrator_mode else 'mp3'}"
        cue_voice = profiles.get(speaker, {}).get("voice", voice)
        if not narrator_mode:
            asyncio.run(_synthesize(cue, cue_voice, clip_path, narrator_mode))
        audio = AudioSegment.from_file(clip_path)
        raw_original_ms = len(audio)
        audio = _trim_edge_silence(audio)
        original_ms = len(audio)
        target_window = _available_speech_window_ms(plan)
        required_tempo = original_ms / max(1, target_window)
        audio = _fit_audio_to_window(audio, target_window, clip_path)
        fitted_path = fitted_dir / f"{cue.index:06d}.wav"
        audio.set_frame_rate(48000).set_channels(1).set_sample_width(2).export(fitted_path, format="wav")
        # Never overlap two narrated sentences. If a translated phrase is
        # longer than its source window, move the next phrase (and its retimed
        # subtitles) forward instead of speaking both at once.
        scheduled_start_ms = max(plan.start_ms + plan.pause_before_ms, timeline_cursor_ms)
        delayed_clips.append((fitted_path, scheduled_start_ms))
        timeline_cursor_ms = scheduled_start_ms + len(audio) + plan.pause_after_ms
        timing_report.append({
            "id": plan.id, "predicted_ms": plan.predicted_duration_ms, "raw_original_ms": raw_original_ms,
            "original_ms": original_ms,
            "fitted_ms": len(audio), "target_ms": plan.target_duration_ms, "hard_deadline_ms": plan.hard_deadline_ms,
            "available_window_ms": target_window, "required_tempo": round(required_tempo, 4),
            "tempo": round(original_ms / max(1, len(audio)), 4),
            "overflow_ms": max(0, scheduled_start_ms + len(audio) - (plan.start_ms + plan.hard_deadline_ms)),
            "emotion": plan.emotion, "intensity": plan.intensity, "style": plan.style,
            "pause_before_ms": plan.pause_before_ms, "pause_after_ms": plan.pause_after_ms,
            "source_start_ms": plan.start_ms, "scheduled_start_ms": scheduled_start_ms,
            "schedule_shift_ms": scheduled_start_ms - plan.start_ms,
            "alignment_mode": "punctuated_sentence",
            "cue_ids": cue_ids,
        })
        if progress and (number == len(utterances) or number % 5 == 0):
            progress(f"Synthesized {number}/{len(utterances)} natural utterances")
    atomic_write_json(job_dir / "tts-timing.json", timing_report)
    aligned_subtitles = retime_subtitles_to_tts(subtitles, timing_report)
    (job_dir / "vi-aligned.srt").write_text(srt.compose(aligned_subtitles), encoding="utf-8")
    # Persist the cue allocation added by retime_subtitles_to_tts.
    atomic_write_json(job_dir / "tts-timing.json", timing_report)
    output = job_dir / "vi-dubbed.mp4"
    mux_delayed_clips(video, delayed_clips, output, original_audio_volume)
    return output
