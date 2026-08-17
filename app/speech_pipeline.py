import json
import os
import subprocess
from pathlib import Path
from typing import Callable

import srt

from .subtitle import seconds_to_timedelta, write_srt

WHISPERX_BIN = Path("/workspace/third_party/pyvideotrans/.venv/bin/whisperx")
LOCAL_LARGE_V3 = Path("/workspace/third_party/pyvideotrans/models/models--Systran--faster-whisper-large-v3")


def _transcript_metrics(payload: dict, video_duration: float) -> dict[str, float | int]:
    """Measure aligned speech conservatively without double-counting overlaps."""
    intervals: list[tuple[float, float]] = []
    units = 0
    for word in payload.get("word_segments", []):
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(word.get("word", "")).strip()
        if not text or end <= start:
            continue
        intervals.append((max(0.0, start), min(video_duration, end)))
        units += len(text.replace(" ", ""))
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    speech_seconds = sum(end - start for start, end in merged)
    return {
        "speech_seconds": speech_seconds,
        "coverage": speech_seconds / video_duration if video_duration > 0 else 0.0,
        "units": units,
        "words": len(intervals),
    }


def _catastrophically_sparse(payload: dict, video_duration: float) -> bool:
    """Flag only near-empty ASR on a substantial video, not naturally quiet scenes."""
    if video_duration < 30:
        return False
    metrics = _transcript_metrics(payload, video_duration)
    return bool(
        metrics["speech_seconds"] < max(0.75, video_duration * 0.015)
        and metrics["units"] < 4
    )


def _group_words(words: list[dict]) -> tuple[list[srt.Subtitle], dict[int, str]]:
    cleaned: list[dict] = []
    for raw in words:
        if "start" not in raw or "end" not in raw or not str(raw.get("word", "")).strip():
            continue
        word = dict(raw)
        # Alignment occasionally stretches one Chinese character across a long
        # silence.  Keeping that end time makes captions and TTS pause mid-word.
        word["end"] = min(float(word["end"]), float(word["start"]) + 1.2)
        cleaned.append(word)

    # Remove isolated one-word speaker flips. Pyannote boundaries and WhisperX
    # word boundaries are not sample-identical, so these are usually boundary
    # noise rather than a real 100 ms speaker turn.
    for index in range(1, len(cleaned) - 1):
        previous, current, following = cleaned[index - 1:index + 2]
        if (
            previous.get("speaker")
            and previous.get("speaker") == following.get("speaker")
            and current.get("speaker") != previous.get("speaker")
            and current["end"] - current["start"] < 0.35
        ):
            current["speaker"] = previous["speaker"]

    groups: list[list[dict]] = []
    current: list[dict] = []
    for word in cleaned:
        current_speakers = [item.get("speaker") for item in current if item.get("speaker")]
        current_speaker = max(set(current_speakers), key=current_speakers.count) if current_speakers else None
        speaker_change = bool(
            current and word.get("speaker") and current_speaker
            and word["speaker"] != current_speaker
            and word["end"] - word["start"] >= 0.35
        )
        split = bool(
            current
            and (
                word["start"] - current[-1]["end"] > 0.45
                or word["end"] - current[0]["start"] > 4.5
                or sum(len(str(item["word"])) for item in current) >= 18
                or speaker_change
            )
        )
        if split:
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    subtitles, speakers = [], {}
    for index, group in enumerate(groups, 1):
        content = "".join(str(item["word"]) for item in group).strip()
        cue = srt.Subtitle(
            index=index,
            start=seconds_to_timedelta(float(group[0]["start"])),
            end=seconds_to_timedelta(float(group[-1]["end"])),
            content=content,
        )
        subtitles.append(cue)
        labels = [item.get("speaker") for item in group if item.get("speaker")]
        if labels:
            speakers[index] = max(set(labels), key=labels.count)
    return subtitles, speakers


def transcribe_aligned(
    video: Path,
    output_srt: Path,
    model_name: str,
    language: str,
    diarize: bool,
    min_speakers: int | None,
    max_speakers: int | None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[srt.Subtitle], dict[int, str]]:
    if not WHISPERX_BIN.is_file():
        raise RuntimeError("WhisperX environment is not installed")
    work_dir = video.parent / "whisperx"
    work_dir.mkdir(exist_ok=True)
    duration_result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        text=True, capture_output=True,
    )
    if duration_result.returncode or not duration_result.stdout.strip():
        raise RuntimeError(f"Cannot measure source duration for ASR coverage: {duration_result.stderr.strip()}")
    video_duration = float(duration_result.stdout.strip())
    model = str(LOCAL_LARGE_V3) if model_name == "large-v3" and LOCAL_LARGE_V3.is_dir() else model_name
    command = [
        str(WHISPERX_BIN), str(video), "--model", model, "--language", language,
        "--device", "cuda", "--compute_type", "float16", "--batch_size", "8",
        "--vad_method", "silero", "--output_format", "json", "--output_dir", str(work_dir),
        "--verbose", "False", "--print_progress", "True", "--segment_resolution", "sentence",
    ]
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if diarize and hf_token:
        command += ["--diarize", "--hf_token", hf_token]
        if min_speakers:
            command += ["--min_speakers", str(min_speakers)]
        if max_speakers:
            command += ["--max_speakers", str(max_speakers)]
    elif diarize and progress:
        progress("HF_TOKEN is missing; continuing with aligned ASR without speaker diarization")
    if progress:
        progress("Running WhisperX VAD, ASR and forced alignment")
    result = subprocess.run(command, text=True, capture_output=True, timeout=3600)
    gated_error = "GatedRepoError" in result.stderr or "Cannot access gated repo" in result.stderr
    if result.returncode and diarize and hf_token and gated_error:
        if progress:
            progress("Pyannote model access is not accepted; retrying aligned ASR without diarization")
        fallback = [part for part in command]
        diarize_index = fallback.index("--diarize")
        fallback = fallback[:diarize_index]
        result = subprocess.run(fallback, text=True, capture_output=True, timeout=3600)
        if not result.returncode:
            # Reuse the proven non-diarized command if sparse coverage also
            # requires a sensitive-VAD retry; otherwise the gated flags would
            # be accidentally reintroduced.
            command = fallback
    if result.returncode:
        raise RuntimeError(f"WhisperX failed: {result.stderr.strip()[-4000:]}")
    json_path = work_dir / f"{video.stem}.json"
    if not json_path.is_file():
        raise RuntimeError("WhisperX did not produce aligned JSON")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if _catastrophically_sparse(payload, video_duration):
        first_payload = payload
        first_metrics = _transcript_metrics(first_payload, video_duration)
        if progress:
            progress(
                "WhisperX transcript coverage is suspiciously sparse "
                f"({first_metrics['speech_seconds']:.2f}s/{video_duration:.2f}s); "
                "retrying with sensitive VAD"
            )
        sensitive_command = command + ["--vad_onset", "0.20", "--vad_offset", "0.15"]
        sensitive_result = subprocess.run(sensitive_command, text=True, capture_output=True, timeout=3600)
        if sensitive_result.returncode:
            raise RuntimeError(
                "WhisperX sensitive-VAD retry failed after sparse transcript: "
                f"{sensitive_result.stderr.strip()[-4000:]}"
            )
        sensitive_payload = json.loads(json_path.read_text(encoding="utf-8"))
        sensitive_metrics = _transcript_metrics(sensitive_payload, video_duration)
        payload = max(
            (first_payload, sensitive_payload),
            key=lambda item: (
                _transcript_metrics(item, video_duration)["units"],
                _transcript_metrics(item, video_duration)["speech_seconds"],
            ),
        )
        json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        best_metrics = _transcript_metrics(payload, video_duration)
        if progress:
            progress(
                "Sensitive-VAD ASR coverage: "
                f"{best_metrics['speech_seconds']:.2f}s speech, {best_metrics['units']} text units"
            )
        if _catastrophically_sparse(payload, video_duration):
            raise RuntimeError(
                "ASR coverage remained catastrophically sparse after sensitive-VAD retry: "
                f"{best_metrics['speech_seconds']:.2f}s speech across a {video_duration:.2f}s video. "
                "Manual review or a different ASR strategy is required; refusing to render a mostly undubbed video."
            )
    subtitles, speakers = _group_words(payload.get("word_segments", []))
    if not subtitles:
        raise RuntimeError("WhisperX returned no aligned speech")
    write_srt(output_srt, subtitles)
    (video.parent / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    return subtitles, speakers
