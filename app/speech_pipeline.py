import json
import os
import subprocess
from pathlib import Path
from typing import Callable

import srt

from .subtitle import seconds_to_timedelta, write_srt

WHISPERX_BIN = Path("/workspace/third_party/pyvideotrans/.venv/bin/whisperx")
LOCAL_LARGE_V3 = Path("/workspace/third_party/pyvideotrans/models/models--Systran--faster-whisper-large-v3")


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
    if result.returncode:
        raise RuntimeError(f"WhisperX failed: {result.stderr.strip()[-4000:]}")
    json_path = work_dir / f"{video.stem}.json"
    if not json_path.is_file():
        raise RuntimeError("WhisperX did not produce aligned JSON")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    subtitles, speakers = _group_words(payload.get("word_segments", []))
    if not subtitles:
        raise RuntimeError("WhisperX returned no aligned speech")
    write_srt(output_srt, subtitles)
    (video.parent / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=2), encoding="utf-8")
    return subtitles, speakers
