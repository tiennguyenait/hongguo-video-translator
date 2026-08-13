import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import librosa
import srt


def _extract_mono_audio(video: Path, wav_path: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot extract audio for voice profiling: {result.stderr.strip()}")


def _median_pitch(samples: np.ndarray, sample_rate: int) -> float | None:
    if len(samples) < sample_rate * 0.35:
        return None
    audio = samples.astype(np.float32) / 32768.0
    f0, voiced, probability = librosa.pyin(audio, fmin=70, fmax=350, sr=sample_rate, frame_length=2048, hop_length=320)
    valid = f0[voiced & (probability >= 0.70) & np.isfinite(f0)]
    return float(np.median(valid)) if len(valid) >= 5 else None


def classify_speaker_voices(
    video: Path,
    subtitles: list[srt.Subtitle],
    speakers: dict[int, str],
    primary_voice: str,
    secondary_voice: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, dict]:
    labels = sorted(set(speakers.values()))
    if not labels:
        return {}
    wav_path = video.parent / "speaker-analysis.wav"
    _extract_mono_audio(video, wav_path)
    with wave.open(str(wav_path), "rb") as source:
        sample_rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
    male_voice = primary_voice if "NamMinh" in primary_voice else secondary_voice
    female_voice = primary_voice if "HoaiMy" in primary_voice else secondary_voice
    profiles: dict[str, dict] = {}
    cue_by_id = {cue.index: cue for cue in subtitles}
    for label in labels:
        intervals = [cue_by_id[index] for index, speaker in speakers.items() if speaker == label and index in cue_by_id]
        intervals = sorted(intervals, key=lambda cue: (cue.end - cue.start), reverse=True)[:8]
        pitches = []
        for cue in intervals:
            start = max(0, int(cue.start.total_seconds() * sample_rate))
            end = min(len(samples), int(cue.end.total_seconds() * sample_rate))
            if end > start:
                value = _median_pitch(samples[start:end], sample_rate)
                if value is not None:
                    pitches.append(value)
        pitch = float(np.median(pitches)) if pitches else None
        male_votes = sum(value < 165 for value in pitches)
        female_votes = sum(value >= 190 for value in pitches)
        agreement = max(male_votes, female_votes) / len(pitches) if pitches else 0.0
        # Pitch is only an advisory signal. Require multiple independent turns
        # and strong agreement; otherwise preserve a distinct fallback voice but
        # report the profile as uncertain instead of inventing a gender.
        if len(pitches) < 2 or agreement < 0.70 or (pitch is not None and 165 <= pitch < 190):
            predicted = "uncertain"
            selected_voice = primary_voice if labels.index(label) % 2 == 0 else secondary_voice
        else:
            predicted = "female" if female_votes > male_votes else "male"
            selected_voice = female_voice if predicted == "female" else male_voice
        if overrides and label in overrides:
            selected_voice = overrides[label]
        profiles[label] = {
            "predicted_voice_type": predicted,
            "confidence": round(agreement, 2) if pitches else 0.0,
            "evidence_turns": len(pitches),
            "median_pitch_hz": round(pitch, 1) if pitch else None,
            "voice": selected_voice,
            "overridden": bool(overrides and label in overrides),
        }
    (video.parent / "voice-profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    return profiles
