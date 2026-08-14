"""Deterministic speech timing plan produced before expensive TTS."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import srt

from .text_normalizer import normalize_spoken_text


@dataclass(slots=True)
class SpeechPlan:
    id: int
    subtitle_text: str
    spoken_text: str
    start_ms: int
    target_duration_ms: int
    hard_deadline_ms: int
    predicted_duration_ms: int
    emotion: str
    pace: float
    intensity: float = 0.45
    style: str = "doc_truyen"
    pause_before_ms: int = 0
    pause_after_ms: int = 100
    prosody_source: str = "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


def predict_duration_ms(text: str) -> int:
    syllables = len(re.findall(r"\b[\wÀ-ỹ]+\b", text, re.UNICODE))
    punctuation_pause = 180 * len(re.findall(r"[,;:]", text)) + 320 * len(re.findall(r"[.!?]", text))
    return max(350, round(syllables / 3.45 * 1000 + punctuation_pause))


def infer_emotion(text: str) -> str:
    if "!" in text:
        return "intense"
    if "?" in text:
        return "questioning"
    if "…" in text or "..." in text:
        return "hesitant"
    return "neutral"


def build_speech_plans(utterances: list[tuple[srt.Subtitle, str | None]], video_duration_ms: int) -> list[SpeechPlan]:
    plans: list[SpeechPlan] = []
    for index, (cue, _) in enumerate(utterances):
        start_ms = round(cue.start.total_seconds() * 1000)
        next_start = round(utterances[index + 1][0].start.total_seconds() * 1000) if index + 1 < len(utterances) else video_duration_ms
        target_ms = max(250, round((cue.end - cue.start).total_seconds() * 1000))
        hard_ms = max(target_ms, next_start - start_ms)
        spoken = normalize_spoken_text(cue.content)
        predicted = predict_duration_ms(spoken)
        plans.append(SpeechPlan(
            id=cue.index, subtitle_text=cue.content, spoken_text=spoken, start_ms=start_ms,
            target_duration_ms=target_ms, hard_deadline_ms=hard_ms, predicted_duration_ms=predicted,
            emotion=infer_emotion(cue.content), pace=min(1.18, max(0.92, predicted / max(1, target_ms))),
        ))
    return plans
