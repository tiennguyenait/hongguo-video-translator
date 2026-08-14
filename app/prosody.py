"""Conservative AI-assisted delivery planning for Vietnamese narration."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable

from .speech_plan import SpeechPlan

PROSODY_VERSION = "conservative-v1"
ALLOWED_EMOTIONS = {"neutral", "warm", "sad", "tense", "angry", "questioning"}
ALLOWED_STYLES = {"doc_truyen", "tu_nhien"}

SYSTEM_PROMPT = """You direct a Vietnamese short-drama narrator.
Return valid JSON only. For every input id return exactly: id, spoken_text, emotion,
intensity, style, pause_before_ms, pause_after_ms.
Rules:
- Preserve every word, number, and name. You may ONLY change punctuation and letter case.
- Do not rewrite, add, remove, or reorder words.
- Use restrained acting, never theatrical or sing-song delivery.
- emotion is one of neutral, warm, sad, tense, angry, questioning.
- intensity is 0.30 to 0.75.
- style is doc_truyen normally; use tu_nhien only for direct, urgent, angry, or questioning speech.
- pause_before_ms is 0 to 120; pause_after_ms is 60 to 280.
- Avoid repeated ellipses and do not fragment short sentences.
- Preserve every id exactly and return the exact number of items."""


def _lexical_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(re.findall(r"[\wÀ-ỹ]+", normalized, re.UNICODE))


def _json_items(raw: str) -> list[dict]:
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    payload = json.loads(clean)
    if isinstance(payload, dict):
        for key in ("prosody", "items", "translations"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Prosody response must be a JSON array")
    return payload


def apply_ai_prosody(plans: list[SpeechPlan], raw: str) -> list[SpeechPlan]:
    """Validate an AI plan strictly; invalid individual items keep their safe defaults."""
    by_id = {int(item["id"]): item for item in _json_items(raw) if "id" in item}
    if set(by_id) != {plan.id for plan in plans}:
        raise ValueError("Prosody response does not preserve every id")
    for plan in plans:
        item = by_id[plan.id]
        spoken = str(item.get("spoken_text", "")).strip()
        if not spoken or _lexical_key(spoken) != _lexical_key(plan.spoken_text):
            continue
        emotion = str(item.get("emotion", "neutral")).lower()
        style = str(item.get("style", "doc_truyen")).lower()
        if emotion not in ALLOWED_EMOTIONS or style not in ALLOWED_STYLES:
            continue
        try:
            intensity = min(0.75, max(0.30, float(item.get("intensity", 0.45))))
            before = min(120, max(0, int(item.get("pause_before_ms", 0))))
            after = min(280, max(60, int(item.get("pause_after_ms", 100))))
        except (TypeError, ValueError):
            continue
        plan.spoken_text = spoken
        plan.emotion = emotion
        plan.intensity = round(intensity, 2)
        plan.style = style
        plan.pause_before_ms = before
        plan.pause_after_ms = after
        plan.prosody_source = "ai"
    return plans


def plan_prosody(
    plans: list[SpeechPlan], provider: str,
    request: Callable[[str, str, str], str] | None = None,
) -> tuple[list[SpeechPlan], str | None]:
    """Plan delivery in scene-sized batches; provider errors never fail the dub."""
    if not plans:
        return plans, None
    if request is None:
        from .translator import _gemini, _openai_compatible
        request = lambda selected, system, user: (
            _gemini(system, user) if selected == "gemini" else _openai_compatible(selected, system, user)
        )
    warning = None
    for offset in range(0, len(plans), 30):
        batch = plans[offset:offset + 30]
        items = [{
            "id": plan.id, "text": plan.spoken_text, "current_emotion": plan.emotion,
            "duration_seconds": round(plan.target_duration_ms / 1000, 2),
        } for plan in batch]
        try:
            raw = request(provider, SYSTEM_PROMPT, "Direct this scene:\n" + json.dumps(items, ensure_ascii=False))
            apply_ai_prosody(batch, raw)
        except Exception as exc:  # Safe deterministic delivery is preferable to a failed job.
            warning = f"AI prosody unavailable; used safe fallback: {exc}"
    return plans, warning
