import json
import os
import re
import time
from collections.abc import Callable

import requests
import srt

from .config import get_settings
from .subtitle import parse_translation_json

SYSTEM_PROMPT = """You are a senior Vietnamese subtitle localizer for Chinese short dramas.
Translate dialogue into natural Vietnamese.
Rules:
- Return valid JSON only.
- Preserve every id exactly.
- Preserve the exact number of items.
- Translate only the text.
- Do not include timestamps.
- Do not add explanations.
- Keep names, brands, numbers, and repeated words.
- Use natural spoken Vietnamese, not word-for-word translation.
- Localize meaning and intent; freely restructure Chinese phrasing so it sounds like dialogue written by a Vietnamese screenwriter.
- Input comes from ASR and may contain corrupted or implausible fragments. Use neighboring items and the scene context to repair obvious recognition errors instead of translating nonsense literally.
- When repairing ASR noise, preserve the most likely intended meaning and do not invent new plot facts.
- Omit isolated ASR garbage that cannot fit the surrounding context, especially noise before a clear introduction; never reproduce nonsensical fragments in Vietnamese.
- Match the requested max_words for each item. Prefer a shorter natural equivalent over a complete literal rendering.
- The Vietnamese line must be comfortably speakable within target_duration_seconds without rushing.
- Punctuate Vietnamese by meaning so TTS pauses naturally. Use commas only for short internal pauses and periods/questions/exclamations for complete thoughts.
- Do not add ellipses merely because a sentence crosses cue boundaries; make the full sequence read continuously and naturally.
- Keep emotional tension and relationship nuance.
- Use pronouns naturally: anh/em, tôi/cô, mẹ/con, sếp/tôi depending on context.
- Keep subtitles concise and easy to read.
- Avoid overly literary, stiff, or machine-like Vietnamese.
- If a sentence is already Vietnamese, lightly polish only if needed."""
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
SHORTEN_SYSTEM_PROMPT = """You are a Vietnamese dialogue editor.
Rewrite each Vietnamese line into natural, conversational short-drama dialogue.
Keep the same intent, emotion, names and important facts, but remove redundant wording.
Never exceed max_words. Return valid JSON only with exactly the same ids and text fields."""


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _user_prompt(items: list[dict], source: str, target: str, glossary: str | None) -> str:
    return (
        f"Source language: {source}\nTarget language: {target}\n"
        "Style: Vietnamese short-drama subtitles.\n"
        f"Glossary:\n{glossary or '(none)'}\n\n"
        "Translate this JSON array and return the same ids:\n"
        + json.dumps(items, ensure_ascii=False)
    )


def _openai_compatible(provider: str, system: str, user: str) -> str:
    prefix = "OPENAI" if provider == "openai" else "DEEPSEEK"
    defaults = {
        "OPENAI_MODEL": "gpt-4o-mini", "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "DEEPSEEK_MODEL": "deepseek-chat", "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
    }
    response = requests.post(
        f"{os.getenv(prefix + '_BASE_URL', defaults[prefix + '_BASE_URL']).rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {_required(prefix + '_API_KEY')}", "Content-Type": "application/json"},
        json={
            "model": os.getenv(prefix + "_MODEL", defaults[prefix + "_MODEL"]),
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _gemini(system: str, user: str) -> str:
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": _required("GEMINI_API_KEY")},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


def translate_subtitles(
    subtitles: list[srt.Subtitle], provider: str, source_language: str, target_language: str,
    glossary: str | None = None, progress: Callable[[str], None] | None = None,
    speakers: dict[int, str] | None = None,
) -> list[srt.Subtitle]:
    settings = get_settings()
    translated: list[srt.Subtitle] = []
    for offset in range(0, len(subtitles), settings.translation_batch_size):
        batch = subtitles[offset : offset + settings.translation_batch_size]
        items = [
            {
                "id": cue.index,
                "text": cue.content,
                "target_duration_seconds": round((cue.end - cue.start).total_seconds(), 2),
                # Edge Vietnamese at +10% averages roughly 3.5 spoken words/s;
                # the timing stage can safely correct the remaining small drift.
                "max_words": max(2, min(28, round((cue.end - cue.start).total_seconds() * 3.5))),
                **({"speaker": speakers[cue.index]} if speakers and cue.index in speakers else {}),
            }
            for cue in batch
        ]
        expected_ids = [cue.index for cue in batch]
        prompt = _user_prompt(items, source_language, target_language, glossary)
        last_error: Exception | None = None
        for attempt in range(1, settings.translation_retries + 1):
            try:
                raw = _gemini(SYSTEM_PROMPT, prompt) if provider == "gemini" else _openai_compatible(provider, SYSTEM_PROMPT, prompt)
                mapping = parse_translation_json(raw, expected_ids)
                untranslated = [item_id for item_id, text in mapping.items() if len(_CJK_RE.findall(text)) > max(1, len(text) * 0.10)]
                if target_language.lower().startswith("vietnam") and untranslated:
                    raise ValueError(f"Provider left Chinese text untranslated for ids: {untranslated}")
                limits = {item["id"]: item["max_words"] for item in items}
                overlong = [item_id for item_id, text in mapping.items() if len(text.split()) > limits[item_id] + 2]
                if target_language.lower().startswith("vietnam") and overlong:
                    shorten_items = [
                        {"id": item_id, "text": mapping[item_id], "max_words": limits[item_id]}
                        for item_id in overlong
                    ]
                    shorten_prompt = (
                        "Shorten these lines. Natural Vietnamese is more important than literal wording. "
                        "Return the same ids:\n" + json.dumps(shorten_items, ensure_ascii=False)
                    )
                    shortened_raw = (
                        _gemini(SHORTEN_SYSTEM_PROMPT, shorten_prompt)
                        if provider == "gemini"
                        else _openai_compatible(provider, SHORTEN_SYSTEM_PROMPT, shorten_prompt)
                    )
                    mapping.update(parse_translation_json(shortened_raw, overlong))
                    overlong = [item_id for item_id in overlong if len(mapping[item_id].split()) > limits[item_id] + 2]
                if target_language.lower().startswith("vietnam") and overlong:
                    raise ValueError(f"Vietnamese dialogue exceeds speaking-time budget for ids: {overlong}")
                break
            except Exception as exc:
                last_error = exc
                if attempt == settings.translation_retries:
                    raise RuntimeError(f"Translation batch failed after {attempt} attempts: {exc}") from exc
                if "speaking-time budget" in str(exc):
                    prompt += (
                        "\n\nRETRY REQUIRED: Your previous answer was too long for the ids listed in this error: "
                        f"{exc}. Rewrite those lines much more concisely. Every line must stay within its max_words; "
                        "use an idiomatic Vietnamese equivalent and omit redundant wording. Return the complete JSON batch again."
                    )
                time.sleep(attempt * 2)
        else:
            raise RuntimeError(str(last_error))
        translated.extend(srt.Subtitle(index=cue.index, start=cue.start, end=cue.end, content=mapping[cue.index]) for cue in batch)
        if progress:
            progress(f"Translated {min(offset + len(batch), len(subtitles))}/{len(subtitles)} subtitle cues")
    return translated
