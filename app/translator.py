import json
import os
import re
import time
from pathlib import Path
from collections.abc import Callable

import requests
import srt

from .config import get_settings
from .subtitle import parse_translation_json
from .artifacts import atomic_write_json

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
- ASR may split one phrase across adjacent ids. Distribute the Vietnamese phrase across those ids as natural continuations; never repeat the final word in a short fragment id.
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
SCENE_EDITOR_PROMPT = """You are the final dialogue editor for a Vietnamese short drama.
Edit the supplied Vietnamese scene as one coherent conversation.
Rules:
- Return valid JSON only, with exactly the same ids and text fields.
- Preserve plot facts, names, numbers, intent and emotional progression.
- Make adjacent lines flow naturally; remove accidental repetition caused by ASR.
- Keep character address and pronouns consistent with speaker and context.
- Prefer idiomatic spoken Vietnamese over literal Chinese syntax.
- When one source sentence crosses adjacent ids, distribute it as a continuous Vietnamese sentence. Never turn a one-character ASR tail into a repeated standalone word.
- Do not exceed each item's max_words. Do not add explanations."""


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
    speakers: dict[int, str] | None = None, job_dir: Path | None = None,
) -> list[srt.Subtitle]:
    settings = get_settings()
    translated: list[srt.Subtitle] = []
    draft_path = job_dir / "vi-draft.json" if job_dir else None
    draft_mapping: dict[int, str] = {}
    if draft_path and draft_path.is_file():
        try:
            draft_mapping = {int(item["id"]): str(item["text"]) for item in json.loads(draft_path.read_text(encoding="utf-8"))}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            draft_mapping = {}
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
        if all(cue.index in draft_mapping for cue in batch):
            mapping = {cue.index: draft_mapping[cue.index] for cue in batch}
            translated.extend(srt.Subtitle(index=cue.index, start=cue.start, end=cue.end, content=mapping[cue.index]) for cue in batch)
            if progress:
                progress(f"Resumed translation {min(offset + len(batch), len(subtitles))}/{len(subtitles)} cues")
            continue
        context_start = max(0, offset - settings.translation_context_before)
        context_end = min(len(subtitles), offset + len(batch) + settings.translation_context_after)
        context = [
            {"id": cue.index, "speaker": (speakers or {}).get(cue.index), "text": cue.content,
             "translate": cue in batch}
            for cue in subtitles[context_start:context_end]
        ]
        prompt = _user_prompt(items, source_language, target_language, glossary)
        prompt += "\n\nContext before/after (for understanding only; translate=false ids must not be returned):\n"
        prompt += json.dumps(context, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(1, settings.translation_retries + 1):
            try:
                raw = _gemini(SYSTEM_PROMPT, prompt) if provider == "gemini" else _openai_compatible(provider, SYSTEM_PROMPT, prompt)
                mapping = parse_translation_json(raw, expected_ids, allow_missing=True)
                missing_ids = [item_id for item_id in expected_ids if item_id not in mapping]
                if missing_ids:
                    missing_items = [item for item in items if item["id"] in missing_ids]
                    repair_prompt = (
                        "The previous response omitted these non-empty dialogue items. Translate only this JSON array. "
                        "Even if ASR seems imperfect, provide the best contextual Vietnamese interpretation; NEVER omit "
                        "an item and NEVER return an empty array. Return every listed id exactly once in this exact shape: "
                        '{"translations":[{"id":25,"text":"Vietnamese dialogue"}]}. '
                        f"The required ids are {missing_ids}:\n"
                        + json.dumps(missing_items, ensure_ascii=False)
                    )
                    repair_raw = (
                        _gemini(SYSTEM_PROMPT, repair_prompt)
                        if provider == "gemini"
                        else _openai_compatible(provider, SYSTEM_PROMPT, repair_prompt)
                    )
                    mapping.update(parse_translation_json(repair_raw, missing_ids))
                    if progress:
                        progress(f"Recovered omitted translation ids: {missing_ids}")
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
                    # A short ASR cue is often only the tail of a sentence. Failing the
                    # whole job here prevents the dialogue-master stage from joining it
                    # with adjacent cues and using their combined speaking window.
                    # Keep the best shortened text and let final timing QA decide whether
                    # the grouped TTS audio is genuinely too long.
                    if progress:
                        progress(
                            "Translation retained for dialogue reflow despite a tight "
                            f"speaking window in cues: {overlong}"
                        )
                break
            except Exception as exc:
                last_error = exc
                if attempt == settings.translation_retries:
                    raise RuntimeError(f"Translation batch failed after {attempt} attempts: {exc}") from exc
                if isinstance(exc, ValueError):
                    prompt += (
                        "\n\nRETRY REQUIRED: The previous response failed validation: "
                        f"{exc}. Return JSON only in exactly this shape: "
                        '{"translations":[{"id":1,"text":"Vietnamese text"}]}. '
                        f"Return every expected id exactly once: {expected_ids}. "
                        "Do not use alternate field names, null values, markdown, or explanations."
                    )
                time.sleep(attempt * 2)
        else:
            raise RuntimeError(str(last_error))
        draft_mapping.update(mapping)
        if draft_path:
            atomic_write_json(draft_path, [{"id": cue.index, "text": draft_mapping[cue.index]} for cue in subtitles if cue.index in draft_mapping])
        translated.extend(srt.Subtitle(index=cue.index, start=cue.start, end=cue.end, content=mapping[cue.index]) for cue in batch)
        if progress:
            progress(f"Translated {min(offset + len(batch), len(subtitles))}/{len(subtitles)} subtitle cues")
    if not settings.translation_scene_review or len(translated) < 2:
        if job_dir:
            atomic_write_json(job_dir / "vi-final.json", [{"id": cue.index, "text": cue.content} for cue in translated])
        return translated

    # Review coherent chunks with overlap-free boundaries. This second pass fixes
    # pronouns and continuity after all draft lines are available.
    final_mapping = {cue.index: cue.content for cue in translated}
    speaker_memory: dict[str, list[str]] = {}
    review_size = 30
    for offset in range(0, len(translated), review_size):
        scene = translated[offset : offset + review_size]
        review_items = [
            {
                "id": cue.index, "speaker": (speakers or {}).get(cue.index), "text": cue.content,
                "max_words": max(2, min(28, round((cue.end - cue.start).total_seconds() * 3.5))),
            }
            for cue in scene
        ]
        previous = translated[max(0, offset - 5):offset]
        user = "Edit this scene:\n" + json.dumps(review_items, ensure_ascii=False)
        if previous:
            user += "\nPrevious dialogue for context only:\n" + json.dumps(
                [{"id": cue.index, "text": final_mapping[cue.index]} for cue in previous], ensure_ascii=False
            )
        if speaker_memory:
            user += "\nEstablished character voice/pronoun examples (context only):\n" + json.dumps(speaker_memory, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(1, settings.translation_retries + 1):
            try:
                raw = _gemini(SCENE_EDITOR_PROMPT, user) if provider == "gemini" else _openai_compatible(provider, SCENE_EDITOR_PROMPT, user)
                reviewed = parse_translation_json(raw, [cue.index for cue in scene])
                limits = {item["id"]: item["max_words"] for item in review_items}
                too_long = [item_id for item_id, text in reviewed.items() if len(text.split()) > limits[item_id] + 2]
                if too_long:
                    raise ValueError(f"Scene editor exceeded word budget for ids: {too_long}")
                final_mapping.update(reviewed)
                for cue in scene:
                    speaker = (speakers or {}).get(cue.index)
                    if speaker:
                        examples = speaker_memory.setdefault(speaker, [])
                        examples.append(reviewed[cue.index])
                        del examples[:-6]
                break
            except Exception as exc:
                last_error = exc
                if attempt < settings.translation_retries:
                    time.sleep(attempt * 2)
        if last_error and any(cue.index not in final_mapping for cue in scene):
            raise RuntimeError(f"Scene editing failed: {last_error}")
        if progress:
            progress(f"Context-edited {min(offset + len(scene), len(translated))}/{len(translated)} cues")
    final = [srt.Subtitle(index=cue.index, start=cue.start, end=cue.end, content=final_mapping[cue.index]) for cue in translated]
    if job_dir:
        atomic_write_json(job_dir / "vi-final.json", [{"id": cue.index, "text": cue.content} for cue in final])
    return final
