import json
import os
import re
import time
from contextvars import ContextVar
from pathlib import Path
from collections.abc import Callable

import requests
import srt

from .config import get_settings
from .subtitle import is_ignorable_asr_fragment, parse_translation_json
from .artifacts import atomic_write_json

_usage_path: ContextVar[Path | None] = ContextVar("ai_usage_path", default=None)
_usage_report: ContextVar[dict | None] = ContextVar("ai_usage_report", default=None)


def start_ai_usage(job_dir: Path, provider: str) -> None:
    """Start an atomic per-job usage ledger for the current worker context."""
    path = job_dir / "ai-usage.json"
    report = {
        "provider": provider, "requests": 0, "prompt_tokens": 0,
        "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0, "total_tokens": 0,
        "estimated_cost_usd": 0.0, "models": {},
    }
    _usage_path.set(path)
    _usage_report.set(report)
    atomic_write_json(path, report)


def _record_usage(provider: str, model: str, usage: dict | None) -> None:
    report, path = _usage_report.get(), _usage_path.get()
    if report is None or path is None:
        return
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    miss = int(usage.get("prompt_cache_miss_tokens", max(0, prompt - hit)) or 0)
    report["requests"] += 1
    report["prompt_tokens"] += prompt
    report["prompt_cache_hit_tokens"] += hit
    report["prompt_cache_miss_tokens"] += miss
    report["completion_tokens"] += completion
    report["total_tokens"] += int(usage.get("total_tokens", prompt + completion) or 0)
    report["models"][model] = report["models"].get(model, 0) + 1
    if provider == "deepseek":
        # Current official USD prices per million tokens. Unknown/legacy chat
        # aliases use the cheaper Flash estimate and are clearly marked as an
        # estimate rather than billing truth.
        if "pro" in model.lower():
            hit_rate, miss_rate, output_rate = 0.003625, 0.435, 0.87
        else:
            hit_rate, miss_rate, output_rate = 0.0028, 0.14, 0.28
        report["estimated_cost_usd"] = round(
            report["estimated_cost_usd"]
            + (hit * hit_rate + miss * miss_rate + completion * output_rate) / 1_000_000,
            6,
        )
    atomic_write_json(path, report)

SYSTEM_PROMPT = """You are a senior Vietnamese subtitle and dubbing localizer specializing in Chinese romance short dramas.

Your task is to convert Chinese dialogue into natural, emotionally accurate Vietnamese for Vietnamese viewers.

CORE PRINCIPLES
- Localize meaning and character intent, not Chinese wording.
- Use natural spoken Vietnamese that sounds like dialogue written by a Vietnamese screenwriter.
- Preserve the original plot, facts, relationships, emotions, and implied meaning.
- Do not add new facts, motivations, jokes, exposition, or explanations.
- Preserve emotional nuances such as longing, jealousy, anger, humiliation, betrayal, reconciliation, family pressure, class/status tension, and romantic tension.
- Avoid literal Chinese syntax, stiff wording, literary language, and machine-translation style.

ADDRESS & RELATIONSHIPS
- Use the provided speaker identity for each cue to keep pronoun consistency.
- Do not assume one speaker dominates a whole batch.
- Never infer identity, gender, social role, age, or intimacy from technical labels alone.
- Use scene context and speaker relationship to choose natural Vietnamese address terms.
- If speaker label is missing/empty, do not force gendered pronouns from 我/你.
- For ambiguous cases with no reliable speaker context, prefer neutral forms (ví dụ: em, tôi, người ấy, bạn) rather than guessing.

SPEAKER / PRONOUN RULES
- Speaker labels identify who is speaking.
- Do not assume the same speaker continues across adjacent subtitle IDs.
- Determine the addressee from dialogue context.
- Determine Vietnamese pronouns from BOTH speaker and addressee context.
- Never infer pronouns from Chinese 我/你 alone.
- Maintain consistent pronouns between the same two interacting speakers.

ASR REPAIR
- Input may come from ASR and may contain recognition errors, duplicated words, missing words, or meaningless fragments.
- Use neighboring dialogue and scene context to repair obvious ASR errors.
- Preserve the most likely intended meaning.
- Never invent new plot information.
- Remove isolated ASR garbage when it clearly cannot belong to the dialogue.
- Do not reproduce accidental ASR duplication.
- Preserve meaningful repetition when it expresses hesitation, panic, anger, emphasis, or emotion.

SUBTITLE & DUBBING
- Keep each line concise, readable, and natural.
- Prefer shorter natural wording when the meaning remains complete.
- Respect max_words when provided.
- Ensure the line is comfortably speakable within target_duration_seconds when provided.
- For dubbing, prioritize natural speaking rhythm and duration over literal completeness.
- Punctuate by meaning so TTS produces natural pauses.
- Every line should keep natural Vietnamese sentence-level punctuation at the end
  (., ?, !, ...). Match question and exclamation tone from source cues where
  possible (for example, source ends with ?/？ should map to ?, !/！ to !).
- Use commas only when the cue is clearly a continuation of the same sentence;
  otherwise prefer . or ?/! for final cadence.
- Do not add ellipses merely because a sentence is split across subtitle cues.
- When one Chinese phrase spans multiple IDs, distribute the Vietnamese naturally across those IDs without repeating words.

OUTPUT
- Return valid JSON only.
- Preserve every id exactly.
- Preserve the exact number of items.
- Translate only the text field.
- Do not include timestamps.
- Do not include explanations or comments.
- Keep names, brands, and numbers unless localization explicitly requires otherwise.
- If the source is already Vietnamese, lightly polish it only when necessary."""

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
SHORTEN_SYSTEM_PROMPT = """You are a Vietnamese dialogue editor.
Rewrite each Vietnamese line into natural, conversational short-drama dialogue.
Keep the same intent, emotion, names and important facts, but remove redundant wording.
Never exceed max_words.

OUTPUT CONTRACT (mandatory):
- Return exactly one JSON object in this shape: {"translations":[{"id":1,"text":"Vietnamese dialogue"}]}.
- Each input item must produce exactly one item in translations.
- Copy every input id exactly; do not omit, add, renumber, or stringify ids.
- The only allowed item fields are id and text. Never rename them to translation,
  shortened_text, content, output, cue_id, index, or any other name.
- translations must always be a JSON array, even when there is only one item.
- text must be a non-empty JSON string.
- Do not return Markdown fences, comments, explanations, or any text outside the JSON object.
Before responding, silently verify that the returned ids exactly match the input ids."""
SCENE_EDITOR_PROMPT = """You are the final dialogue editor for a Vietnamese Chinese-romance short drama.
Edit the supplied Vietnamese scene as one coherent conversation.
Rules:
- Return valid JSON only, with exactly the same ids and text fields.
- Preserve plot facts, names, numbers, intent and emotional progression.
- Make adjacent lines flow naturally; remove accidental repetition caused by ASR.
- Keep character address and pronouns consistent with speaker and context.
- Speaker labels identify who is speaking in each cue.
- Do not assume the same speaker continues across adjacent ids.
- If a cue has no speaker label, do not infer gender or social role from 我/你.
- Avoid introducing gendered pronouns unless supported by clear speaker context.
- Prefer idiomatic spoken Vietnamese over literal Chinese syntax, especially for romance-drama conflict, status, jealousy, longing, and family-pressure scenes.
- When one source sentence crosses adjacent ids, distribute it as a continuous Vietnamese sentence. Never turn a one-character ASR tail into a repeated standalone word.
- Do not exceed each item's max_words. Do not add explanations.
- Ensure each edited line has natural sentence-ending punctuation for TTS pauses
  (., ?, !, ...). If the cue is rhetorical or clearly interrogative, use ?.
"""

PUNCTUATION_SYSTEM_PROMPT = """You are a Vietnamese subtitle punctuation corrector for dubbing.
Return valid JSON only.

Task: add or adjust only final sentence punctuation for each cue while preserving
all original words as much as possible.

Rules:
- Keep exactly the same ids and item count.
- Keep the Vietnamese wording unchanged except where punctuation is needed
  for natural cadence.
- Prefer natural end punctuation from source tone:
  source ?/？ -> ?, source !/！ -> !, source .../…… -> ....
- Use a period for neutral declaratives when no explicit strong clue exists.
- Return JSON in this exact shape: {"translations":[{"id":1,"text":"Vietnamese dialogue"}]}.
"""

_TRAILING_PUNCT_RE = re.compile(r"[.!?…,:;]\s*[\"”'')\]\}]*$")
_CJK_QUESTION_PARTICLES = ("吗", "吗?", "吗？", "呢", "吧", "嘛", "么")
_SOURCE_PUNCT_MAP = {
    "?": "?",
    "？": "?",
    "!": "!",
    "！": "!",
    "。": ".",
    ".": ".",
    "，": ",",
    "、": ",",
    ",": ",",
    "…": "...",
    "……": "...",
}


def _has_terminal_punctuation(text: str) -> bool:
    return bool(_TRAILING_PUNCT_RE.search((text or "").strip()))


def _source_tail_punctuation(source_text: str) -> str:
    cleaned = (source_text or "").strip()
    if not cleaned:
        return ""
    tail = cleaned[-1]
    if tail in _SOURCE_PUNCT_MAP:
        return _SOURCE_PUNCT_MAP[tail]
    if cleaned.endswith("..."):
        return "..."
    if cleaned.endswith("……"):
        return "..."
    for particle in _CJK_QUESTION_PARTICLES:
        if cleaned.endswith(particle):
            return "?"
    return ""


def _append_ending_punctuation(text: str, source: str, fallback: str = ".") -> str:
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    if _has_terminal_punctuation(stripped):
        return stripped
    source_punct = _source_tail_punctuation(source)
    if source_punct == "," and len(stripped.split()) <= 3:
        return stripped + ","
    if source_punct == "," and len(stripped) <= 6 and not stripped.endswith(("à", "ơi", "nha", "nhé", "à.")):
        return stripped + ","
    if source_punct:
        return stripped + source_punct
    return stripped + fallback


def _enforce_punctuation(mapping: dict[int, str], batch: list[dict], provider: str) -> dict[int, str]:
    source_by_id = {item["id"]: item.get("text", "") for item in batch}
    items = [
        {"id": item["id"], "source": item["text"], "draft": mapping[item["id"]], "speaker": item.get("speaker"), "target_duration_seconds": item.get("target_duration_seconds", 0)}
        for item in batch
        if mapping.get(item["id"], "").strip()
    ]
    needed = [item["id"] for item in items if not _has_terminal_punctuation(mapping[item["id"]])]
    if not needed:
        return mapping
    try:
        payload = _call_json_provider(
            provider,
            PUNCTUATION_SYSTEM_PROMPT,
            "Add punctuation to these Vietnamese lines while preserving text. Input JSON:\n" + json.dumps(items, ensure_ascii=False),
            needed,
        )
        for item_id, text in payload.items():
            if item_id in needed and text.strip():
                mapping[item_id] = text.strip()
    except Exception:
        pass
    for item_id in needed:
        text = mapping.get(item_id, "")
        if not _has_terminal_punctuation(text):
            mapping[item_id] = _append_ending_punctuation(text, source_by_id.get(item_id, ""), ".")
    return mapping


def _polish_prompt(
    items: list[dict], source: str, target: str, glossary: str | None, context_note: str = "",
) -> str:
    return (
        f"Source language: {source}\nTarget language: {target}\n"
        "Task: polish rough Vietnamese draft subtitles for emotional accuracy and natural spoken tone.\n"
        "Rules:\n"
        "- Keep exactly the same ids and count.\n"
        "- Output only Vietnamese dialogue text in the `text` field.\n"
        "- Do not change meaning, facts, plot events, names, or character relationships.\n"
        "- Improve pronoun/address choices using speaker context and scene emotion.\n"
        "- Keep lines concise and fit cue duration.\n"
        "Input format per item:\n"
        "- id: numeric subtitle id\n"
        "- source: original Chinese line\n"
        "- draft: rough Vietnamese translation from the first pass\n"
        f"Glossary:\n{glossary or '(none)'}\n\n"
        + json.dumps(items, ensure_ascii=False)
        + ("\n\nContext note:\n" + context_note if context_note else "")
    )


def _call_json_provider(provider: str, system_prompt: str, user_prompt: str, expected_ids: list[int]) -> dict[int, str]:
    raw = _gemini(system_prompt, user_prompt) if provider == "gemini" else _openai_compatible(provider, system_prompt, user_prompt)
    return parse_translation_json(raw, expected_ids, allow_missing=True)


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _user_prompt(items: list[dict], source: str, target: str, glossary: str | None, context_note: str = "") -> str:
    return (
        f"Source language: {source}\nTarget language: {target}\n"
        "Style: Vietnamese Chinese-romance short-drama subtitles, faithful to original timing and plot.\n"
        f"Glossary:\n{glossary or '(none)'}\n\n"
        "Translate this JSON array and return the same ids, preserving natural Vietnamese punctuation for each line.\n"
        + json.dumps(items, ensure_ascii=False)
        + ("\n\nContext note:\n" + context_note if context_note else "")
    )


def _openai_compatible(provider: str, system: str, user: str) -> str:
    prefix = "OPENAI" if provider == "openai" else "DEEPSEEK"
    defaults = {
        "OPENAI_MODEL": "gpt-4o-mini", "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "DEEPSEEK_MODEL": "deepseek-chat", "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
    }
    model = os.getenv(prefix + "_MODEL", defaults[prefix + "_MODEL"])
    response = requests.post(
        f"{os.getenv(prefix + '_BASE_URL', defaults[prefix + '_BASE_URL']).rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {_required(prefix + '_API_KEY')}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    _record_usage(provider, model, payload.get("usage"))
    return payload["choices"][0]["message"]["content"]


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
    payload = response.json()
    metadata = payload.get("usageMetadata", {})
    _record_usage("gemini", model, {
        "prompt_tokens": metadata.get("promptTokenCount", 0),
        "completion_tokens": metadata.get("candidatesTokenCount", 0),
        "total_tokens": metadata.get("totalTokenCount", 0),
    })
    return payload["candidates"][0]["content"]["parts"][0]["text"]


def translate_subtitles(
    subtitles: list[srt.Subtitle], provider: str, source_language: str, target_language: str,
    glossary: str | None = None, progress: Callable[[str], None] | None = None,
    speakers: dict[int, str] | None = None, job_dir: Path | None = None,
    speaker_gender_profile: str | None = None, draft_provider: str | None = None,
    refine_provider: str | None = None,
) -> list[srt.Subtitle]:
    settings = get_settings()
    refine_provider = (refine_provider or provider).lower().strip()
    draft_provider = (draft_provider or provider).lower().strip()
    if draft_provider == "google":
        draft_provider = refine_provider
    if refine_provider not in {"openai", "gemini", "deepseek"}:
        raise ValueError(f"Unsupported refine provider: {refine_provider}")
    if draft_provider not in {"openai", "gemini", "deepseek"}:
        raise ValueError(f"Unsupported draft provider: {draft_provider}")
    primary_provider = refine_provider
    # Preserve speaker consistency without exposing diarizer labels such as
    # SPEAKER_07 to the LLM (those labels carry no identity or gender).
    speaker_aliases = {
        label: f"character_{position}"
        for position, label in enumerate(dict.fromkeys((speakers or {}).values()), 1)
    }
    translated: list[srt.Subtitle] = []
    draft_path = job_dir / "vi-draft.json" if job_dir else None
    draft_mapping: dict[int, str] = {}
    speaker_labels = [label for label in dict.fromkeys((speakers or {}).values()) if label]
    reliable_speaker_count = len(set(speaker_labels))
    has_any_speaker = reliable_speaker_count >= 2

def _speaker_note() -> str:
    profile = (speaker_gender_profile or "auto").lower()
    if has_any_speaker:
        note = (
            "Speaker labels are anonymous aliases only (character_1 / character_2). "
            "Do not infer gender, social status, or relationship from the label alone. "
            "If the cue does not explicitly indicate a clear honorific relationship, avoid Anh/Em and prefer neutral forms "
            "(bạn, mình, người ấy) to reduce wrong gendered address."
        )
    else:
        note = (
            "No reliable speaker IDs were available for this batch. "
            "Do not infer gender or social role from 我/你 alone. "
            "Prefer neutral Vietnamese address forms (bạn, mình, em, tôi, người ấy) and keep consistency where possible."
        )
    if profile == "female":
        note += (
            " The request was explicitly marked as female-narrator style; if a direct 2nd-person line must exist, "
            "use polite female-first forms like em."
        )
    elif profile == "male":
        note += (
            " The request was explicitly marked as male-narrator style; if a direct 2nd-person line must exist, "
            "use polite male-first forms like anh."
        )
    elif profile == "neutral":
        note += (
            " Use neutral address forms by default; avoid shifting to Anh/Em unless explicitly anchored by context."
        )
    return note
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
                # Speaker labels are contextual metadata, never translation
                # text.  Including them fixes pronoun/address drift while
                # keeping the strict JSON output contract unchanged.
                "speaker": speaker_aliases.get((speakers or {}).get(cue.index)) if has_any_speaker else None,
                "target_duration_seconds": round((cue.end - cue.start).total_seconds(), 2),
                # Keep the budget conservative so dubbing stays close to the
                # original cue instead of relying on aggressive time-stretch.
                "max_words": max(2, min(24, round((cue.end - cue.start).total_seconds() * 3.2))),
            }
            for cue in batch
        ]
        expected_ids = [cue.index for cue in batch]
        required_ids = [cue.index for cue in batch if not is_ignorable_asr_fragment(cue)]
        if (
            all(cue.index in draft_mapping for cue in batch)
            and all(draft_mapping[item_id].strip() for item_id in required_ids)
        ):
            mapping = {cue.index: draft_mapping[cue.index] for cue in batch}
            mapping = _enforce_punctuation(mapping, items, primary_provider)
            translated.extend(srt.Subtitle(index=cue.index, start=cue.start, end=cue.end, content=mapping[cue.index]) for cue in batch)
            if progress:
                progress(f"Resumed translation {min(offset + len(batch), len(subtitles))}/{len(subtitles)} cues")
            continue
        context_start = max(0, offset - settings.translation_context_before)
        context_end = min(len(subtitles), offset + len(batch) + settings.translation_context_after)
        context = [
            {"id": cue.index, "text": cue.content,
             "speaker": speaker_aliases.get((speakers or {}).get(cue.index)) if has_any_speaker else None,
             "translate": cue in batch}
            for cue in subtitles[context_start:context_end]
        ]
        speaker_note = _speaker_note()
        prompt = _user_prompt(items, source_language, target_language, glossary, speaker_note)
        prompt += "\n\nContext before/after (for understanding only; translate=false ids must not be returned):\n"
        prompt += json.dumps(context, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(1, settings.translation_retries + 1):
            try:
                mapping = _call_json_provider(primary_provider, SYSTEM_PROMPT, prompt, expected_ids)
                # A present-but-empty translation is equivalent to an omitted
                # item for real dialogue. Only tightly bounded ASR noise may be
                # intentionally silent.
                missing_ids = [
                    item_id for item_id in required_ids
                    if item_id not in mapping or not mapping[item_id].strip()
                ]
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
                    repair_raw = _call_json_provider(primary_provider, SYSTEM_PROMPT, repair_prompt, missing_ids)
                    mapping.update(repair_raw)
                    still_empty = [item_id for item_id in missing_ids if not mapping[item_id].strip()]
                    if still_empty:
                        raise ValueError(f"Provider returned empty dialogue for ids: {still_empty}")
                    if progress:
                        progress(f"Recovered omitted translation ids: {missing_ids}")
                # Vietnamese deliverables must not contain even a short Chinese
                # fragment. The previous percentage threshold allowed mixed
                # lines such as "có ra dáng长辈 không?" to pass because the
                # untranslated tail was small relative to the whole sentence.
                untranslated = [item_id for item_id, text in mapping.items() if _CJK_RE.search(text)]
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
                        "Shorten these lines for Vietnamese romance-drama dubbing. Natural, faithful Vietnamese is more important than literal wording. "
                        f"Required ids, in order: {overlong}. Return each required id exactly once. "
                        "Your entire response must be one JSON object with this exact structure: "
                        '{"translations":[{"id":1,"text":"Vietnamese dialogue"}]}. '
                        "Replace the example values with the results; never change the field names.\n"
                        "INPUT:\n" + json.dumps(shorten_items, ensure_ascii=False)
                    )
                    try:
                        shortened = _call_json_provider(primary_provider, SHORTEN_SYSTEM_PROMPT, shorten_prompt, overlong)
                        empty_shortened = [item_id for item_id, text in shortened.items() if not text.strip()]
                        if empty_shortened:
                            raise ValueError(f"Provider returned empty shortened dialogue for ids: {empty_shortened}")
                        mapping.update(shortened)
                    except (ValueError, requests.RequestException) as exc:
                        # Shortening is an optional timing optimization. The main
                        # translation above is already valid, so a malformed or
                        # transient shortening response must not fail the episode.
                        # Dialogue reflow and TTS fitting have the full sentence
                        # context and can safely handle the original wording.
                        if progress:
                            progress(
                                "Provider shortening was unusable; retaining the "
                                f"validated translation for dialogue timing: {exc}"
                            )
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
                for cue in batch:
                    mapping.setdefault(cue.index, "")
                mapping = _enforce_punctuation(mapping, items, primary_provider)
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
        if target_language.lower().startswith("vietnam"):
            source_by_id = {cue.index: cue.content for cue in subtitles}
            translated = [
                srt.Subtitle(
                    index=cue.index, start=cue.start, end=cue.end,
                    content=_append_ending_punctuation(cue.content, source_by_id.get(cue.index, ""))
                )
                if not is_ignorable_asr_fragment(cue) else cue
                for cue in translated
            ]
        if job_dir:
            atomic_write_json(job_dir / "vi-final.json", [{"id": cue.index, "text": cue.content} for cue in translated])
        return translated

    # Review coherent chunks with overlap-free boundaries. This second pass fixes
    # pronouns and continuity after all draft lines are available.
    final_mapping = {cue.index: cue.content for cue in translated}
    review_size = 30
    for offset in range(0, len(translated), review_size):
        scene = translated[offset : offset + review_size]
        review_items = [
            {
                "id": cue.index, "text": cue.content,
                "speaker": speaker_aliases.get((speakers or {}).get(cue.index)) if has_any_speaker else None,
                "max_words": max(2, min(24, round((cue.end - cue.start).total_seconds() * 3.2))),
            }
            for cue in scene
        ]
        previous = translated[max(0, offset - 5):offset]
        user = "Edit this scene:\n" + json.dumps(review_items, ensure_ascii=False)
        scene_note = _speaker_note()
        if scene_note:
            user += "\n\nContext note:\n" + scene_note
        if previous:
            user += "\nPrevious dialogue for context only:\n" + json.dumps(
                [{"id": cue.index, "text": final_mapping[cue.index]} for cue in previous], ensure_ascii=False
            )
        last_error: Exception | None = None
        for attempt in range(1, settings.translation_retries + 1):
            try:
                raw = _gemini(SCENE_EDITOR_PROMPT, user) if primary_provider == "gemini" else _openai_compatible(primary_provider, SCENE_EDITOR_PROMPT, user)
                reviewed = parse_translation_json(raw, [cue.index for cue in scene])
                empty_reviewed = [
                    cue.index for cue in scene
                    if not is_ignorable_asr_fragment(cue) and not reviewed[cue.index].strip()
                ]
                if empty_reviewed:
                    raise ValueError(f"Scene editor returned empty dialogue for ids: {empty_reviewed}")
                limits = {item["id"]: item["max_words"] for item in review_items}
                too_long = [item_id for item_id, text in reviewed.items() if len(text.split()) > limits[item_id] + 2]
                if too_long:
                    raise ValueError(f"Scene editor exceeded word budget for ids: {too_long}")
                scene_items = [
                    {"id": cue.index, "text": source_text, "speaker": scene_alias}
                    for cue, source_text, scene_alias in [
                        (cue, cue.content, speaker_aliases.get((speakers or {}).get(cue.index)) if has_any_speaker else None)
                        for cue in translated[offset : offset + review_size]
                    ]
                ]
                reviewed = _enforce_punctuation(reviewed, scene_items, primary_provider)
                final_mapping.update(reviewed)
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
    untranslated_final = [cue.index for cue in final if _CJK_RE.search(cue.content)]
    if target_language.lower().startswith("vietnam") and untranslated_final:
        raise RuntimeError(f"Final Vietnamese scene still contains Chinese text for ids: {untranslated_final}")
    if job_dir:
        atomic_write_json(job_dir / "vi-final.json", [{"id": cue.index, "text": cue.content} for cue in final])
    return final
