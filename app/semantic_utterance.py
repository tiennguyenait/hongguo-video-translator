"""Build complete meaningful sentences for TTS independently of display cues."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

import srt


@dataclass(slots=True)
class SemanticUtterance:
    id: int
    cue_ids: list[int]
    source_text: str
    text: str
    speaker: str | None
    start: float
    end: float
    source: str = "fallback"

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You are a senior Vietnamese dubbing dialogue editor.
Group adjacent subtitle fragments into complete, natural spoken utterances and rewrite their
Vietnamese text as coherent dialogue.
Rules:
- Return valid JSON only as {"utterances":[{"cue_ids":[1,2],"text":"..."}]}.
- Use every input cue id exactly once, in the original order. Groups must be contiguous.
- Never cross a hard_break or two different known speakers.
- Merge lowercase tails, orphan words, and ASR fragments into the sentence they complete.
- The Vietnamese translation fields are only rough drafts and may be semantically wrong.
  Retranslate from the complete ordered Chinese source sequence, using neighboring cues.
- Chinese forced alignment may place the final character of one sentence at the beginning of
  the next cue. Repair that boundary by meaning. cue_ids control timing/coverage only; they do
  not prevent meaning from being redistributed between adjacent output groups.
- Never merely concatenate drafts when the result is unnatural. For example, a dangling
  "chúng ta", "rồi", or a name created from a boundary character must be rewritten into the
  correct complete sentence.
- Preserve plot facts, names, numbers, relationships and emotional intent.
- Use natural spoken Vietnamese, not literal Chinese syntax.
- Place possessive phrases in natural Vietnamese order (for example "môn phái X của các ngươi"),
  never append a name as an unnatural afterthought such as "các ngươi, X".
- Pattern example: "今天便是你们青云宗灭门之日" means
  "Hôm nay chính là ngày Thanh Vân Tông của các ngươi bị diệt môn.", not
  "các ngươi đã diệt Thanh Vân Tông" and not "ngày diệt môn của các ngươi, Thanh Vân Tông".
- Every utterance must be a complete meaningful phrase; no standalone fragments such as
  "chúng ta", "rồi", "khi rút kiếm" when they complete a neighboring sentence.
- Keep dialogue concise enough for the supplied total duration.
- Do not add explanations or timestamps."""


def _compact_source(text: str) -> str:
    return re.sub(r"[\s\W_]", "", text, flags=re.UNICODE)


def _fallback_groups(
    translated: list[srt.Subtitle], source: list[srt.Subtitle], speakers: dict[int, str],
) -> list[list[int]]:
    source_by_id = {cue.index: cue for cue in source}
    groups: list[list[int]] = []
    for cue in translated:
        if not groups:
            groups.append([cue.index])
            continue
        previous = next(item for item in translated if item.index == groups[-1][-1])
        gap = (cue.start - previous.end).total_seconds()
        previous_speaker, speaker = speakers.get(previous.index), speakers.get(cue.index)
        speaker_compatible = not previous_speaker or not speaker or previous_speaker == speaker
        previous_text = previous.content.strip()
        continuation = bool(re.search(r'(?:\.{2,}|…+)\s*$', previous_text))
        previous_complete = bool(not continuation and re.search(r'[.!?]["”’)]?\s*$', previous_text))
        source_fragment = len(_compact_source(source_by_id.get(cue.index, cue).content)) <= 2
        vietnamese_fragment = len(re.findall(r"\b[\wÀ-ỹ]+\b", cue.content, re.UNICODE)) <= 3
        lowercase_tail = bool(cue.content.strip()[:1].islower())
        if speaker_compatible and gap <= 2.1 and not previous_complete and (source_fragment or vietnamese_fragment or lowercase_tail):
            groups[-1].append(cue.index)
        else:
            groups.append([cue.index])
    return groups


def _parse_groups(raw: str, expected_ids: list[int], hard_breaks: set[tuple[int, int]]) -> list[tuple[list[int], str]]:
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    payload = json.loads(clean)
    items = payload.get("utterances") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Semantic reflow must return an utterances array")
    result, flattened = [], []
    positions = {item_id: index for index, item_id in enumerate(expected_ids)}
    for item in items:
        ids = [int(value) for value in item.get("cue_ids", [])]
        text = str(item.get("text", "")).strip()
        if not ids or not text or any(item_id not in positions for item_id in ids):
            raise ValueError("Every semantic utterance needs valid cue_ids and text")
        if ids != expected_ids[positions[ids[0]]:positions[ids[0]] + len(ids)]:
            raise ValueError("Semantic cue groups must be contiguous and ordered")
        if any((left, right) in hard_breaks for left, right in zip(ids, ids[1:])):
            raise ValueError("Semantic utterance crosses a hard speaker/scene break")
        result.append((ids, text))
        flattened.extend(ids)
    if flattened != expected_ids:
        raise ValueError("Semantic reflow must preserve every cue id exactly once")
    return result


def build_semantic_utterances(
    translated: list[srt.Subtitle], source: list[srt.Subtitle], speakers: dict[int, str], provider: str,
    request: Callable[[str, str, str], str] | None = None,
) -> tuple[list[SemanticUtterance], str | None]:
    source_by_id, translated_by_id = {cue.index: cue for cue in source}, {cue.index: cue for cue in translated}
    expected_ids = [cue.index for cue in translated]
    hard_breaks: set[tuple[int, int]] = set()
    items = []
    for index, cue in enumerate(translated):
        previous = translated[index - 1] if index else None
        gap = (cue.start - previous.end).total_seconds() if previous else 0.0
        previous_speaker = speakers.get(previous.index) if previous else None
        speaker = speakers.get(cue.index)
        hard_break = bool(previous and ((previous_speaker and speaker and previous_speaker != speaker) or gap >= 2.8))
        if hard_break:
            hard_breaks.add((previous.index, cue.index))
        items.append({
            "id": cue.index, "source": source_by_id.get(cue.index, cue).content,
            "speaker": speaker, "gap_before_seconds": round(gap, 3),
            "hard_break": hard_break,
            "duration_seconds": round((cue.end-cue.start).total_seconds(), 3),
        })
    warning = None
    groups: list[tuple[list[int], str]]
    try:
        if request is None:
            from .translator import _gemini, _openai_compatible
            request = lambda selected, system, user: (
                _gemini(system, user) if selected == "gemini" else _openai_compatible(selected, system, user)
            )
        full_source = "".join(item["source"] for item in items)
        user_prompt = (
            "Complete Chinese scene (authoritative meaning):\n" + full_source
            + "\nMandatory hard boundaries (never place both ids in one group):\n"
            + json.dumps(sorted([list(pair) for pair in hard_breaks]))
            + "\nIf a Chinese boundary character belongs to the sentence before a hard boundary, "
              "use its meaning in the earlier Vietnamese group while keeping its cue id in the later group."
            + "\n\nEdit these timed cue fragments:\n" + json.dumps(items, ensure_ascii=False)
        )
        last_error = None
        for attempt in range(2):
            try:
                raw = request(provider, SYSTEM_PROMPT, user_prompt)
                groups = _parse_groups(raw, expected_ids, hard_breaks)
                break
            except Exception as exc:
                last_error = exc
                user_prompt += (
                    "\n\nRETRY: The previous response was invalid: " + str(exc)
                    + ". Obey every mandatory hard boundary and return all cue ids exactly once."
                )
        else:
            raise ValueError(f"Semantic reflow failed validation twice: {last_error}")
        semantic_source = "ai"
    except Exception as exc:
        warning = f"Semantic reflow unavailable; used deterministic fallback: {exc}"
        fallback = _fallback_groups(translated, source, speakers)
        groups = [(ids, " ".join(translated_by_id[item_id].content for item_id in ids)) for ids in fallback]
        semantic_source = "fallback"
    utterances = []
    for ids, text in groups:
        cues = [translated_by_id[item_id] for item_id in ids]
        labels = [speakers[item_id] for item_id in ids if item_id in speakers]
        speaker = max(set(labels), key=labels.count) if labels else None
        utterances.append(SemanticUtterance(
            id=ids[0], cue_ids=ids,
            source_text="".join(source_by_id.get(item_id, translated_by_id[item_id]).content for item_id in ids),
            text=text, speaker=speaker, start=cues[0].start.total_seconds(), end=cues[-1].end.total_seconds(),
            source=semantic_source,
        ))
    return utterances, warning
