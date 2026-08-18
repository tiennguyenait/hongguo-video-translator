"""One validated dialogue source shared by display subtitles and TTS."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass

import srt

from .config import get_settings


@dataclass(slots=True)
class MasterUtterance:
    id: int
    cue_ids: list[int]
    full_text: str
    start: float
    end: float
    speaker: str | None

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You are the final Vietnamese dialogue editor for dubbing Chinese romance short dramas.
Create one master dialogue that is used by BOTH subtitles and TTS.
Return JSON only as:
{"utterances":[{"cue_ids":[1,2],"full_text":"complete sentence",
"display_lines":[{"id":1,"text":"line one"},{"id":2,"text":"line two"}]}]}.
Rules:
- Use every input cue id exactly once and in order. Groups and ids must be contiguous.
- Return exactly one non-empty display line for every original cue id; preserve the original
  number of display lines and their timing ownership.
- Concatenating display_lines in order must contain exactly the same Vietnamese words in the
  same order as full_text. Only whitespace and punctuation may differ.
- full_text must be a complete, natural spoken Vietnamese sentence/utterance for TTS.
- Redistribute wording across adjacent display lines so no line is an orphan such as
  "chúng ta", "rồi", or "khi rút kiếm" when it completes a neighboring sentence.
- Never cross a mandatory hard boundary or two different known speakers.
- Chinese source is authoritative. Draft Vietnamese is a terminology hint and may be wrong.
  Repair ASR boundary characters using the complete source scene and neighboring cues.
- Keep romance-drama tone natural for Vietnamese viewers: emotional tension, jealousy, family pressure,
  class/status conflict and relationship distance must sound like spoken Vietnamese, not literal Chinese.
- Preserve the original plot, timing ownership and character intent. Do not add new facts or exposition.
- A Chinese character at the start of a cue may grammatically finish the previous sentence even
  across a mandatory timing boundary. Move its MEANING to the previous Vietnamese group while
  keeping the cue id itself in the later group; do not invent a name from that dangling character.
- Preserve facts, names, numbers, relationships, emotion and established Vietnamese names.
- Keep each display line concise for its duration and source box width.
- Pattern example: "今天便是你们青云宗灭门之日" means
  "Hôm nay chính là ngày Thanh Vân Tông của các ngươi bị diệt môn."
- In "X宗灭门之日", X宗 is the sect being annihilated. Never reverse subject/object into
  "các ngươi đã diệt môn phái X". Prefer "ngày X Tông của các ngươi bị diệt môn"."""


def _lexical_key(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[\wÀ-ỹ]+", normalized, re.UNICODE)


def _balanced_lines(text: str, count: int) -> list[str]:
    """Split without changing words, while guaranteeing one visible line per cue."""
    words = text.split()
    if count < 1 or len(words) < count:
        raise ValueError("Full dialogue has too few words for non-empty display lines")
    base, remainder = divmod(len(words), count)
    lines, cursor = [], 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        lines.append(" ".join(words[cursor:cursor + size]))
        cursor += size
    return lines


def _split_at_hard_boundaries(group: dict, hard_breaks: set[tuple[int, int]]) -> list[dict]:
    """Diarization boundaries override an LLM group, without losing its edited words."""
    cue_ids, texts = group["cue_ids"], group["display_texts"]
    parts, start = [], 0
    for position, pair in enumerate(zip(cue_ids, cue_ids[1:]), start=1):
        if pair in hard_breaks:
            ids, lines = cue_ids[start:position], texts[start:position]
            parts.append({"cue_ids": ids, "full_text": " ".join(lines), "display_texts": lines})
            start = position
    ids, lines = cue_ids[start:], texts[start:]
    parts.append({"cue_ids": ids, "full_text": " ".join(lines), "display_texts": lines})
    return parts


def _join_spoken_text(left: str, right: str, source_left: str) -> str:
    """Remove display-cue punctuation that would create a false TTS stop."""
    left = left.rstrip()
    if not re.search(r"[。！？!?]\s*$", source_left.strip()):
        left = re.sub(r"[.!?。！？]+\s*$", "", left).rstrip()
        if re.search(r"[,，、]\s*$", source_left.strip()) and not left.endswith(","):
            left += ","
    return f"{left} {right.lstrip()}".strip()


def _coalesce_speech_groups(
    groups: list[dict], draft: list[srt.Subtitle], source: list[srt.Subtitle],
    speakers: dict[int, str], hard_breaks: set[tuple[int, int]],
) -> list[dict]:
    """Build sentence-level TTS units independently from timed display lines."""
    if not groups:
        return []
    draft_by_id = {cue.index: cue for cue in draft}
    source_by_id = {cue.index: cue for cue in source}
    result: list[dict] = []
    current = {**groups[0], "cue_ids": list(groups[0]["cue_ids"]), "display_texts": list(groups[0]["display_texts"])}
    for group in groups[1:]:
        previous_id, next_id = current["cue_ids"][-1], group["cue_ids"][0]
        previous_cue, next_cue = draft_by_id[previous_id], draft_by_id[next_id]
        gap = (next_cue.start - previous_cue.end).total_seconds()
        previous_speaker, next_speaker = speakers.get(previous_id), speakers.get(next_id)
        speaker_changed = bool(previous_speaker and next_speaker and previous_speaker != next_speaker)
        source_left = source_by_id.get(previous_id, previous_cue).content
        source_sentence_ended = bool(re.search(r"[。！？!?][\"”’)]?\s*$", source_left.strip()))
        first_cue = draft_by_id[current["cue_ids"][0]]
        combined_span = (draft_by_id[group["cue_ids"][-1]].end - first_cue.start).total_seconds()
        combined_chars = len(current["full_text"]) + len(group["full_text"])
        can_join = (
            (previous_id, next_id) not in hard_breaks
            and not speaker_changed and not source_sentence_ended and gap <= 0.95
            and combined_span <= 14.0 and combined_chars <= 220
        )
        if can_join:
            current["full_text"] = _join_spoken_text(current["full_text"], group["full_text"], source_left)
            current["cue_ids"].extend(group["cue_ids"])
            current["display_texts"].extend(group["display_texts"])
        else:
            result.append(current)
            current = {**group, "cue_ids": list(group["cue_ids"]), "display_texts": list(group["display_texts"])}
    result.append(current)
    return result


def _parse_master(raw: str, expected_ids: list[int], hard_breaks: set[tuple[int, int]]) -> list[dict]:
    clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    payload = json.loads(clean)
    items = payload.get("utterances") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("Dialogue master must contain utterances")
    flattened: list[int] = []
    positions = {item_id: index for index, item_id in enumerate(expected_ids)}
    result = []
    for item in items:
        cue_ids = [int(value) for value in item.get("cue_ids", [])]
        full_text = str(item.get("full_text", "")).strip()
        display = item.get("display_lines")
        if not cue_ids or not full_text or not isinstance(display, list):
            raise ValueError("Every utterance requires cue_ids, full_text and display_lines")
        if cue_ids[0] not in positions or cue_ids != expected_ids[positions[cue_ids[0]]:positions[cue_ids[0]] + len(cue_ids)]:
            raise ValueError("Dialogue groups must be contiguous and ordered")
        display_ids = [int(line.get("id", -1)) for line in display if isinstance(line, dict)]
        display_texts = [str(line.get("text", "")).strip() for line in display if isinstance(line, dict)]
        if display_ids != cue_ids or len(display_texts) != len(cue_ids):
            raise ValueError("Display lines must map one-to-one to cue_ids")
        if any(not text for text in display_texts):
            display_texts = _balanced_lines(full_text, len(cue_ids))
        if _lexical_key(" ".join(display_texts)) != _lexical_key(full_text):
            raise ValueError("Display lines and full_text must contain identical ordered words")
        result.extend(_split_at_hard_boundaries(
            {"cue_ids": cue_ids, "full_text": full_text, "display_texts": display_texts}, hard_breaks,
        ))
        flattened.extend(cue_ids)
    if flattened != expected_ids:
        raise ValueError("Dialogue master must preserve every cue id exactly once")
    return result


def build_dialogue_master(
    draft: list[srt.Subtitle], source: list[srt.Subtitle], speakers: dict[int, str], provider: str,
    box_widths: dict[int, int] | None = None,
    request: Callable[[str, str, str], str] | None = None,
) -> tuple[list[srt.Subtitle], list[MasterUtterance], str | None]:
    expected_ids = [cue.index for cue in draft]
    source_by_id = {cue.index: cue for cue in source}
    hard_breaks: set[tuple[int, int]] = set()
    prompt_items = []
    for index, cue in enumerate(draft):
        previous = draft[index - 1] if index else None
        gap = (cue.start-previous.end).total_seconds() if previous else 0.0
        previous_speaker = speakers.get(previous.index) if previous else None
        speaker = speakers.get(cue.index)
        hard = bool(previous and ((previous_speaker and speaker and previous_speaker != speaker) or gap >= 2.8))
        if hard:
            hard_breaks.add((previous.index, cue.index))
        prompt_items.append({
            "id": cue.index, "source": source_by_id.get(cue.index, cue).content,
            "draft_vietnamese": cue.content, "speaker": speaker,
            "gap_before_seconds": round(gap, 3), "hard_break": hard,
            "duration_seconds": round((cue.end-cue.start).total_seconds(), 3),
            "source_box_width": (box_widths or {}).get(cue.index),
        })
    warning = None
    # Translation already handles cross-cue continuity. Production therefore
    # keeps the validated display text and derives sentence groups locally.
    # Supplying request explicitly (tests/manual repair) or enabling the flag
    # preserves the expensive AI repair path for exceptional material.
    use_ai = request is not None or get_settings().dialogue_master_ai_repair
    if not use_ai:
        parsed = [
            {"cue_ids": [cue.index], "full_text": cue.content, "display_texts": [cue.content]}
            for cue in draft
        ]
    else:
        try:
            if request is None:
                from .translator import _gemini, _openai_compatible
                request = lambda selected, system, user: (
                    _gemini(system, user) if selected == "gemini" else _openai_compatible(selected, system, user)
                )
            user = (
                "Complete Chinese scene:\n" + "".join(item["source"] for item in prompt_items)
                + "\nMandatory hard boundaries:\n" + json.dumps(sorted([list(pair) for pair in hard_breaks]))
                + "\nTimed original display lines:\n" + json.dumps(prompt_items, ensure_ascii=False)
            )
            last_error = None
            for _ in range(2):
                try:
                    parsed = _parse_master(request(provider, SYSTEM_PROMPT, user), expected_ids, hard_breaks)
                    break
                except Exception as exc:
                    last_error = exc
                    user += "\nRETRY: Previous JSON was invalid: " + str(exc) + ". Fix it and obey all invariants."
            else:
                raise ValueError(f"Dialogue master validation failed twice: {last_error}")
        except Exception as exc:
            warning = f"Dialogue master unavailable; retained safe draft: {exc}"
            parsed = [{"cue_ids": [cue.index], "full_text": cue.content, "display_texts": [cue.content]} for cue in draft]
    display_mapping = {
        item_id: text for group in parsed for item_id, text in zip(group["cue_ids"], group["display_texts"], strict=True)
    }
    display = [srt.Subtitle(cue.index, cue.start, cue.end, display_mapping[cue.index]) for cue in draft]
    draft_by_id = {cue.index: cue for cue in draft}
    speech_groups = _coalesce_speech_groups(parsed, draft, source, speakers, hard_breaks)
    utterances = []
    for group in speech_groups:
        cues = [draft_by_id[item_id] for item_id in group["cue_ids"]]
        labels = [speakers[item_id] for item_id in group["cue_ids"] if item_id in speakers]
        utterances.append(MasterUtterance(
            id=group["cue_ids"][0], cue_ids=group["cue_ids"], full_text=group["full_text"],
            start=cues[0].start.total_seconds(), end=cues[-1].end.total_seconds(),
            speaker=max(set(labels), key=labels.count) if labels else None,
        ))
    return display, utterances, warning
