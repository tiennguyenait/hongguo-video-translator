import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import srt


def seconds_to_timedelta(seconds: float) -> timedelta:
    return timedelta(milliseconds=round(seconds * 1000))


def segments_to_subtitles(segments: Iterable[Any]) -> list[srt.Subtitle]:
    return [
        srt.Subtitle(index=i, start=seconds_to_timedelta(seg.start), end=seconds_to_timedelta(seg.end), content=seg.text.strip())
        for i, seg in enumerate(segments, start=1)
    ]


def write_srt(path: Path, subtitles: list[srt.Subtitle]) -> None:
    # ASR can legitimately produce an empty punctuation/noise cue. Reindexing
    # after srt drops that cue shifts every later id and makes voice-aligned QA
    # report a false mismatch. Keep source-owned ids stable; empty cues remain
    # represented by an empty SRT block, while every later cue retains its id.
    path.write_text(srt.compose(subtitles, reindex=False), encoding="utf-8")


def is_ignorable_asr_fragment(cue: srt.Subtitle) -> bool:
    """Return true only for tiny ASR artifacts that may safely be silent."""
    compact = re.sub(r"[^\w\u3400-\u9fff]", "", cue.content, flags=re.UNICODE)
    duration = (cue.end - cue.start).total_seconds()
    return len(compact) <= 1 or (
        duration <= 0.5 and re.fullmatch(r"[A-Za-z]{1,3}", compact) is not None
    )


def read_srt(path: Path) -> list[srt.Subtitle]:
    return list(srt.parse(path.read_text(encoding="utf-8-sig")))


def parse_translation_json(raw: str, expected_ids: list[int], allow_missing: bool = False) -> dict[int, str]:
    cleaned = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Some chat models emit valid JSON objects one per line instead of the
        # requested array. Decode a sequence of complete JSON values safely;
        # never attempt regex/eval repair of arbitrary model output.
        decoder = json.JSONDecoder()
        values = []
        position = 0
        try:
            while position < len(cleaned):
                while position < len(cleaned) and cleaned[position].isspace():
                    position += 1
                if position >= len(cleaned):
                    break
                value, position = decoder.raw_decode(cleaned, position)
                values.append(value)
        except json.JSONDecodeError as sequence_exc:
            raise ValueError(f"Provider returned invalid JSON: {sequence_exc.msg}") from sequence_exc
        if not values:
            raise ValueError(f"Provider returned invalid JSON: {exc.msg}") from exc
        payload = values if len(values) > 1 else values[0]
    if isinstance(payload, dict):
        payload = payload.get("translations", payload)
    if isinstance(payload, dict):
        if "id" in payload and any(key in payload for key in ("text", "translation", "translated_text", "translatedText")):
            payload = [payload]
        else:
            payload = [{"id": key, "text": value} for key, value in payload.items()]
    if not isinstance(payload, list):
        raise ValueError("Translation response must be an array or contain a translations array")
    result: dict[int, str] = {}
    for item in payload:
        if isinstance(item, dict) and "id" not in item:
            for alias in ("cue_id", "cueId", "index"):
                if alias in item:
                    item = {**item, "id": item[alias]}
                    break
        if isinstance(item, dict) and "text" not in item:
            for alias in (
                "translation", "translated_text", "translatedText", "translated",
                "vietnamese", "shortened_text", "shortenedText", "content", "output",
            ):
                value = item.get(alias)
                if isinstance(value, str):
                    item = {**item, "text": value}
                    break
                if isinstance(value, dict) and isinstance(value.get("text"), str):
                    item = {**item, "text": value["text"]}
                    break
        if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("text"), str):
            raise ValueError("Every translation item must contain id and text")
        try:
            item_id = int(item["id"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Translation id must be an integer") from exc
        if item_id in result:
            raise ValueError(f"Duplicate translation id: {item_id}")
        result[item_id] = item["text"].strip()
    missing = sorted(set(expected_ids) - set(result))
    extra = sorted(set(result) - set(expected_ids))
    if extra or (missing and not allow_missing):
        raise ValueError(f"Translation ids mismatch; missing={missing}, extra={extra}")
    return result
