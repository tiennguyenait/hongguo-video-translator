"""Semantic dialogue units and lightweight scene context."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import srt


def repair_fragment_speakers(cues: list[srt.Subtitle], speakers: dict[int, str]) -> dict[int, str]:
    """Diarization often mislabels a final 1–2 CJK-character ASR fragment."""
    repaired = dict(speakers)
    for index in range(1, len(cues)):
        cue, previous = cues[index], cues[index - 1]
        compact = re.sub(r"[\s\W_]", "", cue.content)
        gap = (cue.start - previous.end).total_seconds()
        if 0 < len(compact) <= 2 and gap <= 0.18 and previous.index in repaired:
            repaired[cue.index] = repaired[previous.index]
    return repaired


@dataclass(slots=True)
class DialogueUnit:
    id: int
    cue_ids: list[int]
    speaker: str | None
    start: float
    end: float
    source_text: str
    scene_id: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_dialogue_units(cues: list[srt.Subtitle], speakers: dict[int, str]) -> list[DialogueUnit]:
    """Join ASR fragments without crossing speakers, sentences, long pauses, or scenes."""
    result: list[DialogueUnit] = []
    current: list[srt.Subtitle] = []
    current_speaker: str | None = None
    scene_id = 1

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = " ".join(re.sub(r"\s+", " ", cue.content).strip() for cue in current).strip()
        result.append(DialogueUnit(
            id=current[0].index, cue_ids=[cue.index for cue in current], speaker=current_speaker,
            start=current[0].start.total_seconds(), end=current[-1].end.total_seconds(),
            source_text=text, scene_id=scene_id,
        ))
        current = []

    for cue in cues:
        speaker = speakers.get(cue.index)
        gap = (cue.start - current[-1].end).total_seconds() if current else 0.0
        prior_complete = bool(current and re.search(r"[.!?。！？][\"”’)]?\s*$", current[-1].content.strip()))
        scene_break = bool(current and gap >= 2.2)
        can_join = bool(
            current and speaker == current_speaker and not prior_complete and gap <= 0.65
            and (cue.end - current[0].start).total_seconds() <= 8.0
            and sum(len(item.content) for item in current) + len(cue.content) <= 120
        )
        if current and not can_join:
            flush()
            if scene_break:
                scene_id += 1
        if not current:
            current_speaker = speaker
        current.append(cue)
    flush()
    return result
