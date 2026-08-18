"""Automatic content, timing, and media validation."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import srt

from .artifacts import atomic_write_json
from .subtitle import is_ignorable_asr_fragment


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=index,codec_type,codec_name,sample_rate", "-of", "json", str(path)],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def validate_job(job_dir: Path, subtitles: list[srt.Subtitle], expected_ids: list[int], output_video: Path | None) -> dict:
    checks: list[dict] = []

    def add(name: str, severity: str, message: str) -> None:
        checks.append({"name": name, "severity": severity, "message": message})

    actual_ids = [cue.index for cue in subtitles]
    add("subtitle_ids", "pass" if actual_ids == expected_ids else "error", "All subtitle ids preserved" if actual_ids == expected_ids else "Subtitle ids are missing, duplicated, or reordered")
    empty = [cue.index for cue in subtitles if not cue.content.strip()]
    source_by_id: dict[int, srt.Subtitle] = {}
    source_path = job_dir / "source.srt"
    if source_path.is_file():
        source_by_id = {cue.index: cue for cue in srt.parse(source_path.read_text(encoding="utf-8-sig"))}
    ignorable_empty = [
        item_id for item_id in empty
        if item_id in source_by_id
        and is_ignorable_asr_fragment(source_by_id[item_id])
    ]
    unsafe_empty = [item_id for item_id in empty if item_id not in ignorable_empty]
    if unsafe_empty:
        add("empty_lines", "error", f"Empty subtitle ids: {unsafe_empty}")
    elif ignorable_empty:
        add("empty_lines", "warning", f"Ignored empty ASR punctuation/fragments: {ignorable_empty}")
    else:
        add("empty_lines", "pass", "No empty subtitles")
    chinese = [cue.index for cue in subtitles if re.search(r"[\u3400-\u9fff]", cue.content)]
    add(
        "untranslated_text", "pass" if not chinese else "error",
        "No Chinese text remains" if not chinese else f"Chinese text remains in Vietnamese subtitles: {chinese[:20]}",
    )
    invalid_timing = [cue.index for cue in subtitles if cue.start.total_seconds() < 0 or cue.end <= cue.start]
    add("subtitle_timing", "pass" if not invalid_timing else "error", "Subtitle timing is valid" if not invalid_timing else f"Invalid timings: {invalid_timing}")
    repeated = [subtitles[i].index for i in range(1, len(subtitles)) if subtitles[i].content.strip().lower() == subtitles[i - 1].content.strip().lower() and len(subtitles[i].content.strip()) > 4]
    add("adjacent_repetition", "pass" if not repeated else "warning", "No suspicious adjacent repetition" if not repeated else f"Review repeated lines: {repeated[:20]}")
    orphaned = [cue.index for cue in subtitles if len(re.findall(r"\b[\wÀ-ỹ]+\b", cue.content, re.UNICODE)) <= 2 and cue.content.strip().endswith(".")]
    add("orphan_fragments", "pass" if not orphaned else "warning", "No suspicious orphan subtitle fragments" if not orphaned else f"Review short standalone fragments: {orphaned[:20]}")

    timing_path = job_dir / "tts-timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        rushed = [item["id"] for item in timing if item.get("tempo", 1.0) > 1.12]
        overflow = [item["id"] for item in timing if item.get("overflow_ms", 0) > 150]
        timing_ids = [item_id for item in timing for item_id in item.get("cue_ids", [item.get("id")])]
        misaligned = [
            item.get("id") for item in timing
            if item.get("alignment_mode") != "punctuated_sentence"
            or item.get("schedule_shift_ms", 0) < 0
        ]
        add("tts_speed", "pass" if not rushed else "warning", "TTS speed is within natural limits" if not rushed else f"Review rushed utterances: {rushed[:20]}")
        add("tts_deadlines", "pass" if not overflow else "warning", "TTS clips fit their windows" if not overflow else f"Audio exceeds windows: {overflow[:20]}")
        add(
            "tts_cue_coverage", "pass" if timing_ids == expected_ids else "error",
            "Every subtitle cue belongs to exactly one punctuated TTS phrase" if timing_ids == expected_ids
            else "TTS phrases are missing, duplicated, or reorder subtitle cues",
        )
        add(
            "tts_cue_alignment", "pass" if not misaligned else "error",
            "No TTS phrase starts before its first source cue" if not misaligned
            else f"TTS phrases start before their source cues: {misaligned[:20]}",
        )
        aligned_path = job_dir / "vi-aligned.srt"
        if aligned_path.is_file():
            aligned = list(srt.parse(aligned_path.read_text(encoding="utf-8-sig")))
            aligned_ids = [cue.index for cue in aligned]
            invalid_aligned = [
                cue.index for index, cue in enumerate(aligned)
                if cue.end <= cue.start
                or (index and cue.start < aligned[index - 1].end)
            ]
            add(
                "tts_subtitle_sync",
                "pass" if aligned_ids == expected_ids and not invalid_aligned else "error",
                "Voice-aligned subtitles preserve every cue without overlap"
                if aligned_ids == expected_ids and not invalid_aligned
                else (
                    f"Voice-aligned ids differ: expected={expected_ids[:20]}, actual={aligned_ids[:20]}"
                    if aligned_ids != expected_ids
                    else f"Invalid voice-aligned subtitle cues: {invalid_aligned[:20]}"
                ),
            )

    master_path = job_dir / "dialogue-master.json"
    if master_path.is_file():
        master = json.loads(master_path.read_text(encoding="utf-8"))
        utterances = master.get("utterances", [])
        display = master.get("display_lines", [])
        display_ids = [item.get("id") for item in display]
        utterance_ids = [item_id for item in utterances for item_id in item.get("cue_ids", [])]
        exact_ids = display_ids == expected_ids and utterance_ids == expected_ids
        add("dialogue_master_ids", "pass" if exact_ids else "error", "Dialogue master preserves one display line per source cue" if exact_ids else "Dialogue master lost, duplicated, or reordered cue ids")
        display_by_id = {item.get("id"): item.get("text", "") for item in display}
        mismatched = []
        for item in utterances:
            shown = " ".join(display_by_id.get(item_id, "") for item_id in item.get("cue_ids", []))
            words = lambda text: re.findall(r"[\wÀ-ỹ]+", text.casefold(), re.UNICODE)
            if words(shown) != words(item.get("full_text", "")):
                mismatched.append(item.get("id"))
        add("dialogue_master_consistency", "pass" if not mismatched else "error", "Display lines and TTS share identical ordered words" if not mismatched else f"Display/TTS mismatch in utterances: {mismatched}")
        master_warning = master.get("warning")
        add("dialogue_master_provider", "pass" if not master_warning else "warning", "AI dialogue master validated" if not master_warning else master_warning)

    regions_path = job_dir / "subtitle-regions.json"
    if regions_path.is_file():
        mask = json.loads(regions_path.read_text(encoding="utf-8"))
        regions = mask.get("regions", [])
        confidence = min((item.get("confidence", 0) for item in regions), default=0)
        dimensions = mask.get("video", {})
        width, height = dimensions.get("width", 0), dimensions.get("height", 0)
        invalid_regions = [index for index, item in enumerate(regions) if (
            item.get("width", 0) <= 0 or item.get("height", 0) <= 0 or item.get("start", -1) < 0
            or item.get("end", 0) <= item.get("start", 0) or item.get("x", -1) < 0 or item.get("y", -1) < 0
            or item.get("x", 0) + item.get("width", 0) > width
            or item.get("y", 0) + item.get("height", 0) > height
        )]
        if invalid_regions or not regions:
            severity = "error"
        elif confidence < 0.5 or mask.get("method") == "lower_band_fallback":
            severity = "warning"
        else:
            severity = "pass"
        detail = f"; invalid tracks {invalid_regions}" if invalid_regions else ""
        add("source_subtitle_mask", severity, f"Detected {len(regions)} visual text tracks at confidence {confidence:.2f} ({mask.get('method')}){detail}")

    layout_path = job_dir / "subtitle-layout.json"
    if layout_path.is_file():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        invalid = [item["id"] for item in layout if not item.get("font") or not (12 <= item.get("font_size", 0) <= 80) or len(item.get("lines", [])) not in {1, 2} or item.get("background", {}).get("radius", 0) <= 0]
        add("adaptive_subtitle_layout", "pass" if not invalid else "error", f"Adaptive font/layout valid for {len(layout)} cues" if not invalid else f"Invalid adaptive layout ids: {invalid}")
        uncovered = []
        for item in layout:
            background, region = item.get("background", {}), item.get("region", {})
            if (background.get("x", 0) > region.get("x", 0)
                    or background.get("y", 0) > region.get("y", 0)
                    or background.get("x", 0) + background.get("width", 0) < region.get("x", 0) + region.get("width", 0)
                    or background.get("y", 0) + background.get("height", 0) < region.get("y", 0) + region.get("height", 0)):
                uncovered.append(item["id"])
        add("source_pixels_covered", "pass" if not uncovered else "error", "Every rendered mask contains its source-text bounds" if not uncovered else f"Mask can expose source pixels for ids: {uncovered}")

    if output_video and output_video.is_file():
        try:
            probe = _probe(output_video)
            types = {stream.get("codec_type") for stream in probe.get("streams", [])}
            duration = float(probe.get("format", {}).get("duration", 0))
            valid = duration > 0 and {"audio", "video"}.issubset(types)
            add("media_streams", "pass" if valid else "error", f"Output has audio/video; duration {duration:.3f}s" if valid else "Output is missing valid audio/video streams")
        except Exception as exc:
            add("media_streams", "error", f"ffprobe failed: {exc}")

    summary = {level: sum(item["severity"] == level for item in checks) for level in ("pass", "warning", "error")}
    report = {"summary": summary, "checks": checks}
    atomic_write_json(job_dir / "qa-report.json", report)
    return report
