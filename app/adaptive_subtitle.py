"""Generate ASS events that fit Vietnamese text inside detected mask regions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import srt
from PIL import ImageFont

from .source_subtitle_mask import SubtitleRegion

UBUNTU_FONT_PATH = Path("/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf")
FALLBACK_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_PATH = UBUNTU_FONT_PATH if UBUNTU_FONT_PATH.is_file() else FALLBACK_FONT_PATH
FONT_NAME = "Ubuntu Sans" if FONT_PATH == UBUNTU_FONT_PATH else "DejaVu Sans"


@dataclass(slots=True)
class FittedText:
    lines: list[str]
    font_size: int


def _text_width(text: str, font: ImageFont.FreeTypeFont) -> float:
    left, _, right, _ = font.getbbox(text)
    return right - left


def _font(size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONT_PATH), size)
    if FONT_PATH == UBUNTU_FONT_PATH:
        try:
            font.set_variation_by_name("SemiBold")
        except OSError:
            pass
    return font


def fit_text(text: str, width: int, height: int, max_font_size: int) -> FittedText:
    words = text.split()
    # A single subtitle line may comfortably occupy about 70% of the usable
    # band height; the per-layout check below still protects two-line cues.
    max_size = max(14, min(max_font_size, round(height * 0.72)))
    max_size -= max_size % 2
    for size in range(max_size, 11, -2):
        font = _font(size)
        line_height = size * 1.22
        if _text_width(text, font) <= width and line_height <= height:
            return FittedText([text], size)
        if len(words) >= 2 and line_height * 2 <= height:
            splits = [(words[:index], words[index:]) for index in range(1, len(words))]
            first, second = min(splits, key=lambda pair: max(_text_width(" ".join(pair[0]), font), _text_width(" ".join(pair[1]), font)))
            lines = [" ".join(first), " ".join(second)]
            if max(_text_width(line, font) for line in lines) <= width:
                return FittedText(lines, size)
    return FittedText([text], 12)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def generate_adaptive_ass(
    subtitles: list[srt.Subtitle], regions: list[SubtitleRegion], video_width: int,
    video_height: int, output: Path, report_path: Path | None = None,
) -> list[dict]:
    if not regions:
        raise ValueError("Adaptive subtitle layout requires at least one mask region")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Adaptive,{FONT_NAME},28,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1.3,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    report: list[dict] = []
    previous_size: int | None = None
    for cue in subtitles:
        midpoint = (cue.start.total_seconds() + cue.end.total_seconds()) / 2
        matching = [region for region in regions if region.start <= midpoint <= region.end]
        region = matching[0] if matching else min(regions, key=lambda item: min(abs(midpoint - item.start), abs(midpoint - item.end)))
        available_width = round(region.width * 0.90)
        available_height = round(region.height * 0.82)
        fitted = fit_text(cue.content.strip(), available_width, available_height, max_font_size=round(video_height * 0.052))
        # Quantized sizes avoid distracting frame-to-frame pumping. Adjacent cues
        # may shrink for long lines but never jump upward by more than one level.
        if previous_size is not None and fitted.font_size > previous_size + 2:
            fitted.font_size = previous_size + 2
        previous_size = fitted.font_size
        x, y = region.x + region.width // 2, region.y + region.height // 2
        text = r"\N".join(_escape_ass(line) for line in fitted.lines)
        override = rf"{{\an5\pos({x},{y})\fs{fitted.font_size}\b600}}"
        events.append(f"Dialogue: 0,{_ass_time(cue.start.total_seconds())},{_ass_time(cue.end.total_seconds())},Adaptive,,0,0,0,,{override}{text}")
        report.append({"id": cue.index, "font_size": fitted.font_size, "lines": fitted.lines, "x": x, "y": y, "region": {"x": region.x, "y": region.y, "width": region.width, "height": region.height}})
    output.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    if report_path:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
