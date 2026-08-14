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


def _rounded_rectangle_path(x: int, y: int, width: int, height: int, radius: int) -> str:
    radius = max(2, min(radius, width // 2, height // 2))
    k = 0.55228475
    kr = round(radius * k)
    right, bottom = x + width, y + height
    return (
        f"m {x + radius} {y} l {right - radius} {y} "
        f"b {right - radius + kr} {y} {right} {y + radius - kr} {right} {y + radius} "
        f"l {right} {bottom - radius} "
        f"b {right} {bottom - radius + kr} {right - radius + kr} {bottom} {right - radius} {bottom} "
        f"l {x + radius} {bottom} "
        f"b {x + radius - kr} {bottom} {x} {bottom - radius + kr} {x} {bottom - radius} "
        f"l {x} {y + radius} "
        f"b {x} {y + radius - kr} {x + radius - kr} {y} {x + radius} {y}"
    )


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
Style: Background,{FONT_NAME},1,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    background_events: list[str] = []
    text_events: list[str] = []
    report: list[dict] = []
    previous_size: int | None = None
    for cue in subtitles:
        midpoint = (cue.start.total_seconds() + cue.end.total_seconds()) / 2
        matching = [region for region in regions if region.start <= midpoint <= region.end]
        region = matching[0] if matching else min(regions, key=lambda item: min(abs(midpoint - item.start), abs(midpoint - item.end)))
        available_width = round(region.width * 0.90)
        available_height = round(region.height * 0.82)
        fitted = fit_text(cue.content.strip(), available_width, available_height, max_font_size=round(video_height * 0.047))
        # Quantized sizes avoid distracting frame-to-frame pumping. Adjacent cues
        # may shrink for long lines but never jump upward by more than one level.
        if previous_size is not None and fitted.font_size > previous_size + 2:
            fitted.font_size = previous_size + 2
        previous_size = fitted.font_size
        x, y = region.x + region.width // 2, region.y + region.height // 2
        font = _font(fitted.font_size)
        rendered_width = max(_text_width(line, font) for line in fitted.lines)
        padding_x = max(18, round(video_width * 0.014))
        padding_y = max(8, round(video_height * 0.010))
        background_width = min(region.width, max(round(region.width * 0.38), round(rendered_width + padding_x * 2)))
        line_height = fitted.font_size * 1.22
        background_height = min(region.height, max(round(video_height * 0.073), round(line_height * len(fitted.lines) + padding_y * 2)))
        background_x, background_y = x - background_width // 2, y - background_height // 2
        radius = max(8, round(background_height * 0.18))
        path = _rounded_rectangle_path(background_x, background_y, background_width, background_height, radius)
        background_events.append(
            f"Dialogue: 0,{_ass_time(cue.start.total_seconds())},{_ass_time(cue.end.total_seconds())},Background,,0,0,0,,"
            rf"{{\an7\pos(0,0)\p1\1c&H000000&\bord0\shad0}}{path}"
        )
        text = r"\N".join(_escape_ass(line) for line in fitted.lines)
        override = rf"{{\an5\pos({x},{y})\fs{fitted.font_size}\b600}}"
        text_events.append(f"Dialogue: 1,{_ass_time(cue.start.total_seconds())},{_ass_time(cue.end.total_seconds())},Adaptive,,0,0,0,,{override}{text}")
        report.append({
            "id": cue.index, "font_size": fitted.font_size, "lines": fitted.lines, "x": x, "y": y,
            "background": {"x": background_x, "y": background_y, "width": background_width, "height": background_height, "radius": radius},
            "region": {"x": region.x, "y": region.y, "width": region.width, "height": region.height},
        })
    output.write_text(header + "\n".join(background_events + text_events) + "\n", encoding="utf-8")
    if report_path:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
