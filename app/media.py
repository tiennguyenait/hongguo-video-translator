import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import srt

from .adaptive_subtitle import FONT_PATH, generate_adaptive_ass

if TYPE_CHECKING:
    from .source_subtitle_mask import SubtitleRegion

VIDEO_CRF = "23"
AUDIO_BITRATE = "128k"


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip()[-4000:]
        raise RuntimeError(f"FFmpeg failed (exit {result.returncode}): {detail}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return float(result.stdout.strip())


def probe_video_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        text=True, capture_output=True,
    )
    if result.returncode or "x" not in result.stdout:
        raise RuntimeError(f"ffprobe video size failed: {result.stderr.strip()}")
    width, height = result.stdout.strip().split("x", 1)
    return int(width), int(height)


def _subtitle_filter(path: Path, force_style: bool = True, fonts_dir: Path | None = None) -> str:
    escaped = str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace(",", "\\,")
    style = (
        "FontName=Arial,FontSize=11,PrimaryColour=&H00FFFFFF,"
        "BackColour=&H70000000,OutlineColour=&H00000000,"
        "BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginV=28"
    )
    suffix = f":force_style='{style}'" if force_style else ""
    if fonts_dir:
        escaped_fonts = str(fonts_dir.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace(",", "\\,")
        suffix += f":fontsdir='{escaped_fonts}'"
    return f"subtitles=filename='{escaped}'{suffix}"


def burn_subtitles(video: Path, subtitle: Path, output: Path, regions: list["SubtitleRegion"] | None = None) -> None:
    filters: list[str] = []
    if regions:
        width, height = probe_video_size(video)
        ass_path = output.parent / "vi.ass"
        generate_adaptive_ass(
            list(srt.parse(subtitle.read_text(encoding="utf-8-sig"))), regions, width, height,
            ass_path, output.parent / "subtitle-layout.json",
        )
        filters.append(_subtitle_filter(ass_path, force_style=False, fonts_dir=FONT_PATH.parent))
    else:
        filters.append(_subtitle_filter(subtitle))
    run_ffmpeg(["ffmpeg", "-y", "-i", str(video), "-vf", ",".join(filters), "-c:v", "libx264", "-preset", "medium", "-crf", VIDEO_CRF, "-c:a", "copy", "-movflags", "+faststart", str(output)])


def mux_dub(video: Path, wav: Path, output: Path, original_audio_volume: float) -> None:
    if original_audio_volume > 0:
        filters = f"[0:a]volume={original_audio_volume}[original];[1:a]volume=1.0[dub];[original][dub]amix=inputs=2:duration=longest:normalize=0[a]"
        command = ["ffmpeg", "-y", "-i", str(video), "-i", str(wav), "-filter_complex", filters, "-map", "0:v:0", "-map", "[a]"]
    else:
        command = ["ffmpeg", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0"]
    command += ["-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest", "-movflags", "+faststart", str(output)]
    run_ffmpeg(command)


def mux_delayed_clips(
    video: Path,
    clips: list[tuple[Path, int]],
    output: Path,
    original_audio_volume: float,
) -> None:
    """Place mono WAV clips at millisecond timestamps and mux without a PCM timeline."""
    if not clips:
        raise ValueError("At least one dubbed audio clip is required")
    duration = probe_duration(video)
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
    filters: list[str] = []
    labels: list[str] = []
    for number, (clip, start_ms) in enumerate(clips, 1):
        command += ["-i", str(clip)]
        label = f"clip{number}"
        filters.append(
            f"[{number}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=mono,"
            f"adelay={max(0, start_ms)}:all=1[{label}]"
        )
        labels.append(f"[{label}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[dubmix]")
    else:
        filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[dubmix]")
    filters.append(
        f"[dubmix]apad=whole_dur={duration:.6f},atrim=0:{duration:.6f},"
        "aresample=48000:async=1:first_pts=0,highpass=f=65,lowpass=f=12000,"
        "dynaudnorm=f=250:g=7:p=0.9[dub]"
    )
    output_label = "dub"
    if original_audio_volume > 0:
        filters.append("[dub]asplit=2[dub_sc][dub_mix]")
        filters.append(f"[0:a]aresample=48000:async=1:first_pts=0,volume={original_audio_volume}[original]")
        # Duck the source only while Vietnamese speech is active. This preserves
        # music/effects between lines and avoids a permanently muffled soundtrack.
        filters.append("[original][dub_sc]sidechaincompress=threshold=0.025:ratio=8:attack=18:release=280:makeup=1[ducked]")
        filters.append(
            f"[ducked][dub_mix]amix=inputs=2:duration=longest:normalize=0,atrim=0:{duration:.6f},"
            "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,alimiter=limit=0.94[mixed]"
        )
        output_label = "mixed"
    else:
        filters.append("[dub]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,alimiter=limit=0.94[master]")
        output_label = "master"
    script = output.parent / "dub-filter.ffscript"
    script.write_text(";\n".join(filters), encoding="utf-8")
    command += [
        "-filter_complex_script", str(script), "-map", "0:v:0", "-map", f"[{output_label}]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-t", f"{duration:.6f}",
        "-movflags", "+faststart", str(output),
    ]
    run_ffmpeg(command)
