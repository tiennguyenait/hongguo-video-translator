import json
import subprocess
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import srt

from .adaptive_subtitle import FONT_NAME, FONT_PATH, generate_adaptive_ass

if TYPE_CHECKING:
    from .source_subtitle_mask import SubtitleRegion

VIDEO_CRF = "23"
AUDIO_BITRATE = "128k"


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """Probe the encoder itself; listing it does not guarantee a usable driver."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
            "color=black:s=256x256:d=0.04", "-frames:v", "1",
            "-c:v", "h264_nvenc", "-f", "null", "-",
        ],
        text=True, capture_output=True,
    )
    return result.returncode == 0


def _merge_video_encode_args(inputs: list[Path], duration: float) -> list[str]:
    """Keep delivery size near the inputs while preferring hardware encoding."""
    # Input bytes include audio. Reserve our delivery audio bitrate and a small
    # container allowance instead of allowing quality-based encoding to grow
    # without a ceiling after scaling or adding a watermark.
    aggregate_kbps = sum(path.stat().st_size for path in inputs) * 8 / max(duration, 0.001) / 1000
    # Aim for ~112% of the aggregate input size, leaving enough headroom for
    # scaling mixed-resolution episodes without permitting the old 2x growth.
    # The VBV ceiling corresponds to ~125% of the aggregate delivery bitrate.
    target_kbps = round(max(500, min(8000, aggregate_kbps * 1.12 - 128)))
    maximum_kbps = round(max(target_kbps, min(10000, aggregate_kbps * 1.25 - 128)))
    buffer_kbps = target_kbps * 2
    if nvenc_available():
        return [
            "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
            "-rc", "vbr", "-b:v", f"{target_kbps}k", "-maxrate", f"{maximum_kbps}k",
            "-bufsize", f"{buffer_kbps}k", "-cq", "25", "-spatial-aq", "1",
        ]
    return [
        "-c:v", "libx264", "-preset", "fast", "-b:v", f"{target_kbps}k",
        "-maxrate", f"{maximum_kbps}k", "-bufsize", f"{buffer_kbps}k",
    ]


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


def _stream_signature(path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels", "-of", "json", str(path)],
        text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise RuntimeError(f"Batch episode {path.name} must contain video and audio")
    return (
        video.get("codec_name"), video.get("width"), video.get("height"), video.get("r_frame_rate"),
        audio.get("codec_name"), audio.get("sample_rate"), audio.get("channels"),
    )


def _watermark_filter_lines(base: str, logo_index: int, width: int, height: int, channel_name: str, opacity: float) -> list[str]:
    logo_size = max(32, round(height * 0.055))
    margin_x, margin_y = max(10, round(width * 0.012)), max(10, round(height * 0.018))
    gap, font_size = max(8, round(width * 0.007)), max(17, round(height * 0.025))
    opacity = min(0.85, max(0.15, float(opacity)))
    text = channel_name.replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
    font = str(FONT_PATH.resolve()).replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
    return [
        f"[{logo_index}:v]format=rgba,colorchannelmixer=aa={opacity:.3f},scale={logo_size}:{logo_size}:force_original_aspect_ratio=decrease,pad={logo_size}:{logo_size}:(ow-iw)/2:(oh-ih)/2:color=black@0[wm]",
        f"[{base}][wm]overlay={margin_x}:{margin_y}:format=auto[branded]",
        f"[branded]drawtext=fontfile='{font}':text='{text}':expansion=none:fontcolor=white@{opacity:.3f}:fontsize={font_size}:x={margin_x + logo_size + gap}:y={margin_y}+({logo_size}-text_h)/2:shadowcolor=black@0.30:shadowx=1:shadowy=1[outv]",
    ]


def concat_videos(
    inputs: list[Path], output: Path, logo: Path | None = None,
    channel_name: str = "", opacity: float = 0.58,
) -> None:
    """Concatenate episodes and, when requested, brand in the same encode pass."""
    if not inputs:
        raise ValueError("At least one episode is required")
    signatures = [_stream_signature(path) for path in inputs]
    total_duration = sum(probe_duration(path) for path in inputs)
    branded_duration = total_duration if logo and channel_name else None
    matching = all(signature == signatures[0] for signature in signatures[1:])
    if matching:
        manifest = output.parent / "concat-files.txt"
        lines = ["file '" + str(path.resolve()).replace("'", "'\\''") + "'\n" for path in inputs]
        manifest.write_text("".join(lines), encoding="utf-8")
        if not logo or not channel_name:
            run_ffmpeg([
                "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
                "-c", "copy", "-movflags", "+faststart", str(output),
            ])
            return
        width, height = int(signatures[0][1]), int(signatures[0][2])
        script = output.parent / "concat-filter.ffscript"
        script.write_text(";\n".join(_watermark_filter_lines("0:v", 1, width, height, channel_name, opacity)), encoding="utf-8")
        run_ffmpeg([
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
            "-loop", "1", "-i", str(logo), "-filter_complex_script", str(script),
            "-map", "[outv]", "-map", "0:a:0", *_merge_video_encode_args(inputs, total_duration),
            "-c:a", "copy", "-t", f"{branded_duration:.6f}", "-movflags", "+faststart", str(output),
        ])
        return
    width, height = int(signatures[0][1]), int(signatures[0][2])
    target_fps = Counter(str(signature[3]) for signature in signatures).most_common(1)[0][0]
    command = ["ffmpeg", "-y", "-v", "error"]
    for path in inputs:
        command += ["-i", str(path)]
    filters, labels = [], []
    for index in range(len(inputs)):
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={target_fps},setsar=1,format=yuv420p[v{index}]"
        )
        filters.append(f"[{index}:a]aresample=48000,aformat=channel_layouts=stereo[a{index}]")
        labels.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(labels)}concat=n={len(inputs)}:v=1:a=1[joined][outa]")
    if logo and channel_name:
        command += ["-loop", "1", "-i", str(logo)]
        filters.extend(_watermark_filter_lines("joined", len(inputs), width, height, channel_name, opacity))
    else:
        filters.append("[joined]null[outv]")
    script = output.parent / "concat-filter.ffscript"
    script.write_text(";\n".join(filters), encoding="utf-8")
    command += [
        "-filter_complex_script", str(script), "-map", "[outv]", "-map", "[outa]",
        *_merge_video_encode_args(inputs, total_duration),
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
    ]
    if branded_duration is not None:
        command += ["-t", f"{branded_duration:.6f}"]
    command += ["-movflags", "+faststart", str(output)]
    run_ffmpeg(command)


def apply_channel_watermark(video: Path, output: Path, logo: Path, channel_name: str, opacity: float = 0.58) -> None:
    """Apply a restrained top-left logo + channel label calibrated to the reference style."""
    width, height = probe_video_size(video)
    duration = probe_duration(video)
    logo_size = max(32, round(height * 0.055))
    margin_x = max(10, round(width * 0.012))
    margin_y = max(10, round(height * 0.018))
    gap = max(8, round(width * 0.007))
    font_size = max(17, round(height * 0.025))
    opacity = min(0.85, max(0.15, float(opacity)))
    escaped_text = channel_name.replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
    escaped_font = str(FONT_PATH.resolve()).replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
    script = output.parent / "watermark-filter.ffscript"
    script.write_text(
        f"[1:v]format=rgba,colorchannelmixer=aa={opacity:.3f},scale={logo_size}:{logo_size}:force_original_aspect_ratio=decrease,"
        f"pad={logo_size}:{logo_size}:(ow-iw)/2:(oh-ih)/2:color=black@0[wm];\n"
        f"[0:v][wm]overlay={margin_x}:{margin_y}:format=auto[branded];\n"
        f"[branded]drawtext=fontfile='{escaped_font}':text='{escaped_text}':expansion=none:"
        f"fontcolor=white@{opacity:.3f}:fontsize={font_size}:x={margin_x + logo_size + gap}:"
        f"y={margin_y}+({logo_size}-text_h)/2:shadowcolor=black@0.30:shadowx=1:shadowy=1[outv]",
        encoding="utf-8",
    )
    run_ffmpeg([
        "ffmpeg", "-y", "-v", "error", "-i", str(video), "-loop", "1", "-i", str(logo),
        "-filter_complex_script", str(script), "-map", "[outv]", "-map", "0:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", VIDEO_CRF, "-c:a", "copy",
        "-t", f"{duration:.6f}", "-movflags", "+faststart", str(output),
    ])


def _subtitle_filter(path: Path, force_style: bool = True, fonts_dir: Path | None = None) -> str:
    escaped = str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace(",", "\\,")
    style = (
        f"FontName={FONT_NAME},FontSize=11,PrimaryColour=&H00FFFFFF,"
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
        filters.append(_subtitle_filter(subtitle, fonts_dir=FONT_PATH.parent))
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
