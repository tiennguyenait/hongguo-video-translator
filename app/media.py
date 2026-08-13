import subprocess
from pathlib import Path


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


def _subtitle_filter(path: Path) -> str:
    escaped = str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''").replace(",", "\\,")
    style = (
        "FontName=Arial,FontSize=11,PrimaryColour=&H00FFFFFF,"
        "BackColour=&H70000000,OutlineColour=&H00000000,"
        "BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginV=28"
    )
    return f"subtitles=filename='{escaped}':force_style='{style}'"


def burn_subtitles(video: Path, subtitle: Path, output: Path) -> None:
    run_ffmpeg(["ffmpeg", "-y", "-i", str(video), "-vf", _subtitle_filter(subtitle), "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "copy", "-movflags", "+faststart", str(output)])


def mux_dub(video: Path, wav: Path, output: Path, original_audio_volume: float) -> None:
    if original_audio_volume > 0:
        filters = f"[0:a]volume={original_audio_volume}[original];[1:a]volume=1.0[dub];[original][dub]amix=inputs=2:duration=longest:normalize=0[a]"
        command = ["ffmpeg", "-y", "-i", str(video), "-i", str(wav), "-filter_complex", filters, "-map", "0:v:0", "-map", "[a]"]
    else:
        command = ["ffmpeg", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0"]
    command += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output)]
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
        "aresample=48000:async=1:first_pts=0[dub]"
    )
    output_label = "dub"
    if original_audio_volume > 0:
        filters.append(
            f"[0:a]aresample=48000:async=1:first_pts=0,volume={original_audio_volume}[original]"
        )
        filters.append(
            f"[original][dub]amix=inputs=2:duration=longest:normalize=0,"
            f"atrim=0:{duration:.6f}[mixed]"
        )
        output_label = "mixed"
    script = output.parent / "dub-filter.ffscript"
    script.write_text(";\n".join(filters), encoding="utf-8")
    command += [
        "-filter_complex_script", str(script), "-map", "0:v:0", "-map", f"[{output_label}]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.6f}",
        "-movflags", "+faststart", str(output),
    ]
    run_ffmpeg(command)
