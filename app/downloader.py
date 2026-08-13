from pathlib import Path

from yt_dlp import YoutubeDL


def download_video(url: str, job_dir: Path, progress_hook=None) -> Path:
    output = job_dir / "source.mp4"
    if output.is_file() and output.stat().st_size > 0:
        return output
    options = {
        "format": "bv*+ba/b",
        "outtmpl": str(job_dir / "source.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "continuedl": True,
        "windowsfilenames": True,
        "progress_hooks": [progress_hook] if progress_hook else [],
        "quiet": True,
        "no_warnings": False,
    }
    with YoutubeDL(options) as ydl:
        ydl.download([url])
    if not output.is_file():
        candidates = list(job_dir.glob("source.*"))
        raise RuntimeError(f"yt-dlp did not produce source.mp4 (created: {[p.name for p in candidates]})")
    return output
