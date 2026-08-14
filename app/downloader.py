from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


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
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            "Referer": "https://novelquickapp.com/",
        },
    }
    try:
        with YoutubeDL(options) as ydl:
            ydl.download([url])
    except DownloadError as exc:
        detail = str(exc)
        if "403" in detail:
            raise RuntimeError("Video host returned HTTP 403. The signed Hongguo URL may have expired; obtain a fresh video URL and retry.") from exc
        raise RuntimeError(f"Video download failed: {detail}") from exc
    if not output.is_file():
        candidates = list(job_dir.glob("source.*"))
        raise RuntimeError(f"yt-dlp did not produce source.mp4 (created: {[p.name for p in candidates]})")
    return output
