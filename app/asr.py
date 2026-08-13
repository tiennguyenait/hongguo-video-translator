import logging
import shutil
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

from .subtitle import segments_to_subtitles, write_srt

logger = logging.getLogger(__name__)
_models: dict[tuple[str, str, str], WhisperModel] = {}


def cuda_available() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0


def get_model(model_name: str) -> WhisperModel:
    device, compute_type = ("cuda", "float16") if cuda_available() else ("cpu", "int8")
    key = (model_name, device, compute_type)
    if key not in _models:
        logger.info("Loading faster-whisper model=%s device=%s compute_type=%s", *key)
        _models[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
    return _models[key]


def transcribe(video_path: Path, output_path: Path, model_name: str, language: str):
    segments, info = get_model(model_name).transcribe(str(video_path), language=language or None, vad_filter=True)
    subtitles = segments_to_subtitles(segments)
    if not subtitles:
        raise RuntimeError("Whisper returned no speech segments")
    write_srt(output_path, subtitles)
    logger.info("Transcribed %d cues; detected language=%s", len(subtitles), info.language)
    return subtitles
