from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_text(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip().lower() if raw and raw.strip() else default


class Settings(BaseModel):
    server_dir: Path = SERVER_DIR
    data_dir: Path = SERVER_DIR / "data"
    jobs_dir: Path = SERVER_DIR / "data" / "jobs"
    database_path: Path = SERVER_DIR / "data" / "jobs.sqlite3"
    # One scene-sized request avoids paying the prompt overhead for every 15
    # display lines while remaining small enough for reliable structured JSON.
    translation_batch_size: int = 30
    job_auto_retries: int = 3
    translation_retries: int = 3
    translation_context_before: int = 6
    translation_context_after: int = 4
    # The primary translation prompt already edits the batch as a coherent
    # scene. Re-reading every line in a second pass doubles token use.
    # Optional expensive editorial passes. They remain opt-in so existing jobs
    # do not unexpectedly double their LLM usage; the safer resegmentation and
    # speaker/timing fixes are enabled independently below.
    translation_scene_review: bool = False
    dialogue_master_ai_repair: bool = False
    ai_prosody_enabled: bool = False
    tts_cache_dir: Path = SERVER_DIR / "data" / "cache" / "tts"
    vieneu_python: Path = Path("/workspace/vieneu-tts/.venv/bin/python")
    vieneu_runner: Path = SERVER_DIR / "scripts" / "vieneu_batch.py"
    narrator_reference: Path = SERVER_DIR / "data" / "voices" / "ngoc-huyen-authorized-reference.wav"
    default_provider: Literal["openai", "gemini", "deepseek"] = "deepseek"
    default_translation_draft_provider: Literal["openai", "gemini", "deepseek"] = "deepseek"
    default_translation_refine_provider: Literal["auto", "openai", "gemini", "deepseek"] = "auto"
    default_asr_model: Literal["tiny", "base", "small", "medium", "large-v3", "qwen3-asr-1.7b"] = "qwen3-asr-1.7b"
    default_diarize: bool = True
    default_min_speakers: int | None = None
    default_max_speakers: int | None = 6
    default_source_language_code: str = "zh"
    default_source_language: str = "Chinese"
    default_target_language: str = "Vietnamese"
    default_burn_subtitles: bool = True
    default_hide_source_subtitles: bool = False
    default_dub: bool = True
    default_tts_voice: str = "Ngọc Huyền — authorized clone"
    tts_sentence_gap_break_seconds: float = 0.8
    tts_long_utterance_warning_ms: int = 14000
    default_original_audio_volume: float = 0.08
    default_watermark_opacity: float = 0.58
    default_channel_name: str = ""
    default_channel_logo: str = ""
    max_upload_bytes: int = 5 * 1024 * 1024 * 1024
    storage_warning_bytes: int = 15 * 1024 * 1024 * 1024
    storage_reserve_bytes: int = 8 * 1024 * 1024 * 1024
    combine_size_multiplier: float = 3.2


@lru_cache
def get_settings() -> Settings:
    settings = Settings(
        default_provider=os.getenv("HONGGUO_DEFAULT_PROVIDER", "deepseek"),
        default_translation_draft_provider=_env_text("HONGGUO_DEFAULT_TRANSLATION_DRAFT_PROVIDER", "deepseek"),
        default_translation_refine_provider=_env_text("HONGGUO_DEFAULT_TRANSLATION_REFINER_PROVIDER", "auto"),
        default_asr_model=os.getenv("HONGGUO_DEFAULT_ASR_MODEL", "qwen3-asr-1.7b"),
        default_diarize=_env_bool("HONGGUO_DEFAULT_DIARIZE", True),
        job_auto_retries=_env_int("HONGGUO_JOB_AUTO_RETRIES", 3),
        default_min_speakers=_env_int("HONGGUO_DEFAULT_MIN_SPEAKERS", None),
        default_max_speakers=_env_int("HONGGUO_DEFAULT_MAX_SPEAKERS", 6),
        default_source_language_code=os.getenv("HONGGUO_DEFAULT_SOURCE_LANGUAGE_CODE", "zh"),
        default_source_language=os.getenv("HONGGUO_DEFAULT_SOURCE_LANGUAGE", "Chinese"),
        default_target_language=os.getenv("HONGGUO_DEFAULT_TARGET_LANGUAGE", "Vietnamese"),
        default_burn_subtitles=_env_bool("HONGGUO_DEFAULT_BURN_SUBTITLES", True),
        default_hide_source_subtitles=_env_bool("HONGGUO_DEFAULT_HIDE_SOURCE_SUBTITLES", False),
        default_dub=_env_bool("HONGGUO_DEFAULT_DUB", True),
        default_tts_voice=os.getenv("HONGGUO_DEFAULT_TTS_VOICE", "Ngọc Huyền — authorized clone"),
        tts_sentence_gap_break_seconds=_env_float("HONGGUO_TTS_SENTENCE_GAP_BREAK_SECONDS", 0.8),
        tts_long_utterance_warning_ms=_env_int("HONGGUO_TTS_LONG_UTTERANCE_WARNING_MS", 14000),
        default_original_audio_volume=_env_float("HONGGUO_DEFAULT_ORIGINAL_AUDIO_VOLUME", 0.08),
        default_watermark_opacity=_env_float("HONGGUO_DEFAULT_WATERMARK_OPACITY", 0.58),
        default_channel_name=os.getenv("HONGGUO_DEFAULT_CHANNEL_NAME", "").strip(),
        default_channel_logo=os.getenv("HONGGUO_DEFAULT_CHANNEL_LOGO", "").strip(),
    )
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.tts_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
