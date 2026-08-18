from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_DIR / ".env")


class Settings(BaseModel):
    server_dir: Path = SERVER_DIR
    data_dir: Path = SERVER_DIR / "data"
    jobs_dir: Path = SERVER_DIR / "data" / "jobs"
    database_path: Path = SERVER_DIR / "data" / "jobs.sqlite3"
    # One scene-sized request avoids paying the prompt overhead for every 15
    # display lines while remaining small enough for reliable structured JSON.
    translation_batch_size: int = 30
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
    max_upload_bytes: int = 5 * 1024 * 1024 * 1024
    storage_warning_bytes: int = 15 * 1024 * 1024 * 1024
    storage_reserve_bytes: int = 8 * 1024 * 1024 * 1024
    combine_size_multiplier: float = 3.2


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.tts_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
