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
    translation_batch_size: int = 15
    translation_retries: int = 3
    translation_context_before: int = 6
    translation_context_after: int = 4
    translation_scene_review: bool = True
    tts_cache_dir: Path = SERVER_DIR / "data" / "cache" / "tts"
    vieneu_python: Path = Path("/workspace/vieneu-tts/.venv/bin/python")
    vieneu_runner: Path = SERVER_DIR / "scripts" / "vieneu_batch.py"
    narrator_reference: Path = SERVER_DIR / "data" / "voices" / "ngoc-huyen-authorized-reference.wav"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.tts_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
