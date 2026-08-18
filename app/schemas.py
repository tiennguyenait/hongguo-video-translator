from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


class JobCreate(BaseModel):
    url: str
    provider: Literal["openai", "gemini", "deepseek"] = "openai"
    # large-v3 is materially more reliable for Chinese drama dialogue than the
    # small Qwen local default; Qwen remains available as an explicit option.
    asr_model: Literal["tiny", "base", "small", "medium", "large-v3", "qwen3-asr-1.7b"] = "large-v3"
    diarize: bool = True
    min_speakers: int | None = Field(default=None, ge=1, le=20)
    max_speakers: int | None = Field(default=6, ge=1, le=20)
    source_language_code: str = Field(default="zh", min_length=2, max_length=12)
    source_language: str = Field(default="Chinese", min_length=2, max_length=50)
    target_language: str = Field(default="Vietnamese", min_length=2, max_length=50)
    glossary: str | None = Field(default=None, max_length=10_000)
    burn_subtitles: bool = True
    hide_source_subtitles: bool = False
    dub: bool = True
    tts_voice: str = Field(default="Ngọc Huyền — authorized clone", max_length=100)
    original_audio_volume: float = Field(default=0.08, ge=0.0, le=1.0)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("URL must begin with http:// or https:// and include a host")
        return value.strip()


class JobOutputs(BaseModel):
    video: str | None = None
    source_srt: str | None = None
    translated_srt: str | None = None
    burned_video: str | None = None
    dubbed_video: str | None = None
    speaker_report: str | None = None
    qa_report: str | None = None
    subtitle_regions: str | None = None
    subtitle_layout: str | None = None


class JobResponse(BaseModel):
    id: str
    url: str
    status: str
    step: str
    progress_message: str
    error: str | None
    provider: str
    asr_model: str
    received_bytes: int | None = None
    sha256: str | None = None
    diarize: bool
    burn_subtitles: bool
    hide_source_subtitles: bool
    dub: bool
    created_at: datetime
    updated_at: datetime
    outputs: JobOutputs


class SubtitleEdit(BaseModel):
    id: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=1000)


class JobReview(BaseModel):
    translations: list[SubtitleEdit] = Field(min_length=1, max_length=5000)


class BatchEpisode(BaseModel):
    batch_id: str
    position: int
    filename: str
    received_bytes: int | None = None
    sha256: str | None = None
    job_id: str
    job: JobResponse


class BatchResponse(BaseModel):
    id: str
    status: str
    progress_message: str
    error: str | None
    created_at: datetime
    updated_at: datetime
    output: str | None
    download_confirmed_at: datetime | None = None
    download_confirmed_bytes: int | None = None
    eligible_for_cleanup: bool = False
    episodes: list[BatchEpisode]


class BatchDownloadAck(BaseModel):
    size_bytes: int = Field(gt=0)
