from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class JobStep(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    DETECTING_SUBTITLES = "detecting_subtitles"
    BURNING = "burning"
    DUBBING = "dubbing"
    QA = "qa"
    DONE = "done"
