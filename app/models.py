from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobStep(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    BURNING = "burning"
    DUBBING = "dubbing"
    DONE = "done"
