"""Versioned, atomic checkpoints for resumable media jobs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_VERSION = "2.0"


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ArtifactManifest:
    def __init__(self, job_dir: Path) -> None:
        self.path = job_dir / "artifacts.json"
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {"pipeline_version": PIPELINE_VERSION, "artifacts": {}}
        if self.data.get("pipeline_version") != PIPELINE_VERSION:
            self.data = {"pipeline_version": PIPELINE_VERSION, "artifacts": {}}

    def valid(self, name: str, fingerprint: str, paths: list[Path]) -> bool:
        record = self.data["artifacts"].get(name, {})
        return record.get("fingerprint") == fingerprint and all(path.is_file() and path.stat().st_size > 0 for path in paths)

    def complete(self, name: str, fingerprint: str, paths: list[Path], metadata: dict[str, Any] | None = None) -> None:
        self.data["artifacts"][name] = {
            "fingerprint": fingerprint,
            "files": [path.name for path in paths],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        atomic_write_json(self.path, self.data)

    def invalidate_after(self, names: list[str], current: str) -> None:
        try:
            index = names.index(current)
        except ValueError:
            return
        for name in names[index + 1 :]:
            self.data["artifacts"].pop(name, None)
        atomic_write_json(self.path, self.data)
