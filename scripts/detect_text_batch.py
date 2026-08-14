"""Bounded subprocess for local PPOCR text geometry detection (no recognition)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from app.source_subtitle_mask import _neural_candidate_boxes


def main() -> None:
    video, manifest, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    timestamps = json.loads(manifest.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("Cannot open video for text detection")
    result = []
    for timestamp in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
        ok, frame = capture.read()
        result.append({
            "timestamp": timestamp,
            "boxes": _neural_candidate_boxes(frame)[:8] if ok else [],
        })
    capture.release()
    output.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
