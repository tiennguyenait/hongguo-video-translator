"""OCR burned-in Chinese subtitles into source SRT cues."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import srt
from rapidocr import RapidOCR

from .subtitle import seconds_to_timedelta, write_srt

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
LOW_SCORE_RECOGNITION_THRESHOLD = 0.88
REFINEMENT_SCORE_GAIN = 0.06
REFINEMENT_TIMESTAMP_SAMPLES = 7
REFINEMENT_VARIANTS = (0.45, 0.39, 0.33)


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text.strip())
    text = text.replace("，", "").replace("。", "").replace("！", "").replace("？", "")
    text = text.replace(",", "").replace(".", "").replace("!", "").replace("?", "")
    return text


@dataclass(slots=True)
class OCRCue:
    start: float
    end: float
    text: str
    score: float
    samples: int

    def to_dict(self) -> dict:
        return asdict(self)


def _best_line(output, crop_width: int) -> tuple[str, float] | None:
    txts = getattr(output, "txts", None)
    boxes = getattr(output, "boxes", None)
    scores = getattr(output, "scores", None)
    txts = list(txts) if txts is not None else []
    boxes = list(boxes) if boxes is not None else []
    scores = list(scores) if scores is not None else []
    candidates = []
    for text, box, score in zip(txts, boxes, scores, strict=False):
        normalized = _normalize_text(str(text))
        if not normalized or not _CJK_RE.search(normalized):
            continue
        xs = [float(point[0]) for point in box]
        center = (min(xs) + max(xs)) / 2
        center_score = max(0.0, 1.0 - abs(center - crop_width / 2) / (crop_width / 2))
        if center_score < 0.45:
            continue
        candidates.append((float(score) + center_score * 0.12 + min(len(normalized), 18) * 0.004, normalized, float(score)))
    if not candidates:
        return None
    _, text, score = max(candidates, key=lambda item: item[0])
    return text, score


def _subtitle_variants(frame):
    h, w = frame.shape[:2]
    variants: list[tuple[int, object]] = []
    crop_starts = (0.40, 0.44, 0.48, 0.52)
    for crop_start in crop_starts:
        crop = frame[int(h * crop_start):, :]
        if crop.size == 0:
            continue
        variants.append((crop.shape[1], crop))
        resized = cv2.resize(crop, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
        variants.append((resized.shape[1], resized))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        variants.append((gray.shape[1], cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        )
        # Small local-contrast boost for low-contrast subtitles.
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))
        l = clahe.apply(l)
        boosted = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        variants.append((boosted.shape[1], boosted))
    deduped: list[tuple[int, object]] = []
    seen = set[tuple[int, int]]()
    for width, image in variants:
        key = (width, int(image.mean()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((width, image))
    return deduped


def _scan_timestamp(
    engine: RapidOCR,
    capture: cv2.VideoCapture,
    timestamp_s: float,
    threshold: float,
) -> tuple[str, float] | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000)
    ok, frame = capture.read()
    if not ok:
        return None
    for text_score in (threshold, 0.4, 0.3):
        for width, variant in _subtitle_variants(frame):
            result = engine(variant, text_score=text_score)
            candidate = _best_line(result, width)
            if candidate:
                return candidate
    return None


def _refine_low_confidence_cues(
    video: Path,
    cues: list[OCRCue],
    engine: RapidOCR,
    threshold: float = LOW_SCORE_RECOGNITION_THRESHOLD,
) -> int:
    low = [cue for cue in cues if cue.score < threshold]
    if not low:
        return 0

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return 0
    refined_count = 0
    try:
        for cue in low:
            best: tuple[str, float] | None = None
            span = max(0.4, cue.end - cue.start)
            base = (cue.start + cue.end) / 2
            for offset in REFINEMENT_VARIANTS:
                start_offset = base - span * offset
                step = span / max(1, REFINEMENT_TIMESTAMP_SAMPLES - 1)
                for sample in range(REFINEMENT_TIMESTAMP_SAMPLES):
                    timestamp = start_offset + sample * step
                    candidate = _scan_timestamp(engine, capture, timestamp, threshold)
                    if not candidate:
                        continue
                    text, score = candidate
                    if not text:
                        continue
                    if best is None or score > best[1]:
                        best = (text, score)
            if best and best[1] >= cue.score + REFINEMENT_SCORE_GAIN:
                cue.text = best[0]
                cue.score = best[1]
                cue.samples = max(cue.samples, 3)
                refined_count += 1
    finally:
        capture.release()
    return refined_count


def _append_observation(cues: list[OCRCue], timestamp: float, text: str, score: float, sample_step: float) -> None:
    start = max(0.0, timestamp - sample_step * 0.75)
    end = timestamp + sample_step * 0.75
    if cues and cues[-1].text == text and start - cues[-1].end <= sample_step * 2.25:
        previous = cues[-1]
        total = previous.samples + 1
        previous.end = max(previous.end, end)
        previous.score = (previous.score * previous.samples + score) / total
        previous.samples = total
        return
    cues.append(OCRCue(start=start, end=end, text=text, score=score, samples=1))


def _normalize_timing(cues: list[OCRCue]) -> list[OCRCue]:
    result: list[OCRCue] = []
    for cue in sorted(cues, key=lambda item: (item.start, item.end)):
        start, end = cue.start, max(cue.end, cue.start + 0.35)
        if result and start < result[-1].end + 0.02:
            start = result[-1].end + 0.02
            end = max(end, start + 0.35)
        result.append(OCRCue(start=start, end=end, text=cue.text, score=cue.score, samples=cue.samples))
    return result


def extract_burned_subtitles(
    video: Path,
    output_srt: Path,
    report_json: Path,
    *,
    sample_fps: float = 2.5,
    min_cues: int = 8,
) -> tuple[list[srt.Subtitle], dict] | None:
    """Return OCR subtitles when the video has a strong burned-in Chinese track."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("OpenCV cannot open video for OCR")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if frame_count > 0 else 0.0
    if duration <= 0:
        capture.release()
        raise RuntimeError("Cannot measure video duration for OCR")

    engine = RapidOCR()
    sample_step = 1.0 / sample_fps
    timestamps = [min(duration, (index + 0.5) * sample_step) for index in range(max(1, round(duration * sample_fps)))]
    observed: list[OCRCue] = []
    frames_read = 0
    for timestamp in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        frames_read += 1
        height, width = frame.shape[:2]
        crop_y = round(height * 0.45)
        crop = frame[crop_y:height, :]
        result = engine(crop, text_score=0.45)
        best = _best_line(result, width)
        if best:
            _append_observation(observed, timestamp, best[0], best[1], sample_step)
    capture.release()

    cues = [
        cue for cue in observed
        if cue.samples >= 2 or cue.score >= 0.94 and len(cue.text) <= 2
    ]
    low_before = len([cue for cue in cues if cue.score < LOW_SCORE_RECOGNITION_THRESHOLD])
    refined_count = _refine_low_confidence_cues(video, cues, engine)
    low_after = len([cue for cue in cues if cue.score < LOW_SCORE_RECOGNITION_THRESHOLD])
    cues = _normalize_timing(cues)
    subtitles = [
        srt.Subtitle(index=index, start=seconds_to_timedelta(cue.start), end=seconds_to_timedelta(cue.end), content=cue.text)
        for index, cue in enumerate(cues, 1)
    ]
    report = {
        "engine": "RapidOCR",
        "sample_fps": sample_fps,
        "frames_read": frames_read,
        "duration": duration,
        "raw_cues": len(observed),
        "cues": len(subtitles),
        "average_score": round(sum(cue.score for cue in cues) / len(cues), 4) if cues else 0.0,
        "low_confidence_threshold": LOW_SCORE_RECOGNITION_THRESHOLD,
        "low_confidence_before": low_before,
        "low_confidence_after": low_after,
        "refined_count": refined_count,
        "items": [cue.to_dict() for cue in cues],
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(subtitles) < min_cues:
        return None
    write_srt(output_srt, subtitles)
    return subtitles, report
