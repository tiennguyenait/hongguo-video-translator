"""Detect and mask burned-in source subtitles without recognizing their text."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import srt

TEXT_DETECTOR_MODEL = Path(__file__).resolve().parent / "models" / "text_detection_cn_ppocrv3.onnx"
_TEXT_DETECTOR = None
_TEXT_DETECTOR_CALLS = 0


def _neural_observations(video: Path, timestamps: list[float]) -> list[tuple[tuple[int, int, int, int, float], float]]:
    """Run bounded detector processes; isolates OpenCV DNN native allocations."""
    observations = []
    with tempfile.TemporaryDirectory(prefix="hongguo-text-det-") as temporary:
        root = Path(temporary)
        for offset in range(0, len(timestamps), 48):
            batch = timestamps[offset:offset + 48]
            manifest, result_path = root / f"in-{offset}.json", root / f"out-{offset}.json"
            manifest.write_text(json.dumps(batch), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "scripts.detect_text_batch", str(video), str(manifest), str(result_path)],
                text=True, capture_output=True, timeout=180,
            )
            if result.returncode or not result_path.is_file():
                raise RuntimeError(f"Text detector batch failed: {result.stderr.strip()[-2000:]}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            for item in payload:
                observations.extend((tuple(box), float(item["timestamp"])) for box in item["boxes"])
    return observations


@dataclass(slots=True)
class SubtitleRegion:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    start: float
    end: float

    def to_dict(self) -> dict:
        return asdict(self)


def _neural_candidate_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Detect text geometry only; no text recognition/OCR is performed."""
    global _TEXT_DETECTOR, _TEXT_DETECTOR_CALLS
    if not TEXT_DETECTOR_MODEL.is_file():
        return []
    # OpenCV's DB wrapper accumulates native buffers on long runs in some builds.
    # Recreate the tiny model periodically so 30-minute jobs remain bounded.
    if _TEXT_DETECTOR is None or _TEXT_DETECTOR_CALLS >= 72:
        detector = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(TEXT_DETECTOR_MODEL)))
        detector.setBinaryThreshold(0.20)
        detector.setPolygonThreshold(0.40)
        detector.setUnclipRatio(1.5)
        detector.setMaxCandidates(200)
        detector.setInputSize((1280, 736))
        detector.setInputMean((123.675, 116.28, 103.53))
        detector.setInputScale(1.0 / 255.0 / np.asarray([0.229, 0.224, 0.225]))
        _TEXT_DETECTOR = detector
        _TEXT_DETECTOR_CALLS = 0
    height, width = frame.shape[:2]
    polygons, confidences = _TEXT_DETECTOR.detect(cv2.resize(frame, (1280, 736)))
    _TEXT_DETECTOR_CALLS += 1
    scale_x, scale_y = width / 1280, height / 736
    pieces: list[list[float]] = []
    for polygon, confidence in zip(polygons, confidences):
        if float(confidence) < 0.45:
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        pieces.append([x * scale_x, y * scale_y, (x + w) * scale_x, (y + h) * scale_y, float(confidence)])
    # DBNet can return one polygon per word/character for very small captions.
    # Join pieces on the same baseline before applying line-level geometry rules.
    groups: list[list[list[float]]] = []
    for piece in sorted(pieces, key=lambda item: (item[1] + item[3]) / 2):
        cx, cy = (piece[0] + piece[2]) / 2, (piece[1] + piece[3]) / 2
        target = None
        for group in groups:
            gx1 = min(item[0] for item in group); gx2 = max(item[2] for item in group)
            gy = sum((item[1] + item[3]) / 2 for item in group) / len(group)
            gh = max(item[3] - item[1] for item in group)
            horizontal_gap = max(0, max(gx1 - piece[2], piece[0] - gx2))
            if abs(cy - gy) <= max(12, gh * 0.7, (piece[3] - piece[1]) * 0.7) and horizontal_gap <= width * 0.075:
                target = group
                break
        (target if target is not None else groups.append([]) or groups[-1]).append(piece)
    candidates = []
    for group in groups:
        x1, y1 = min(item[0] for item in group), min(item[1] for item in group)
        x2, y2 = max(item[2] for item in group), max(item[3] for item in group)
        w, h = x2 - x1, y2 - y1
        center_score = max(0.0, 1.0 - abs((x1 + x2) / 2 - width / 2) / (width / 2))
        if w / width >= 0.035 and h / height <= 0.15 and w / max(h, 1) >= 1.15 and center_score >= 0.48:
            candidates.append((round(x1), round(y1), round(w), round(h), max(item[4] for item in group)))
    return sorted(candidates, key=lambda item: item[4], reverse=True)


def _candidate_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    neural = _neural_candidate_boxes(frame)
    if neural:
        return neural
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / width)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape
    # Subtitle glyphs produce dense alternating edges. Horizontal closing joins
    # characters into lines while leaving most faces and scenery fragmented.
    candidates: list[tuple[int, int, int, int, float]] = []
    # Burned drama subtitles are normally bright glyphs with a dark outline.
    # Edge-only candidates are intentionally excluded: fast camera motion creates
    # long false text lines on clothing, furniture, and architecture.
    # Two thresholds cover both solid white drama captions and the thinner,
    # semi-transparent grey captions used by some motion-comic exports.
    top_hat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, sw // 100), max(5, sh // 70))),
    )
    raw_masks = [
        cv2.inRange(gray, 185, 255), cv2.inRange(gray, 145, 255),
        cv2.inRange(top_hat, 28, 255),
    ]
    for raw_mask in raw_masks:
        mask = raw_mask.copy()
        kernel_width = max(11, sw // 36)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3)), iterations=1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            wr, hr = w / sw, h / sh
            center_score = max(0.0, 1.0 - abs((x + w / 2) - sw / 2) / (sw / 2))
            compact_centered = 0.04 <= wr < 0.08 and center_score >= 0.65 and y / sh >= 0.62
            if not ((0.08 <= wr <= 0.96 or compact_centered) and center_score >= 0.48 and 0.010 <= hr <= 0.14 and w / max(h, 1) >= 1.2):
                continue
            raw_roi = raw_mask[y:y + h, x:x + w]
            occupancy = cv2.countNonZero(raw_roi) / max(1, w * h)
            component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(raw_roi)
            glyph_components = sum(
                2 <= int(component_stats[index, cv2.CC_STAT_AREA]) <= w * h * 0.25
                for index in range(1, component_count)
            )
            minimum_glyphs = 2 if center_score >= 0.65 and y / sh >= 0.62 else 3
            if occupancy < 0.025 or glyph_components < minimum_glyphs:
                continue
            density = min(1.0, occupancy / 0.30)
            lower_prior = 0.35 + 0.65 * ((y + h / 2) / sh)
            score = 0.45 * center_score + 0.30 * density + 0.25 * lower_prior
            inv = 1 / scale
            candidates.append((round(x * inv), round(y * inv), round(w * inv), round(h * inv), score))
    return sorted(candidates, key=lambda item: item[4], reverse=True)


def _merge_intervals(cues: list[srt.Subtitle], duration: float) -> list[tuple[float, float]]:
    intervals: list[list[float]] = []
    for cue in cues:
        start = max(0.0, cue.start.total_seconds() - 0.12)
        end = min(duration, cue.end.total_seconds() + 0.18)
        if intervals and start - intervals[-1][1] <= 0.35:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return [(start, end) for start, end in intervals]


def detect_source_subtitle_regions(video: Path, cues: list[srt.Subtitle], output_json: Path) -> list[SubtitleRegion]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("OpenCV cannot open video for source subtitle detection")
    width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if frame_count > 0 else cues[-1].end.total_seconds()
    # Detection has its own visual clock. Whisper timings can lead or trail the
    # burned-in text by seconds, so they must never decide when a mask is shown.
    sample_fps = 4.0 if duration <= 120 else 2.0 if duration <= 900 else 1.0
    sample_count = min(3600, max(1, round(duration * sample_fps)))
    sample_step = duration / sample_count
    timestamps = [min(duration, (index + 0.5) * sample_step) for index in range(sample_count)]
    capture.release()
    if TEXT_DETECTOR_MODEL.is_file():
        observations = _neural_observations(video, timestamps)
    else:
        observations: list[tuple[tuple[int, int, int, int, float], float]] = []
        capture = cv2.VideoCapture(str(video))
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if ok:
                observations.extend((candidate, timestamp) for candidate in _candidate_boxes(frame)[:5] if candidate[4] >= 0.48)
        capture.release()

    active_cue_ids: set[int] = set()
    regions: list[SubtitleRegion] = []
    if observations:
        # Keep recurrent horizontal bands. One-frame objects and vertical titles
        # cannot form a visual track and are discarded below.
        bin_height = max(10, round(height * 0.04))
        bands: dict[int, list[tuple[tuple[int, int, int, int, float], float]]] = {}
        for item in observations:
            box = item[0]
            bands.setdefault(round((box[1] + box[3] / 2) / bin_height), []).append(item)
        support = {key: len({round(t / sample_step) for _, t in items}) for key, items in bands.items()}
        # Subtitles repeatedly return to one horizontal band, while highlights on
        # faces, weapons and scenery scatter across the image. Select that stable
        # band first; adjacent bins allow one/two-line subtitles and mild motion.
        def band_rank(key: int) -> float:
            items = bands[key]
            centered = sum(max(0.0, 1.0 - abs((box[0] + box[2] / 2) - width / 2) / (width / 2)) for box, _ in items) / len(items)
            vertical = sum((box[1] + box[3] / 2) / height for box, _ in items) / len(items)
            return support[key] * (0.72 + 0.18 * centered + 0.10 * vertical)
        dominant_band = max(bands, key=band_rank)
        selected_bands = {key for key in bands if abs(key - dominant_band) <= 2 and support[key] >= 2}
        selected = sorted((item for key in selected_bands for item in bands[key]), key=lambda item: item[1])

        tracks: list[list[tuple[tuple[int, int, int, int, float], float]]] = []
        for observation in selected:
            box, timestamp = observation
            best_track = None
            best_distance = float("inf")
            for track in tracks:
                previous, previous_time = track[-1]
                if timestamp - previous_time > sample_step * 2.2:
                    continue
                cy, previous_cy = box[1] + box[3] / 2, previous[1] + previous[3] / 2
                cx, previous_cx = box[0] + box[2] / 2, previous[0] + previous[2] / 2
                distance = abs(cy - previous_cy)
                horizontal_distance = abs(cx - previous_cx)
                if (distance <= max(height * 0.045, min(box[3], previous[3]) * 0.8)
                        and horizontal_distance <= max(width * 0.12, min(box[2], previous[2]) * 0.45)
                        and distance < best_distance):
                    best_track, best_distance = track, distance
            (best_track if best_track is not None else tracks.append([]) or tracks[-1]).append(observation)

        for track in tracks:
            timestamps = sorted({item[1] for item in track})
            if len(timestamps) < 2:
                continue
            array = np.asarray([[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box, _ in track], dtype=np.float64)
            x1, y1 = np.percentile(array[:, :2], 5, axis=0)
            x2, y2 = np.percentile(array[:, 2:], 95, axis=0)
            detected_w, detected_h = x2 - x1, y2 - y1
            pad_x, pad_y = max(12, detected_w * 0.08), max(8, detected_h * 0.30)
            x1, x2 = max(0, round(x1 - pad_x)), min(width, round(x2 + pad_x))
            y1, y2 = max(0, round(y1 - pad_y)), min(height, round(y2 + pad_y))
            average_score = sum(box[4] for box, _ in track) / len(track)
            temporal_score = min(1.0, len(timestamps) / 4)
            confidence = min(0.99, 0.55 * average_score + 0.45 * temporal_score)
            start, end = max(0.0, timestamps[0] - sample_step * 0.65), min(duration, timestamps[-1] + sample_step * 0.65)
            regions.append(SubtitleRegion(x1, y1, x2 - x1, y2 - y1, round(confidence, 3), start, end))
        # Close very short visual gaps without borrowing Whisper timing.
        regions.sort(key=lambda item: item.start)
        merged: list[SubtitleRegion] = []
        for region in regions:
            if merged and region.start - merged[-1].end <= sample_step * 1.2 and abs((region.y + region.height / 2) - (merged[-1].y + merged[-1].height / 2)) <= height * 0.045:
                previous = merged[-1]
                left, top = min(previous.x, region.x), min(previous.y, region.y)
                right, bottom = max(previous.x + previous.width, region.x + region.width), max(previous.y + previous.height, region.y + region.height)
                merged[-1] = SubtitleRegion(left, top, right-left, bottom-top, min(previous.confidence, region.confidence), previous.start, region.end)
            else:
                merged.append(region)
        regions = merged
        for cue in cues:
            if any(max(cue.start.total_seconds(), region.start) < min(cue.end.total_seconds(), region.end) for region in regions):
                active_cue_ids.add(cue.index)
        method = "visual_tracks_v2"
    else:
        # Safe visual fallback: common subtitle band, deliberately marked low
        # confidence so QA/UI makes the fallback visible.
        method = "lower_band_fallback"
    if not regions:
        x1, x2 = round(width * 0.08), round(width * 0.92)
        y1, y2 = round(height * 0.76), round(height * 0.90)
        regions = [SubtitleRegion(x1, y1, x2-x1, y2-y1, 0.25, start, end) for start, end in _merge_intervals(cues, duration)]
    output_json.write_text(json.dumps({
        "method": method, "video": {"width": width, "height": height},
        "samples": sample_count, "sample_fps": round(1 / sample_step, 3), "detections": len(observations), "active_cues": sorted(active_cue_ids),
        "regions": [region.to_dict() for region in regions],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return regions
