"""Detect and mask burned-in source subtitles without recognizing their text."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import srt


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


def _candidate_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
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
    raw_masks = [cv2.inRange(gray, 185, 255)]
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
            if not ((0.08 <= wr <= 0.96 or compact_centered) and 0.010 <= hr <= 0.14 and w / max(h, 1) >= 1.2):
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
    observations: list[tuple[tuple[int, int, int, int, float], float, int]] = []
    # A few points within every spoken cue are enough; source subtitles normally
    # remain static during that interval. Cap samples to keep 30-minute jobs fast.
    sample_points: list[tuple[float, int]] = []
    for cue in cues:
        start, end = cue.start.total_seconds(), cue.end.total_seconds()
        sample_points.extend([(start + (end - start) * ratio, cue.index) for ratio in (0.30, 0.62)])
    if len(sample_points) > 360:
        step = len(sample_points) / 360
        sample_points = [sample_points[int(index * step)] for index in range(360)]
    for timestamp, cue_id in sample_points:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        candidates = _candidate_boxes(frame)
        for candidate in candidates[:5]:
            if candidate[4] >= 0.48:
                observations.append((candidate, timestamp, cue_id))
    capture.release()

    active_cue_ids: set[int] = set()
    if observations:
        # Cluster vertical positions across time. Dialogue subtitles recur in the
        # same band while titles/signs are brief or move with the scene.
        clusters: dict[int, list[tuple[tuple[int, int, int, int, float], float, int]]] = {}
        bin_height = max(12, round(height * 0.055))
        for observation in observations:
            candidate = observation[0]
            center_y = candidate[1] + candidate[3] / 2
            clusters.setdefault(round(center_y / bin_height), []).append(observation)
        def cluster_rank(items):
            sample_support = len({round(timestamp, 2) for _, timestamp, _ in items})
            average_score = sum(item[0][4] for item in items) / len(items)
            lower_prior = sum((item[0][1] + item[0][3] / 2) / height for item in items) / len(items)
            return sample_support + average_score + 0.35 * lower_prior
        selected = max(clusters.values(), key=cluster_rank)
        # Robust median ignores occasional signs/UI detections. Expand enough to
        # cover outline/shadow, then use one stable band to avoid mask flicker.
        array = np.asarray([item[0][:4] for item in selected], dtype=np.float64)
        x, y, w, h = np.median(array, axis=0)
        pad_x, pad_y = max(10, w * 0.07), max(8, h * 0.45)
        # Use detected vertical placement but a generous centered horizontal band;
        # dialogue length changes from frame to frame and must never leak at edges.
        x1 = max(0, min(round(x - pad_x), round(width * 0.14)))
        x2 = min(width, max(round(x + w + pad_x), round(width * 0.86)))
        y1, y2 = max(0, round(y - pad_y)), min(height, round(y + h + pad_y))
        active_cue_ids = {item[2] for item in selected}
        confidence = min(0.99, len({round(item[1], 2) for item in selected}) / max(2, len(sample_points) * 0.45))
        method = "temporal_text_detection"
    else:
        # Safe visual fallback: common subtitle band, deliberately marked low
        # confidence so QA/UI makes the fallback visible.
        x1, x2 = round(width * 0.08), round(width * 0.92)
        y1, y2 = round(height * 0.76), round(height * 0.88)
        confidence, method = 0.25, "lower_band_fallback"

    active_cues = [cue for cue in cues if cue.index in active_cue_ids] if active_cue_ids else cues
    regions = [SubtitleRegion(x1, y1, x2 - x1, y2 - y1, round(confidence, 3), start, end) for start, end in _merge_intervals(active_cues, duration)]
    output_json.write_text(json.dumps({
        "method": method, "video": {"width": width, "height": height},
        "samples": len(sample_points), "detections": len(observations), "active_cues": sorted(active_cue_ids),
        "regions": [region.to_dict() for region in regions],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return regions
