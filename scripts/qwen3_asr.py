#!/usr/bin/env python3
"""Run Qwen3-ASR with forced alignment and emit WhisperX-compatible JSON."""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("output")
    parser.add_argument("--language", default="Chinese")
    args = parser.parse_args()

    started = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=4096,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
    )
    audio_path = Path(args.audio)
    temporary_audio: tempfile.TemporaryDirectory[str] | None = None
    try:
        if audio_path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
            temporary_audio = tempfile.TemporaryDirectory(prefix="qwen3-asr-")
            extracted = Path(temporary_audio.name) / "audio.wav"
            conversion = subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-i", str(audio_path),
                    "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(extracted),
                ],
                text=True, capture_output=True,
            )
            if conversion.returncode:
                raise RuntimeError(f"FFmpeg audio extraction failed: {conversion.stderr.strip()}")
            audio_path = extracted
        result = model.transcribe(
            audio=str(audio_path), language=args.language, return_time_stamps=True,
        )[0]
    finally:
        if temporary_audio is not None:
            temporary_audio.cleanup()
    words = [
        {
            "word": item.text,
            "start": float(item.start_time),
            "end": float(item.end_time),
            "score": 1.0,
        }
        for item in result.time_stamps.items
        if item.text.strip() and float(item.end_time) > float(item.start_time)
    ]
    payload = {
        "language": result.language,
        "text": result.text,
        "word_segments": words,
        "segments": [{
            "start": words[0]["start"] if words else 0.0,
            "end": words[-1]["end"] if words else 0.0,
            "text": result.text,
            "words": words,
        }] if words else [],
        "engine": "Qwen3-ASR-1.7B+ForcedAligner-0.6B",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
