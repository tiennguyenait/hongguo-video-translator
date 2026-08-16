"""Isolated VieNeu-TTS batch renderer invoked by the FastAPI worker."""

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


PUNCTUATION_PAUSE_MS = {"minor": 180, "medium": 240, "sentence": 360, "question": 320}


def split_for_natural_pauses(text: str) -> list[tuple[str, int]]:
    """Split at spoken punctuation and retain an explicit post-clause pause."""
    parts = re.split(r"(?<=[,;:!?])\s+|(?<=\.)\s+|(?<=…)\s+", text.strip())
    result: list[tuple[str, int]] = []
    for part in (item.strip() for item in parts):
        if not part:
            continue
        if re.search(r"(?:\.{2,}|…+)$", part):
            pause = PUNCTUATION_PAUSE_MS["sentence"]
        elif re.search(r"[!?]+$", part):
            pause = PUNCTUATION_PAUSE_MS["question"]
        elif re.search(r"\.+$", part):
            pause = PUNCTUATION_PAUSE_MS["sentence"]
        elif re.search(r"[:;]+$", part):
            pause = PUNCTUATION_PAUSE_MS["medium"]
        elif part.endswith(","):
            pause = PUNCTUATION_PAUSE_MS["minor"]
        else:
            pause = 0
        result.append((part, pause))
    return result


def infer_with_natural_pauses(engine: Any, texts: list[str], style: str, **kwargs) -> list[np.ndarray]:
    plans = [split_for_natural_pauses(text) for text in texts]
    flat_parts = [part for plan in plans for part, _ in plan]
    flat_audio = iter(engine.infer_batch(flat_parts, style=style, **kwargs))
    results: list[np.ndarray] = []
    for plan in plans:
        chunks: list[np.ndarray] = []
        for index, (_, pause_ms) in enumerate(plan):
            chunks.append(np.asarray(next(flat_audio), dtype=np.float32))
            if index < len(plan) - 1 and pause_ms:
                chunks.append(np.zeros(round(engine.sample_rate * pause_ms / 1000), dtype=np.float32))
        results.append(np.concatenate(chunks) if chunks else np.array([], dtype=np.float32))
    return results


def main() -> None:
    from vieneu import Vieneu

    manifest_path, output_dir, voice = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    reference = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not items:
        raise ValueError("Empty VieNeu input")
    # Auto backend: CUDA batches all utterances and enables reference cloning;
    # ONNX remains available for preset voices when no reference is configured.
    engine = Vieneu()
    # speaker_emb preserves timbre. ref_codes also copy the reference clip's
    # pauses/cadence, which sounds repetitive on unrelated dialogue, so disable it.
    kwargs = {"ref_audio": str(reference), "denoise": True, "use_ref_codes": False} if reference else {"voice": voice}
    # VieNeu accepts one style per batch, so retain batching efficiency while
    # allowing the director to choose a restrained style per utterance.
    for style in ("doc_truyen", "tu_nhien"):
        group = [item for item in items if item.get("style", "doc_truyen") == style]
        if not group:
            continue
        audios = infer_with_natural_pauses(engine, [item["text"] for item in group], style, **kwargs)
        if len(audios) != len(group):
            raise RuntimeError("VieNeu returned an unexpected batch size")
        for item, audio in zip(group, audios, strict=True):
            engine.save(audio, str(output_dir / f"{int(item['id']):06d}.wav"))


if __name__ == "__main__":
    main()
