"""Isolated VieNeu-TTS batch renderer invoked by the FastAPI worker."""

import json
import sys
from pathlib import Path

from vieneu import Vieneu


def main() -> None:
    manifest_path, output_dir, voice = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    reference = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not items:
        raise ValueError("Empty VieNeu input")
    # Auto backend: CUDA batches all utterances and enables reference cloning;
    # ONNX remains available for preset voices when no reference is configured.
    engine = Vieneu()
    texts = [item["text"] for item in items]
    # speaker_emb preserves timbre. ref_codes also copy the reference clip's
    # pauses/cadence, which sounds repetitive on unrelated dialogue, so disable it.
    kwargs = {"ref_audio": str(reference), "denoise": True, "use_ref_codes": False} if reference else {"voice": voice}
    audios = engine.infer_batch(texts, style="doc_truyen", **kwargs)
    if len(audios) != len(items):
        raise RuntimeError("VieNeu returned an unexpected batch size")
    for item, audio in zip(items, audios, strict=True):
        engine.save(audio, str(output_dir / f"{int(item['id']):06d}.wav"))


if __name__ == "__main__":
    main()
