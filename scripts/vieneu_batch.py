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
    # speaker_emb preserves timbre. ref_codes also copy the reference clip's
    # pauses/cadence, which sounds repetitive on unrelated dialogue, so disable it.
    kwargs = {"ref_audio": str(reference), "denoise": True, "use_ref_codes": False} if reference else {"voice": voice}
    # VieNeu accepts one style per batch, so retain batching efficiency while
    # allowing the director to choose a restrained style per utterance.
    for style in ("doc_truyen", "tu_nhien"):
        group = [item for item in items if item.get("style", "doc_truyen") == style]
        if not group:
            continue
        audios = engine.infer_batch([item["text"] for item in group], style=style, **kwargs)
        if len(audios) != len(group):
            raise RuntimeError("VieNeu returned an unexpected batch size")
        for item, audio in zip(group, audios, strict=True):
            engine.save(audio, str(output_dir / f"{int(item['id']):06d}.wav"))


if __name__ == "__main__":
    main()
