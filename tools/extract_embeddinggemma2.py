#!/usr/bin/env python3
"""Extract BF16 EmbeddingGemma 2 text and optional vision/audio into a new bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.bf16_artifact import load_json_object
from tools.component_weights import write_component
from tools.embeddinggemma2_contract import tensor_specs, validate_config
from tools import embeddinggemma2_vision_contract as vision_contract
from tools import embeddinggemma2_audio_contract as audio_contract
from tools.safetensors_source import SourceFiles, identity


def extract(snapshot: Path, output: Path, vision: bool = False, audio: bool = False):
    assets = [snapshot / name for name in ("config.json", "tokenizer.json")]
    if vision or audio:
        assets.append(snapshot / "processor_config.json")
    identities = {path: identity(path) for path in assets}
    validate_config(load_json_object(snapshot / "config.json"))
    source = SourceFiles(snapshot)
    components = {"text":tensor_specs()}
    if vision:
        vision_contract.validate_config(load_json_object(snapshot / "config.json"))
        vision_contract.validate_processor(load_json_object(snapshot / "processor_config.json"))
        components["vision"] = vision_contract.tensor_specs()
    if audio:
        audio_contract.validate_config(load_json_object(snapshot / "config.json"))
        audio_contract.validate_processor(load_json_object(snapshot / "processor_config.json"))
        components["audio"] = audio_contract.tensor_specs()
    tensors = {name:[source.tensor(spec.source_name, "BF16", spec.shape) for spec in specs]
               for name,specs in components.items()}
    # New bundles only: a failed attempt cannot remove or replace a successful bundle.
    output.mkdir(parents=True, exist_ok=False)
    try:
        for path in assets:
            shutil.copyfile(path, output / path.name)
        for name,specs in components.items():
            write_component(output / (name+".safetensors"), specs, tensors[name],
                            metadata={"format": "pt", "architecture": "embedding_gemma2", "component": name})
        for path, before in (source.identities | identities).items():
            if identity(path) != before:
                raise ValueError(f"source changed during extraction: {path}")
    except BaseException:
        shutil.rmtree(output)
        raise
    imported = [spec for specs in components.values() for spec in specs]
    return {"tensors": len(imported), "payload_bytes": sum(s.byte_length for s in imported)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--audio", action="store_true")
    args = parser.parse_args()
    print(extract(args.snapshot, args.output, args.vision, args.audio))


if __name__ == "__main__":
    main()
