"""Cross-language inventory, structural rejection and non-destructive import."""
import json
from pathlib import Path
import struct
import subprocess

import pytest

from tools.embeddinggemma2_contract import tensor_specs, validate_config
from tools.extract_embeddinggemma2 import extract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "tests/fixtures/embeddinggemma2_config.json").read_text())
BINARY = ROOT / "build/gewell_embeddinggemma2_contract_test"


def test_native_inventory_matches_python():
    if not BINARY.exists():
        pytest.skip("build gewell_embeddinggemma2_contract_test")
    actual = json.loads(subprocess.check_output([BINARY, "--dump"]))
    specs = tensor_specs()
    assert len(specs) == 413
    assert sum(s.byte_length for s in specs) == 542005296
    assert actual == [{"id": s.physical_id, "name": s.source_name, "shape": list(s.shape)} for s in specs]


@pytest.mark.parametrize("field,value", [
    ("sliding_window", 1024), ("rms_norm_eps", 1e-5),
    ("hidden_activation", "silu"), ("attention_bias", True),
    ("max_position_embeddings", 8191), ("num_key_value_heads", 1),
    ("per_layer_config", {}), ("rope_parameters", {}),
])
def test_incompatible_semantics_rejected_in_both_languages(tmp_path, field, value):
    config = json.loads(json.dumps(CONFIG))
    config["text_config"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config)
    if BINARY.exists():
        (tmp_path / "config.json").write_text(json.dumps(config))
        result = subprocess.run([BINARY, tmp_path], capture_output=True, text=True)
        assert result.returncode != 0
        assert field in result.stderr


def test_compatible_finetune_metadata_is_not_an_allowlist():
    config = json.loads(json.dumps(CONFIG))
    config.update(_name_or_path="my-finetune", transformers_version="future", revision="local")
    validate_config(config)


def source(tmp_path, dtype="BF16", shape=(262144, 512)):
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(CONFIG))
    (snapshot / "tokenizer.json").write_text("{}")
    name = "language_model.embed_tokens.weight"
    size = shape[0] * shape[1] * 2
    raw = json.dumps({name: {"dtype": dtype, "shape": shape, "data_offsets": [0, size]}}).encode()
    with (snapshot / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + size)
    return snapshot


@pytest.mark.parametrize("dtype,shape,reason", [
    ("F16", (262144, 512), "dtype"),
    ("BF16", (262144, 256), "shape"),
    ("BF16", (262144, 512), "missing source tensor"),
])
def test_bad_source_preserves_existing_output(tmp_path, dtype, shape, reason):
    snapshot = source(tmp_path, dtype, shape)
    output = tmp_path / "bundle"
    output.mkdir()
    sentinel = output / "successful-result"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match=reason):
        extract(snapshot, output)
    assert sentinel.read_text() == "keep"
    assert list(output.iterdir()) == [sentinel]


def test_failed_validation_does_not_publish_bundle(tmp_path):
    snapshot = source(tmp_path)
    with pytest.raises(ValueError, match="missing source tensor"):
        extract(snapshot, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
