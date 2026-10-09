"""Cross-language image contract and import failure boundaries."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from tools import embeddinggemma2_vision_contract as vision
from tools.extract_embeddinggemma2 import extract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "tests/fixtures/embeddinggemma2_multimodal_config.json").read_text())
PROCESSOR = json.loads((ROOT / "tests/fixtures/embeddinggemma2_processor_config.json").read_text())
BINARY = ROOT / "build/gewell_embeddinggemma2_contract_test"


def test_inventory_matches_native():
    specs = vision.tensor_specs()
    assert len(specs) == 211
    assert sum(s.byte_length for s in specs) == 335515648
    if not BINARY.exists():
        pytest.skip("build gewell_embeddinggemma2_contract_test")
    rows = json.loads(subprocess.check_output([BINARY, "--vision", "--dump"]))
    assert rows == [{"id": s.physical_id, "name": s.source_name, "shape": list(s.shape)} for s in specs]


@pytest.mark.parametrize("section,field,value", [
    ("vision_config", "hidden_size", 1152), ("vision_config", "standardize", True),
    ("vision_config", "num_hidden_layers", 27), ("vision_config", "head_dim", 72),
    ("vision_config", "pooling_kernel_size", 2), ("vision_config", "attention_bias", True),
    ("vision_config", "use_clipped_linears", True), ("vision_config", "rope_parameters", {}),
    ("image_processor", "do_normalize", True), ("image_processor", "resample", 2),
    ("image_processor", "rescale_factor", 1), ("image_processor", "max_soft_tokens", 560),
    ("image_processor", "patch_size", 32), ("image_processor", "do_convert_rgb", False),
    ("video_processor", "fps", 2), ("video_processor", "max_frames", 64),
    ("video_processor", "max_soft_tokens", 70), ("video_processor", "overflow_strategy", "truncate"),
    ("video_processor", "add_timestamps", True), ("video_processor", "do_sample_frames", False),
    ("video_processor", "do_normalize", True), ("video_processor", "resample", 2),
])
def test_bad_semantics_rejected(tmp_path, section, field, value):
    config, processor = copy.deepcopy(CONFIG), copy.deepcopy(PROCESSOR)
    target = config if section == "vision_config" else processor
    target[section][field] = value
    with pytest.raises(ValueError, match=field):
        (vision.validate_config if section == "vision_config" else vision.validate_processor)(target)
    if BINARY.exists():
        (tmp_path / "config.json").write_text(json.dumps(config))
        (tmp_path / "processor_config.json").write_text(json.dumps(processor))
        result = subprocess.run([BINARY, "--vision", tmp_path], capture_output=True, text=True)
        assert result.returncode != 0 and field in result.stderr


def test_compatible_metadata_and_unloaded_audio():
    config = copy.deepcopy(CONFIG)
    config.update(_name_or_path="my-finetune", audio_config=None)
    vision.validate_config(config)
    vision.validate_processor(PROCESSOR)


def test_missing_processor_preserves_existing_bundle(tmp_path):
    snapshot, output = tmp_path / "source", tmp_path / "bundle"
    snapshot.mkdir()
    output.mkdir()
    (snapshot / "config.json").write_text(json.dumps(CONFIG))
    (snapshot / "tokenizer.json").write_text("{}")
    sentinel = output / "saved-result"
    sentinel.write_text("preserve")
    with pytest.raises(FileNotFoundError):
        extract(snapshot, output, vision=True)
    assert sentinel.read_text() == "preserve"
    assert list(output.iterdir()) == [sentinel]
