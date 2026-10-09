"""Audio inventory, scalar preservation and source/processor rejection gates."""
import copy
import json
from pathlib import Path
import struct
import subprocess

import pytest

from tools import embeddinggemma2_audio_contract as audio
from tools.embeddinggemma2_contract import tensor_specs as text_specs
from tools.extract_embeddinggemma2 import extract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/"tests/fixtures/embeddinggemma2_multimodal_config.json").read_text())
PROCESSOR = json.loads((ROOT/"tests/fixtures/embeddinggemma2_processor_config.json").read_text())
BINARY = ROOT/"build/gewell_embeddinggemma2_contract_test"


def test_inventory_matches_native_including_scalar_clipping_bounds():
    specs = audio.tensor_specs()
    assert len(specs) == 752
    assert sum(s.byte_length for s in specs) == 611223040
    assert sum(not s.shape for s in specs) == 480
    if not BINARY.exists():
        pytest.skip("build gewell_embeddinggemma2_contract_test")
    actual = json.loads(subprocess.check_output([BINARY,"--audio","--dump"]))
    assert actual == [{"id":s.physical_id,"name":s.source_name,"shape":list(s.shape)} for s in specs]


@pytest.mark.parametrize("section,field,value", [
    ("audio_config","hidden_size",768), ("audio_config","num_hidden_layers",16),
    ("audio_config","num_attention_heads",4), ("audio_config","output_proj_dims",1024),
    ("audio_config","subsampling_conv_channels",[64,32]), ("audio_config","conv_kernel_size",3),
    ("audio_config","attention_chunk_size",24), ("audio_config","attention_context_left",12),
    ("audio_config","attention_context_right",1), ("audio_config","attention_logit_cap",30),
    ("audio_config","attention_invalid_logits_value",-100), ("audio_config","gradient_clipping",1),
    ("audio_config","residual_weight",1), ("audio_config","rms_norm_eps",1e-5),
    ("audio_config","hidden_act","gelu"), ("audio_config","use_clipped_linears",False),
    ("feature_extractor","sampling_rate",24000), ("feature_extractor","feature_size",64),
    ("feature_extractor","frame_length",400), ("feature_extractor","hop_length",80),
    ("feature_extractor","fft_length",1024), ("feature_extractor","fft_overdrive",True),
    ("feature_extractor","min_frequency",20), ("feature_extractor","max_frequency",7900),
    ("feature_extractor","input_scale_factor",True), ("feature_extractor","dither",.001),
    ("feature_extractor","preemphasis",.97), ("feature_extractor","mel_floor",.01),
    ("feature_extractor","per_bin_mean",[0]*128), ("feature_extractor","per_bin_stddev",[1]*128),
    ("feature_extractor","padding_side","left"), ("feature_extractor","padding_value",1),
    ("feature_extractor","return_attention_mask",False),
    ("feature_extractor","frame_length_ms",30), ("feature_extractor","hop_length_ms",20),
])
def test_bad_semantics_rejected_in_both_languages(tmp_path, section, field, value):
    config, processor = copy.deepcopy(CONFIG), copy.deepcopy(PROCESSOR)
    target = config if section == "audio_config" else processor
    target[section][field] = value
    with pytest.raises(ValueError,match=field):
        (audio.validate_config if section == "audio_config" else audio.validate_processor)(target)
    if BINARY.exists():
        (tmp_path/"config.json").write_text(json.dumps(config))
        (tmp_path/"processor_config.json").write_text(json.dumps(processor))
        result = subprocess.run([BINARY,"--audio",tmp_path],capture_output=True,text=True)
        assert result.returncode != 0 and field in result.stderr, result.stderr


def test_compatible_metadata_and_unused_duration_estimates():
    config, processor = copy.deepcopy(CONFIG), copy.deepcopy(PROCESSOR)
    config.update(_name_or_path="audio-finetune",vision_config=None)
    # Actual spans derive from feature masks, not the duration-estimate helper.
    processor.update(audio_seq_length=1,audio_ms_per_token=999)
    audio.validate_config(config)
    audio.validate_processor(processor)


@pytest.mark.parametrize("mutation,reason", [("missing","missing source tensor"),("dtype","dtype"),("shape","shape")])
def test_bad_scalar_source_preserves_completed_output(tmp_path, mutation, reason):
    snapshot, output = tmp_path/"source", tmp_path/"bundle"
    snapshot.mkdir(); output.mkdir()
    for name,value in (("config.json",CONFIG),("processor_config.json",PROCESSOR),("tokenizer.json",{})):
        (snapshot/name).write_text(json.dumps(value))
    sentinel = output/"successful-result"
    sentinel.write_text("preserve")
    header, cursor = {}, 0
    scalar = "audio_tower.layers.0.feed_forward1.ffw_layer_1.input_min"
    for spec in text_specs()+audio.tensor_specs():
        if spec.source_name == scalar and mutation == "missing":
            continue
        dtype = "F16" if spec.source_name == scalar and mutation == "dtype" else "BF16"
        shape = [1] if spec.source_name == scalar and mutation == "shape" else list(spec.shape)
        header[spec.source_name] = {"dtype":dtype,"shape":shape,"data_offsets":[cursor,cursor+spec.byte_length]}
        cursor += spec.byte_length
    raw = json.dumps(header).encode()
    with (snapshot/"model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q",len(raw))); stream.write(raw)
        stream.truncate(8+len(raw)+cursor)
    with pytest.raises(ValueError,match=reason):
        extract(snapshot,output,audio=True)
    assert sentinel.read_text() == "preserve"
    assert list(output.iterdir()) == [sentinel]
