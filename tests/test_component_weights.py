"""CPU coverage of standalone component files, extraction, and CLI fallback."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

from tools import bf16_artifact as bf16
from tools.component_weights import write_component
from tools.gemma4_26b_contract import assistant_tensor_specs as assistant26_specs, vision_tensor_specs as vision26_specs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    directory = tmp_path_factory.mktemp("component-loader")
    source = directory / "main.cc"
    source.write_text(r'''
#include "gewell/models/gemma4/31b/component_weights.h"
#include "gewell/models/gemma4/26b_a4b/component_weights.h"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    using namespace gewell::gemma4_31b;
    const auto specs = std::string(argv[1]) == "assistant26" ?
        gewell::gemma4_26b_a4b::assistant_tensor_specs() :
        std::string(argv[1]) == "vision26" ? gewell::gemma4_26b_a4b::vision_tensor_specs() :
        component_specs(std::string(argv[1]) == "assistant" ? Component::assistant : Component::vision);
    gewell::component::File file(argv[2], specs);
    std::cout << file.tensors().size() << " " << file.device_bytes() << "\n";
    for (const auto& tensor : file.tensors())
      std::cout << tensor.physical_id << " " << unsigned(tensor.data[0]) << " " << unsigned(tensor.data[tensor.bytes - 1]) << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
''')
    binary = directory / "loader"
    subprocess.run(["c++", "-std=c++17", "-O1", "-I", str(ROOT / "include"), "-I", str(ROOT / "vendor/nlohmann"),
                    str(source), str(ROOT / "src/component_weights.cc"), str(ROOT / "src/models/gemma4/31b/component_weights.cc"),
                    str(ROOT / "src/models/gemma4/26b_a4b/component_weights.cc"), "-o", str(binary)], check=True)
    return binary


def sparse_component(path, specs, mutate=None, duplicate=False):
    # Reverse storage order to exercise name lookup independently of GPU order.
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for spec in reversed(specs):
        header[spec.source_name] = {"dtype": "BF16", "shape": list(spec.shape),
                                    "data_offsets": [offset, offset + spec.byte_length]}
        offset += spec.byte_length
    if mutate:
        mutate(header)
    raw = json.dumps(header).encode()
    if duplicate:
        name = specs[0].source_name
        raw = raw[:-1] + b"," + json.dumps(name).encode() + b":" + json.dumps(header[name]).encode() + b"}"
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(raw)) + raw)
        output.truncate(8 + len(raw) + offset)
        if not mutate and not duplicate:
            for spec in specs:
                begin, end = header[spec.source_name]["data_offsets"]
                for position in (begin, end - 1):
                    output.seek(8 + len(raw) + position)
                    output.write(bytes([spec.physical_id % 251]))


@pytest.mark.parametrize("component,specs", [("assistant", bf16.assistant_tensor_specs()), ("vision", bf16.vision_tensor_specs()),
                                          ("assistant26", assistant26_specs()), ("vision26", vision26_specs())])
def test_original_names_shapes_and_bit_patterns_load_in_execution_order(loader, tmp_path, component, specs):
    path = tmp_path / "component.safetensors"
    sparse_component(path, specs)
    result = subprocess.run([str(loader), component, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == f"{len(specs)} {sum(bf16.align_up(s.byte_length) for s in specs)}"
    assert lines[1:] == [f"{s.physical_id} {s.physical_id % 251} {s.physical_id % 251}" for s in specs]
    wrong = "vision" if component == "assistant" else "assistant"
    assert subprocess.run([str(loader), wrong, str(path)], capture_output=True).returncode != 0


@pytest.mark.parametrize("corruption", ["dtype", "shape", "missing", "offset", "overlap", "duplicate", "trailing", "header"])
@pytest.mark.parametrize("component,specs", [("assistant", bf16.assistant_tensor_specs()),
                                              ("assistant26", assistant26_specs()), ("vision26", vision26_specs())])
def test_rejects_invalid_component_before_device_loading(loader, tmp_path, corruption, component, specs):
    path = tmp_path / "invalid.safetensors"
    name = specs[0].source_name
    def mutate(header):
        entry = header[name]
        if corruption == "dtype": entry["dtype"] = "F8_E4M3"
        if corruption == "shape": entry["shape"][0] += 1
        if corruption == "missing": del header[name]
        if corruption == "offset": entry["data_offsets"][1] = 2**64 - 1
        if corruption == "overlap": entry["data_offsets"] = [0, specs[0].byte_length]
    sparse_component(path, specs, mutate, duplicate=corruption == "duplicate")
    if corruption == "trailing":
        with path.open("ab") as output: output.write(b"x")
    if corruption == "header":
        with path.open("r+b") as output: output.write(struct.pack("<Q", 2**64 - 1))
    result = subprocess.run([str(loader), component, str(path)], capture_output=True, text=True)
    assert result.returncode == 1, result.stdout
    assert "component" in result.stderr or "parse_error" in result.stderr


def test_extraction_preserves_bits_and_refuses_replacement(tmp_path):
    spec = bf16.TensorSpec(832, -1, bf16.Role.VISION_PATCH_PROJ, "patch", "model.vision_tower.patch", (2, 3))
    raw = struct.pack("<6H", 0, 0x8000, 0x0001, 0x3f80, 0x7f80, 0x7fc0)
    source = tmp_path / "source"
    source.write_bytes(b"prefix" + raw)
    declaration = bf16.SourceTensor(source, source.name, spec.source_name, 6, len(raw))
    output = tmp_path / "vision.safetensors"
    write_component(output, [spec], [declaration], metadata={"component": "vision"},
                    expected_hashes=[hashlib.sha256(raw).digest()])
    record = bf16.read_safetensors_header(output)[spec.source_name]
    assert output.read_bytes()[record.offset:] == raw
    with pytest.raises(bf16.ArtifactError, match="refusing to replace"):
        write_component(output, [spec], [declaration], metadata={})
    failed = tmp_path / "failed.safetensors"
    with pytest.raises(bf16.ArtifactError, match="checksum mismatch"):
        write_component(failed, [spec], [declaration], metadata={}, expected_hashes=[bytes(32)])
    assert not failed.exists() and not failed.with_suffix(".safetensors.partial").exists()


def test_cli_missing_component_paths_and_depth_fallback(tmp_path):
    binary = ROOT / "build/release/gewell"
    if not binary.exists(): pytest.skip("build the application to exercise CLI fallback")
    result = subprocess.run([str(binary), "--mtp-depth", "3", "generate", "missing", "missing", "1", str(tmp_path / "out")],
                            capture_output=True, text=True)
    assert result.returncode != 0  # The input files deliberately do not exist.
    assert "forcing depth to 0" in result.stderr
    caption = subprocess.run([str(binary), "caption", "missing", "missing", "missing", "missing", "1", str(tmp_path / "out")],
                             capture_output=True, text=True)
    assert "image input requires --vision PATH" in caption.stderr
    for option in ("--assistant", "--vision"):
        missing = subprocess.run([str(binary), option], capture_output=True, text=True)
        assert missing.returncode != 0 and "requires a path" in missing.stderr


def test_assistant_models_are_not_interchangeable(loader, tmp_path):
    for name, specs, wrong in [("assistant26", assistant26_specs(), "assistant"),
                               ("assistant", bf16.assistant_tensor_specs(), "assistant26")]:
        path = tmp_path / (name + ".safetensors")
        sparse_component(path, specs)
        result = subprocess.run([str(loader), wrong, str(path)], capture_output=True, text=True)
        assert result.returncode == 1


@pytest.mark.parametrize("component,variable,specs,count,byte_length", [
    ("assistant26", "GEWELL_TEST_26B_ASSISTANT", assistant26_specs(), 48, 839422472),
    ("vision26", "GEWELL_TEST_26B_VISION", vision26_specs(), 356, 1145588832),
])
def test_official_26b_component(loader, component, variable, specs, count, byte_length):
    source = os.environ.get(variable)
    if not source:
        pytest.skip(f"set {variable} to the pinned component safetensors")
    source = Path(source)
    records = bf16.read_safetensors_header(source)
    assert set(records) == {s.source_name for s in specs}
    assert len(specs) == count and sum(s.byte_length for s in specs) == byte_length
    expected = [f"{count} {sum(bf16.align_up(s.byte_length) for s in specs)}"]
    with source.open("rb") as stream:
        for spec in specs:
            record = records[spec.source_name]
            assert record.dtype == "BF16" and record.shape == spec.shape
            stream.seek(record.offset)
            first = stream.read(1)[0]
            stream.seek(record.offset + spec.byte_length - 1)
            last = stream.read(1)[0]
            expected.append(f"{spec.physical_id} {first} {last}")
    result = subprocess.run([str(loader), component, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected


def test_vision_models_are_not_interchangeable(loader, tmp_path):
    for name, specs, wrong in [('vision26', vision26_specs(), 'vision'),
                               ('vision', bf16.vision_tensor_specs(), 'vision26')]:
        path = tmp_path / (name + '.safetensors')
        sparse_component(path, specs)
        result = subprocess.run([str(loader), wrong, str(path)], capture_output=True, text=True)
        assert result.returncode != 0


def test_vision26_contract_has_local_ids_and_bridge():
    specs = vision26_specs()
    assert len(specs) == 356
    assert [s.physical_id for s in specs] == list(range(356))
    assert specs[-1].shape == (2816, 1152)
    assert [(s.source_name, s.shape) for s in specs[:-1]] == [
        (s.source_name, s.shape) for s in bf16.vision_tensor_specs()[:-1]]
