"""CPU serving-bundle validation, using sparse metadata from an opt-in local bundle."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def loader(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required')
    work = tmp_path_factory.mktemp('26b-serving-loader')
    source = work/'main.cc'
    source.write_text('''
#include "gewell/models/gemma4/26b_a4b/serving_assets.h"
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try { auto assets = gewell::gemma4_26b_a4b::ServingAssets::Open(argv[1]); }
  catch (const std::exception& e) { std::cerr << e.what(); return 1; }
}
''')
    binary = work/'loader'
    sources = ['src/models/gemma4/26b_a4b/artifact.cc', 'src/models/gemma4/26b_a4b/serving_assets.cc',
               'src/tokenizer.cc', 'src/models/gemma4/text/contract.cc', 'src/models/gemma4/text/chat_template.cc']
    subprocess.run([compiler, '-std=c++17', '-I', str(ROOT/'include'), '-I', str(ROOT/'vendor/nlohmann'),
                    str(source), *(str(ROOT/p) for p in sources), '-lcrypto', '-o', str(binary)], check=True)
    return binary


@pytest.mark.parametrize('bundle_env', ['GEWELL_TEST_26B_BUNDLE', 'GEWELL_TEST_26B_MIXED_BUNDLE'])
def test_bundle_metadata_and_custom_provenance(loader, tmp_path, bundle_env):
    bundle = os.environ.get(bundle_env)
    if not bundle:
        pytest.skip('set GEWELL_TEST_26B_BUNDLE to a converted local26B bundle')
    bundle = Path(bundle)
    actual = subprocess.run([str(loader), str(bundle)], text=True, capture_output=True)
    assert actual.returncode == 0, actual.stderr
    manifest = json.loads((bundle/'manifest.json').read_text())
    # Startup authenticates metadata, not all 50GB of payload; full verification is separate.
    with (bundle/manifest['artifact']['file']).open('rb') as source, (tmp_path/'custom.gwt').open('wb') as out:
        out.write(source.read(876544))
        out.truncate(manifest['artifact']['file_bytes'])
        # Quantized globals are part of bounded startup validation.
        for entry in manifest['tensors']:
            if entry['dtype'] != 'BF16':
                offset = entry['file_offset'] + entry['byte_length'] - 8
                source.seek(offset); out.seek(offset); out.write(source.read(8))
    shutil.copyfile(bundle/'tokenizer.json', tmp_path/'tokenizer.json')
    manifest['artifact']['file'] = 'custom.gwt'
    manifest['model'].update(repository='community/custom', revision='custom-revision')
    manifest['serving'].update(repository='community/tokenizer', revision='custom-revision')
    def run(value):
        (tmp_path/'manifest.json').write_text(json.dumps(value))
        return subprocess.run([str(loader), str(tmp_path)], text=True, capture_output=True)
    result = run(manifest)
    assert result.returncode == 0, result.stderr
    for section, key, value, expected in [
        ('artifact', 'header_sha256', '0'*64, 'weight manifest'),
        ('artifact', 'entry_table_sha256', '0'*64, 'weight manifest'),
        ('artifact', 'payload_sha256', '0'*64, 'weight manifest'),
        ('model', 'config_sha256', '1'*64, 'model manifest'),
        ('model', 'index_sha256', '0'*64, 'model manifest'),
        ('artifact', 'file', '../escape.gwt', 'local basename'),
    ]:
        bad = copy.deepcopy(manifest)
        bad[section][key] = value
        result = run(bad)
        assert result.returncode == 1 and expected in result.stderr, result.stderr
    bad = copy.deepcopy(manifest)
    bad['architecture'] = 'gemma4_31b'
    result = run(bad)
    assert result.returncode == 1 and 'wrong 26B' in result.stderr
    bad = copy.deepcopy(manifest)
    bad['serving']['assets'][0]['sha256'] = '0'*64
    result = run(bad)
    assert result.returncode == 1 and 'SHA-256 mismatch' in result.stderr

    bad = copy.deepcopy(manifest)
    bad['format'] = ('gemma4-26b-a4b-mixed-v1' if manifest['format'] == 'gemma4-26b-a4b-bf16-v1'
                     else 'gemma4-26b-a4b-bf16-v1')
    result = run(bad)
    assert result.returncode == 1 and 'format does not match' in result.stderr


@pytest.mark.parametrize('bundle_env', ['GEWELL_TEST_26B_BUNDLE', 'GEWELL_TEST_26B_MIXED_BUNDLE',
                                       'GEWELL_TEST_NATIVE_ARTIFACT'])
def test_text_codec_dispatch_and_jsonl_recovery(bundle_env):
    bundle = os.environ.get(bundle_env)
    binary = Path(os.environ.get('GEWELL_TEST_NATIVE_BINARY', ROOT / 'build/gewell'))
    if not bundle or not binary.is_file():
        pytest.skip('local native binary and bundle required')
    def run(requests):
        return subprocess.run([str(binary), 'text-codec', bundle],
                              input=''.join(json.dumps(r) + '\n' for r in requests),
                              text=True, capture_output=True)
    encoded = run([{'operation': 'encode', 'text': 'café hello'}])
    assert encoded.returncode == 0, encoded.stderr
    tokens = json.loads(encoded.stdout)['token_ids']
    checked = run([{'operation': 'decode', 'tokens': tokens},
                   {'operation': 'completion', 'tokens': tokens},
                   {'operation': 'chat', 'messages': [{'role': 'user', 'content': 'Hello'}]},
                   {'operation': 'chat_decode', 'tokens': tokens},
                   {'operation': 'invalid'},
                   {'operation': 'encode', 'text': 'café hello'}])
    assert checked.returncode == 1
    rows = [json.loads(line) for line in checked.stdout.splitlines()]
    assert rows[0]['text'] == 'café hello'
    assert rows[1]['token_ids'] == tokens
    assert rows[2]['token_ids'][0] == 2 and 'Hello' in rows[2]['prompt']
    assert rows[3]['content'] == 'café hello' and rows[3]['reasoning'] == ''
    assert 'unknown codec operation' in rows[4]['error']
    assert rows[5]['token_ids'] == tokens


@pytest.mark.parametrize('bundle_env', ['GEWELL_TEST_26B_BUNDLE', 'GEWELL_TEST_26B_MIXED_BUNDLE'])
def test_inspect_26b_command(bundle_env):
    bundle = os.environ.get(bundle_env)
    binary = Path(os.environ.get('GEWELL_TEST_NATIVE_BINARY', ROOT / 'build/gewell'))
    if not bundle or not binary.is_file():
        pytest.skip('local native binary and26B bundle required')
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    result = subprocess.run([str(binary), '--log-format', 'json', 'inspect',
                             str(bundle / manifest['artifact']['file'])], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    fields = {row['name']: row['value'] for line in result.stdout.splitlines()
              for row in [json.loads(line)] if row['event'] == 'field'}
    assert fields['architecture'] == 'gemma4_26b_a4b'
    assert fields['weight_format'] == manifest['format']
    assert fields['validation'] == 'header+table (payload not scanned)'
    assert fields['physical_tensors'] == 12117 and fields['logical_tensors'] == 12118
    assert fields['lm_head_target_physical_id'] == 0
    assert fields['file_bytes'] == manifest['artifact']['file_bytes']
    assert fields['payload_bytes'] == fields['file_bytes'] - 876544
