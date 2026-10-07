import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import bf16_artifact as wire
from tools import gemma4_26b_artifact as artifact
from tools import gemma4_26b_convert as converter
from tools.gemma4_26b_contract import TensorSpec, Role


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        specs = (TensorSpec(0, Role.EMBEDDING, -1, -1, (2, 3), 'embedding'),
                 TensorSpec(1, Role.EXPERT_UP_PROJ, 29, 127, (2, 2), 'stacked', 10))
        self.patched = patch.object(artifact, 'native_tensor_specs', return_value=specs)
        self.patched.start()
        conversion_specs = patch.object(converter, "native_tensor_specs", return_value=specs)
        conversion_specs.start()
        self.addCleanup(conversion_specs.stop)
        mixed_specs = patch.object(converter.mixed, "native_tensor_specs", return_value=specs)
        mixed_specs.start()
        self.addCleanup(mixed_specs.stop)
        self.addCleanup(self.patched.stop)
        source = self.root / 'source'
        source.write_bytes(bytes(range(48)))
        self.views = (wire.SourceTensor(source, 'source', 'embedding', 2, 12),
                      wire.SourceTensor(source, 'source', 'stacked', 20, 8))
        self.path = self.root / 'weights.gwt'
        artifact.write_artifact(self.path, self.views, bytes(32), bytes(32))

    def test_streamed_ranges_and_padding(self):
        result = artifact.inspect_artifact(self.path, verify=True)
        self.assertEqual(result['logical_data_bytes'], 20)
        with self.path.open('rb') as f:
            f.seek(result['data_offset'])
            self.assertEqual(f.read(12), bytes(range(2, 14)))
            f.seek(result['data_offset'] + wire.ALIGNMENT)
            self.assertEqual(f.read(8), bytes(range(20, 28)))

    def test_rejects_other_model_header(self):
        with self.path.open('r+b') as f:
            f.write(wire.MAGIC)
        with self.assertRaisesRegex(ValueError, 'identity'):
            artifact.inspect_artifact(self.path)

    def test_startup_does_not_scan_payload_but_verify_does(self):
        with self.path.open('r+b') as f:
            f.seek(artifact.plan()['data_offset'])
            f.write(b'\xff')
        artifact.inspect_artifact(self.path)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            artifact.inspect_artifact(self.path, verify=True)

    def test_rejects_nonzero_tensor_padding(self):
        with self.path.open('r+b') as f:
            f.seek(artifact.plan()['data_offset'] + 12)
            f.write(b'\x01')
        with self.assertRaisesRegex(ValueError, 'padding'):
            artifact.inspect_artifact(self.path, verify=True)

    def test_rejects_wrong_geometry_even_with_updated_checksums(self):
        raw = bytearray(self.path.read_bytes())
        raw[wire.HEADER_BYTES + 8] ^= 1  # dim0
        table_end = wire.HEADER_BYTES + 2 * wire.ENTRY_BYTES
        header, _ = wire.decode_header(bytes(raw[:wire.HEADER_BYTES]))
        raw[:wire.HEADER_BYTES] = artifact.encode_header(bytes(32), bytes(32),
            hashlib.sha256(raw[wire.HEADER_BYTES:table_end]).digest(), header.payload_sha256)
        self.path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'entry 0'):
            artifact.inspect_artifact(self.path)

    def test_package_existing_native_artifact_without_rewriting_weights(self):
        from types import SimpleNamespace
        assets = self.root / 'assets'
        assets.mkdir()
        (assets / 'tokenizer.json').write_text('{"test":true}')
        before = (self.path.stat().st_ino, self.path.read_bytes())
        args = SimpleNamespace(mask=None, input_scales=None, verify=None, snapshot=None,
                               artifact=self.path, serving_snapshot=assets, plan=False, output=self.root)
        result = converter.convert_bundle(args)
        self.assertEqual((self.path.stat().st_ino, self.path.read_bytes()), before)
        self.assertTrue((self.root / 'manifest.json').is_file())
        self.assertEqual(result['tensor_count'], 2)
        with self.assertRaisesRegex(ValueError, 'existing output'):
            converter.convert_bundle(args)

    def test_native_packaging_records_matching_snapshot_provenance(self):
        import json
        from types import SimpleNamespace
        revision = 'a' * 40
        assets = self.root / 'models--google--gemma-4-26B-A4B-it' / 'snapshots' / revision
        assets.mkdir(parents=True)
        (assets / 'tokenizer.json').write_text('{"test":true}')
        for name in ('config.json', 'model.safetensors.index.json'):
            (assets / name).write_bytes(b'{}')
        digest = hashlib.sha256(b'{}').digest()
        self.path.unlink()
        artifact.write_artifact(self.path, self.views, digest, digest)
        before = (self.path.stat().st_ino, self.path.read_bytes())
        args = SimpleNamespace(mask=None, input_scales=None, verify=None, snapshot=None,
                               artifact=self.path, serving_snapshot=assets, plan=False, output=self.root)
        converter.convert_bundle(args)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        for section in ('model', 'serving'):
            self.assertEqual(manifest[section]['repository'], 'google/gemma-4-26B-A4B-it')
            self.assertEqual(manifest[section]['revision'], revision)
        self.assertEqual((self.path.stat().st_ino, self.path.read_bytes()), before)

    def test_does_not_overwrite_existing_artifact(self):
        with self.assertRaises(FileExistsError):
            artifact.write_artifact(self.path, self.views, bytes(32), bytes(32))
        artifact.inspect_artifact(self.path, verify=True)


class NativeReaderTest(unittest.TestCase):
    def setUp(self):
        import os
        self.binary = Path(os.environ.get('GEWELL_26B_ARTIFACT_PROBE',
                                          'build/cleanup-host/gewell_gemma4_26b_artifact_probe'))
        if not self.binary.is_file():
            self.skipTest('build the native 26B artifact probe')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'sparse.gwt'
        # Real production geometry with sparse payload: metadata validation must
        # not touch 47 GiB of weights, and must reject a rehashed wrong entry.
        table = bytearray()
        offset = artifact.plan()['data_offset']
        for s in artifact.native_tensor_specs():
            table.extend(wire.ENTRY_STRUCT.pack(s.physical_id, s.layer, int(s.role), len(s.shape), 0,
                s.shape[0], s.shape[1] if len(s.shape) == 2 else 0, 0, offset, s.byte_length, bytes(32)))
            offset += wire.align_up(s.byte_length)
        self.table = table
        self.publish()

    def publish(self):
        with self.path.open('wb') as f:
            f.write(artifact.encode_header(bytes(32), bytes(32), hashlib.sha256(self.table).digest(), bytes(32)))
            f.write(self.table)
            f.truncate(artifact.plan()['file_bytes'])

    def run_probe(self):
        import subprocess
        return subprocess.run([str(self.binary), str(self.path)], capture_output=True, text=True)

    def test_real_geometry_metadata_only(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('12117 tensors', result.stdout)

    def test_native_rejects_rehashed_wrong_expert_shape(self):
        specs = artifact.native_tensor_specs()
        expert = next(s for s in specs if s.expert == 127 and s.layer == 29)
        self.table[expert.physical_id * wire.ENTRY_BYTES + 8] ^= 1
        self.publish()
        result = self.run_probe()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid 26B tensor entry', result.stderr)


if __name__ == '__main__':
    unittest.main()
