"""Cross-language inventory and stacked-source slice coverage for 26B."""
import math
import os
from pathlib import Path
import subprocess
import unittest
from collections import defaultdict

from tools.gemma4_26b_contract import native_tensor_specs, source_shapes, Role


class ContractTest(unittest.TestCase):
    def test_every_source_element_has_exactly_one_native_entry(self):
        specs = native_tensor_specs()
        self.assertEqual(len(specs), 12117)
        self.assertEqual(sum(s.byte_length for s in specs), 50466283580)
        ranges = defaultdict(list)
        for spec in specs:
            ranges[spec.source_name].append((spec.source_element_offset,
                                            spec.source_element_offset + math.prod(spec.shape)))
        self.assertEqual(ranges.keys(), source_shapes().keys())
        for name, shape in source_shapes().items():
            end = 0
            for begin, stop in sorted(ranges[name]):
                self.assertEqual(begin, end, name)
                end = stop
            self.assertEqual(end, math.prod(shape), name)

    def test_gate_up_halves_and_separate_expert_names(self):
        specs = native_tensor_specs()
        for layer in (0, 29):
            for expert in (0, 127):
                group = {s.role: s for s in specs if s.layer == layer and s.expert == expert}
                gate, up, down = (group[r] for r in (Role.EXPERT_GATE_PROJ, Role.EXPERT_UP_PROJ,
                                                    Role.EXPERT_DOWN_PROJ))
                self.assertEqual(gate.source_element_offset, expert * 1408 * 2816)
                self.assertEqual(up.source_element_offset, (expert * 1408 + 704) * 2816)
                self.assertEqual(down.source_element_offset, expert * 2816 * 704)
                for s, name in ((gate, 'gate'), (up, 'up'), (down, 'down')):
                    self.assertEqual(s.separate_source_name,
                                     f'model.language_model.layers.{layer}.experts.{expert}.{name}_proj.weight')

    def test_native_inventory(self):
        binary = Path(os.environ.get('GEWELL_26B_CONTRACT_TEST',
                                     'build/cleanup-host/gewell_gemma4_26b_model_contract_test'))
        if not binary.is_file():
            self.skipTest('build the native 26B contract test to compare inventories')
        lines = subprocess.check_output([str(binary), '--dump'], text=True).splitlines()
        actual = [tuple(map(int, line.split())) for line in lines]
        expected = [(int(s.role), s.layer, s.expert, len(s.shape), s.shape[0],
                     s.shape[1] if len(s.shape) == 2 else 0) for s in native_tensor_specs()]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
