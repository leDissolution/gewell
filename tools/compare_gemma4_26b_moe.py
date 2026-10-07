#!/usr/bin/env python3
"""Check exact MoE routing/permutation and report grouped BF16 numerical errors."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference',type=Path)
    parser.add_argument('native',type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.reference/'manifest.json').read_text())
    report = {}
    for name, case in manifest['cases'].items():
        a, b = args.reference/name, args.native/name
        ids = np.fromfile(a/'ids.i32',dtype='<i4')
        actual = np.fromfile(b/'ids.i32',dtype='<i4')
        if not np.array_equal(ids,actual): raise ValueError(f'{name}: selected expert mismatch')
        weights = np.fromfile(a/'weights.f32',dtype='<f4')
        actual_weights = np.fromfile(b/'weights.f32',dtype='<f4')
        if not np.array_equal(weights,actual_weights): raise ValueError(f'{name}: routing weight mismatch')
        if len(ids):
            order = np.argsort(ids,kind='stable')
            offsets = np.searchsorted(ids[order],np.arange(129))
            if not np.array_equal(order,np.fromfile(b/'assignments.i32',dtype='<i4')):
                raise ValueError(f'{name}: assignment loss, duplication or permutation mismatch')
            if not np.array_equal(offsets,np.fromfile(b/'offsets.i32',dtype='<i4')):
                raise ValueError(f'{name}: expert offsets mismatch')
        row = {'rows':case['rows'],'assignments':len(ids),'routing_exact':True}
        if not case['dispatch_only']:
            def floats(path):
                return (np.fromfile(path,dtype='<u2').astype('<u4') << 16).view('<f4')
            x, y = floats(a/'output.bf16'), floats(b/'output.bf16')
            if x.shape != y.shape or not np.isfinite(y).all(): raise ValueError(f'{name}: invalid output')
            difference = x.astype('f8')-y.astype('f8')
            row.update(exact=bool(np.array_equal(x,y)),differing_elements=int(np.count_nonzero(x!=y)),
                       relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(x.astype('f8')),1e-12)),
                       max_abs=float(np.max(np.abs(difference))))
        report[name] = row
    (args.native/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
