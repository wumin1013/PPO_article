"""Verify the portable NNC input snapshot or replay it into a new directory."""
import argparse
from pathlib import Path
import shutil
import evaluate_restored_nnc_v6 as adapter
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('verify', 'run'))
    parser.add_argument('--evidence', type=Path, default=ROOT/'evidence/frozen_main_v6/nnc_runtime')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    adapter.OUT = evidence
    registration = adapter.verify()
    if args.mode == 'verify':
        print('Verified portable checkpoint, configuration, source and driver hashes; no commands executed.')
        return
    if args.output is None:
        parser.error('run requires a new --output directory')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Preserve existing evidence: '+str(output))
    output.mkdir(parents=True)
    for directory in ('source', 'programs'):
        shutil.copytree(evidence/directory, output/directory)
    for name in ('checkpoint.pth', 'baseline_policy.yaml', 'registration.json'):
        shutil.copy2(evidence/name, output/name)
    for spec in registration['runs']:
        name = spec['id']+'_preflight.json'
        shutil.copy2(evidence/name, output/name)
    adapter.OUT = output
    adapter.verify()
    for spec in registration['runs']:
        adapter.worker(spec['id'])
    print('All four replay outcomes retained in', output)


if __name__ == '__main__':
    main()
