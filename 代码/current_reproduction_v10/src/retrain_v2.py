"""Reproduce the revised training in a fresh folder using unchanged trainers.

This convenience entry point never evaluates a final-test program or replaces
the published models. All scientific operations use the frozen implementations.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
from common import ROOT, sha, utc, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Without this flag, only validate and print the plan.')
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to((ROOT / 'results').resolve()):
        parser.error('--output must be a new directory beneath results/')
    if target.exists():
        parser.error('Output already exists; retained runs are never overwritten.')
    from reserved_evaluation import verify
    verify()
    print('Fresh revised training folder:', target)
    print('Run all 36 teacher evaluations; train seed 101 in both arms; check the')
    print('original validation gate; then train seeds 211, 307, 401 and 503 in both')
    print('arms and write a new model index. No final-test paths will be loaded.')
    if not args.execute:
        return
    import learning_pilot as pilot
    import learning_fullstudy as full
    pilot.OUT = target / 'learning_001'
    full.PILOT = pilot.OUT
    full.OUT = target / 'learning_002'
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'rerun_registration.json', dict(
        created_utc=utc(), seeds=[101, 211, 307, 401, 503],
        arms=['fixed', 'adaptive'], trainer_changes='Output folders only',
        source_sha256={name: sha(ROOT / 'src' / name) for name in
                       ['learning_pilot.py', 'learning_fullstudy.py', 'retrain_v2.py']},
        final_test_executions_permitted=0,
        schedule='Sequential CPU single-thread; measured time differs from the original partly concurrent run.',
        random_stream_note='Pilot action sampling continues the torch stream after critic initialization at seed+100000; saved RNG state is authoritative.'))
    pilot.prepare()
    for arm in ('fixed', 'adaptive'):
        pilot.train(arm, 101)
    full.register()  # Retains the original teacher and paired validation gate.
    for seed in full.SEEDS:
        for arm in ('fixed', 'adaptive'):
            full.train(seed, arm)
    full.index()
    print('Finished:', full.OUT / 'models.json')


if __name__ == '__main__':
    main()
