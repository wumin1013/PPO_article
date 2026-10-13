"""Reuse the existing PPO loop with explicit historical-residual components."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from evaluate_historical_dense import ROOT, BASE, HIST, CASE, sha, write, verify_bound_inputs

STAGE = ROOT / 'historical_residual_ppo_seed922301_pilot'


def worker():
    import train_compact_ppo_mc as training
    from historical_residual_ppo import HistoricalResidualRuntime, ResidualPolicy
    from paper_complete_evaluation import evaluate

    verify_bound_inputs()
    torch.set_num_threads(1)
    reference = np.loadtxt(CASE / 'reference.csv', delimiter=',', skiprows=1)
    policy = ResidualPolicy()
    initial = evaluate(policy, HistoricalResidualRuntime(reference), STAGE / 'zero_residual.npz', time.perf_counter()+120.)
    expected = np.load(ROOT / 'historical_dense_cluster_preview_probe/floor_3/trace.npz')
    actual = np.load(STAGE / 'zero_residual.npz')
    assert np.array_equal(expected['rows'], actual['rows'])
    assert np.array_equal(expected['actions'], actual['actions'])
    runtime = HistoricalResidualRuntime(reference)
    first = runtime.observation()
    state = runtime.state()
    delta = np.array([.002, -.015, .01])
    following, reward, done, info = runtime.step(delta)
    replay = HistoricalResidualRuntime(reference)
    replay.restore(state)
    assert np.array_equal(replay.observation(), first)
    following2, reward2, done2, info2 = replay.step(delta)
    assert np.array_equal(info['rows'], info2['rows'])
    assert np.array_equal(following, following2) and reward == reward2 and done == done2
    write(STAGE / 'interface_validation.json', dict(zero_residual_rows_identical=True,
        zero_residual_actions_identical=True, restored_next_step_and_observation_identical=True,
        zero_residual_result=initial, reference_trace_sha256=sha(ROOT / 'historical_dense_cluster_preview_probe/floor_3/trace.npz')))

    # Reuse the audited collection/update/checkpoint loop; substitution is local
    # to this worker process and recorded in the training registration.
    training.TrainingRuntime = HistoricalResidualRuntime
    training.CompactPolicy = ResidualPolicy
    args = SimpleNamespace(seed=922301, dimension=17, action_dim=3, no_swing=False, preview=3.,
        potential_speed=30., remaining_encoding='log', resume=None, steps=16384, batch=1024,
        evaluate_every=4, budget=1050.)
    training.worker(args, STAGE)
    result = json.loads((STAGE / 'result.json').read_text(encoding='utf-8'))
    result.update(training_kind='PPO training of an added three-action residual around the inherited historical policy',
        independently_initialized_residual_seed=922301, independent_full_method_training=False,
        residual_observation_dimension=17, inherited_observation_dimension=72,
        inherited_causal_memory_scalars=474, inherited_model=str(HIST),
        success_scope='Development pilot; not final independent three-seed acceptance or a complete-method dimensional reduction.')
    write(STAGE / 'result.json', result)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    manifest = verify_bound_inputs()
    STAGE.mkdir(exist_ok=False)
    source = STAGE / 'source'
    source.mkdir()
    files = [Path(__file__), ROOT / 'historical_residual_ppo.py', ROOT / 'train_compact_ppo_mc.py',
        ROOT / 'learning_update_compact_mc.py', ROOT / 'compact_control.py', ROOT / 'local_arc_support.py',
        ROOT / 'evaluate_historical_dense.py', BASE / 'learning_update_v4.py', BASE / 'joint_policy.py']
    for path in files:
        shutil.copy2(path, source / path.name)
    hashes = {str(path): sha(path) for path in files}
    write(STAGE / 'registration.json', dict(kind='Historical main-method residual PPO pilot',
        wall_budget_s=1200, child_timeout_s=1150, ppo_training_budget_s=1050,
        seed=922301, new_decisions=16384, batch_size=1024, scheduled_evaluation_every_updates=4,
        source_sha256=hashes, inherited_manifest=manifest,
        reference_sha256=sha(CASE / 'reference.csv'),
        base='Historical selected policy plus clustered-turn arc support and3mm preview floor, preserved under exact zero-residual replay.',
        residual_actor='17inputs,2x64tanh,3Gaussian residual outputs; zero mean initialized; std=[.01,.04,.04]; actor and critic separate.',
        action_composition='tanh(historical latent)+raw Gaussian residual, clipped in the common normalized physical-action coordinates, then atanh. PPO log probabilities use the original raw residual, never clipped values.',
        data_and_memory_scope='Complete controller retains inherited72-input policy and474 causal memory scalars.17inputs describe the new residual only. Full parent training is shared, not independent.',
        adoption='First require exact zero-residual and checkpoint continuation verification; then scheduled complete success before time. All failures are preserved.',
        stop='At16384 decisions or registered wall limit. No wider search or baseline optimization; review evidence before continuing.',
        baseline='Dense S2=19.957s;5percent threshold18.95915s; a fair two-output comparison is still required.', goal_complete=False))
    start = time.perf_counter()
    with (STAGE / 'stdout.txt').open('w', encoding='utf-8') as out, (STAGE / 'stderr.txt').open('w', encoding='utf-8') as err:
        try:
            run = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--worker'], stdout=out, stderr=err, timeout=1150)
            status, code = 'finished', run.returncode
        except subprocess.TimeoutExpired:
            status, code = 'timeout', None
    write(STAGE / 'receipt.json', dict(status=status, exit_code=code, wall_s=time.perf_counter()-start,
          sources_unchanged=all(sha(path)==expected for path, expected in hashes.items()), goal_complete=False))
    print((STAGE / 'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code != 0:
        print((STAGE / 'stderr.txt').read_text(encoding='utf-8')[-5000:])


if __name__ == '__main__':
    main()
