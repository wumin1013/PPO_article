"""Independent paired-initialization, sampling and dense-resolution spot check."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from strong_policy_v4 import tensor_hash
from runtime_physical_time_v5 import install
from metrics_exact_index_v5 import metrics


def read(path):
    return json.loads(path.read_text('utf-8'))


def main():
    torch.set_num_threads(1)
    out = ROOT/'results/recovery_training_v6'
    reg = read(out/'registration.json')
    assert all(sha(ROOT/'src'/name) == digest for name, digest in reg['source_sha256'].items())
    receipts = {arm: read(out/f'seed101_{arm}/receipt.json') for arm in ('adaptive', 'fixed')}
    initial = {arm: torch.load(out/f'seed101_{arm}/initial.pt', weights_only=False, map_location='cpu')
               for arm in receipts}
    a, b = (initial[arm]['policy'] for arm in ('adaptive', 'fixed'))
    assert set(a) == set(b) and all(torch.equal(a[key], b[key]) for key in a)
    for arm, receipt in receipts.items():
        folder = out/f'seed101_{arm}'
        assert receipt['iterations'] == 24000 and receipt['status'] == 'completed'
        assert sha(folder/'final.pt') == sha(folder/'fit_24000.pt') == receipt['model_sha256']
        final = torch.load(folder/'final.pt', weights_only=False, map_location='cpu')
        assert final['offline_iterations'] == 24000 and final['seed'] == 101
        lines = [json.loads(line) for line in (folder/'losses.jsonl').read_text('utf-8').splitlines()]
        assert [row['iteration'] for row in lines] == list(range(1, 24001))
    keys = ('initial_trainable_tensor_sha256', 'sampled_row_stream_sha256', 'feature_tensor_sha256')
    assert all(receipts['adaptive'][key] == receipts['fixed'][key] for key in keys)
    with np.load(out/'training_data/supervision.npz') as data:
        groups = [np.flatnonzero(data['group'] == i) for i in range(4)]
    digest = hashlib.sha256()
    rng = np.random.default_rng(101)
    for _ in range(24000):
        ids = np.concatenate([group[rng.integers(len(group), size=128)] for group in groups])
        digest.update(ids.tobytes())
    assert digest.hexdigest() == receipts['adaptive']['sampled_row_stream_sha256']
    results = []
    for dt_us in (1000, 500):
        file = out/f'seed101_adaptive/evaluation/dense_dt{dt_us}us.npz'
        recorded = read(file.with_suffix('.json'))
        assert sha(file) == recorded['trace_sha256']
        with np.load(file) as data:
            ref, positions, rows = data['reference'], data['positions'], data['rows']
        assert len(positions) == len(rows)+1 and np.array_equal(positions[1:], rows[:, :2])
        install(dt=dt_us/1e6, tolerance=recorded['tolerance_mm'])
        recomputed = metrics(ref, positions, dt_us/1e6, recorded['tolerance_mm'],
                             declared_success=recorded['success'], reason=recorded['reason'])
        for key in ('success', 'complete_time_s', 'elapsed_prefix_s', 'vector_peaks', 'geometry', 'terminal_rest_s'):
            assert recomputed[key] == recorded[key], key
        results.append(dict(dt_s=dt_us/1e6, success=recorded['success'],
            complete_time_s=recorded['complete_time_s'], elapsed_prefix_s=recorded['elapsed_prefix_s'],
            reason=recorded['reason'], geometry=recorded['geometry'],
            terminal_rest_s=recorded['terminal_rest_s'], trace_sha256=sha(file)))
    report = dict(status='passed', created_utc=utc(), audit_source_sha256=sha(__file__),
        registration_sha256=sha(out/'registration.json'), source_hashes_verified=True,
        all_initial_state_tensors_exactly_equal=True, both_final_checkpoints_exactly_24000=True,
        both_loss_logs_have_every_iteration=True, independently_regenerated_same_sample_stream=True,
        sample_stream_sha256=digest.hexdigest(), group_sizes=[len(g) for g in groups],
        paired_receipt_hashes={arm: sha(out/f'seed101_{arm}/receipt.json') for arm in receipts},
        independent_dense_adaptive_resolution_checks=results,
        scope='One paired supervised seed and two saved complete-program traces; no new trajectory evaluation.',
        interpretation='Pairing is verified. The 0.5 ms dense failure remains in the result set and prevents a resolution-robustness claim for this pilot.')
    path = ROOT/'audit/recovery_training_v6_independent_spotcheck.json'
    assert not path.exists()
    write_json(path, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
