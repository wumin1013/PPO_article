"""Reuse unchanged joint gates for a fixed supplemental endpoint, without rollout."""
import argparse
import json
import shutil
from pathlib import Path
from evaluate_historical_dense import ROOT, write, sha
from audit_mixed_resolution_checkpoints import audit


def read(path):
    return Path(path).read_text(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation', type=Path)
    args = parser.parse_args()
    directory = args.evaluation.resolve()
    registration = json.loads(read(directory/'registration.json'))
    checkpoint = Path(registration['checkpoint'])
    assert sha(checkpoint) == registration['checkpoint_sha256']
    results = []
    bindings = {str(checkpoint): sha(checkpoint)}
    for case in ['original', 'dense']:
        for key in ['1ms', 'half_ms']:
            paths = list(directory.glob(f'eval_*_{case}_{key}.json'))
            assert len(paths) == 1
            path = paths[0]
            result = json.loads(read(path))
            assert result['checkpoint_sha256'] == sha(checkpoint)
            assert result['trace_sha256'] == sha(path.with_suffix('.npz'))
            assert (result['observation_dimension'], result['actor_outputs'], result['policy_memory_scalars']) == (14, 3, 0)
            results.append((path, result))
            for item in [path, path.with_suffix('.npz')]:
                bindings[str(item)] = sha(item)
    # An immutable adapter view lets the existing gate implementation run unchanged.
    view = directory/'joint_gate_audit'
    view.mkdir(exist_ok=False)
    shutil.copy2(checkpoint, view/'checkpoint_0000.pt')
    write(view/'evaluations.json', [dict(update=0)])
    for path, result in results:
        suffix = path.name.split('_', 2)[2]
        shutil.copy2(path, view/('eval_0000_'+suffix))
        metric = path.with_name(path.stem+'_metrics.json')
        if metric.exists():
            shutil.copy2(metric, view/('eval_0000_'+metric.name.split('_', 2)[2]))
            bindings[str(metric)] = sha(metric)
    write(view/'adapter_provenance.json', dict(
        original_evaluation=str(directory), checkpoint=str(checkpoint),
        source_sha256=bindings, new_simulations=0,
        reason='File-name adaptation only; unchanged historical, S2, two-action, all34 and numerical-margin gates.'))
    audit(str(view), ROOT/'all34_two_action_comparator_addendum_20260923.json')
    assert all(sha(p) == h for p, h in bindings.items())


if __name__ == '__main__':
    main()
