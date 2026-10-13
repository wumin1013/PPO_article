"""Bind every fixed seed checkpoint and its successes/failures; no seed selection."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from evaluate_historical_dense import write, sha
from replay_lowdim_training_stage import SUPPLEMENT, SEEDS

STAGES = [
    ('compression', 'fit_0600.pt', 'eval_0600'),
    ('ppo_initial', 'checkpoint_0016.pt', 'eval_0016'),
    ('geometry_fit', 'fit_1800.pt', 'eval_1800'),
    ('saturation_fit', 'fit_6000.pt', 'eval_6000'),
    ('ppo_deficit', 'checkpoint_0024.pt', 'eval_0024'),
    ('preview_fit', 'fit_12000.pt', 'eval_12000'),
    ('ppo_preview', 'checkpoint_0024.pt', 'eval_0024'),
    ('ppo_guided', 'checkpoint_0024.pt', 'eval_0024'),
    ('farvector_fit', 'fit_16000.pt', 'eval_16000'),
    ('local_feedback_fit', 'fit_08000.pt', 'eval_8000'),
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    rows, records, bindings = [], [], {}
    for seed in SEEDS:
        root = SUPPLEMENT/'independent_training'/f'seed_{seed}'
        for stage_name, checkpoint_name, prefix in STAGES:
            folder = root/stage_name
            registration = folder/'registration.json'
            if not registration.exists():
                continue
            bindings[str(registration)] = sha(registration)
            registration_value = read(registration)
            assert registration_value['seed'] == seed
            if registration_value.get('initial_checkpoint'):
                parent = Path(registration_value['initial_checkpoint'])
                assert parent.is_relative_to(root), parent
                assert sha(parent) == registration_value['initial_checkpoint_sha256']
            receipt_path = folder/'replay_receipt.json'
            receipt = read(receipt_path) if receipt_path.exists() else {'status': 'pending'}
            if receipt_path.exists():
                bindings[str(receipt_path)] = sha(receipt_path)
            checkpoint = folder/checkpoint_name
            record = dict(seed=seed, stage=stage_name, fixed_checkpoint=str(checkpoint),
                checkpoint_exists=checkpoint.exists(), stage_status=receipt['status'], results={})
            if checkpoint.exists():
                record['checkpoint_sha256'] = sha(checkpoint)
            for case in ('original', 'dense'):
                path = folder/f'{prefix}_{case}.json'
                if stage_name == 'compression':
                    if case == 'original':
                        continue
                    path = folder/f'{prefix}.json'
                if not path.exists():
                    continue
                value = read(path)
                if value.get('checkpoint_sha256'):
                    assert value['checkpoint_sha256'] == record['checkpoint_sha256'], path
                trace = path.with_suffix('.npz')
                assert trace.exists()
                bindings[str(path)] = sha(path)
                success = bool(value['success'])
                assert success == (value.get('complete_time_s') is not None)
                error = value.get('error')
                evaluation_status = ('complete_success' if success else
                    'incomplete_deadline' if str(error).startswith('TimeoutError:') else 'failed_execution')
                result = dict(success=success, complete_time_s=value.get('complete_time_s'),
                    elapsed_prefix_or_complete_s=value.get('elapsed_prefix_or_complete_s'),
                    reason=value['reason'], error=error, evaluation_status=evaluation_status,
                    result=str(path), result_sha256=sha(path),
                    trace=str(trace), trace_sha256=sha(trace))
                record['results'][case] = result
                rows.append(dict(seed=seed, stage=stage_name, case=case,
                    success=success, complete_time_s=result['complete_time_s'],
                    prefix_or_complete_s=result['elapsed_prefix_or_complete_s'],
                    reason=result['reason'], evaluation_status=evaluation_status,
                    fixed_checkpoint_sha256=record['checkpoint_sha256']))
            records.append(record)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
        scope='Fixed-stage checkpoints from independent initializations using shared historical training data. Intermediate rows are not final repetitions.',
        records=records, source_sha256=bindings,
        selected_paper_model_changed=False, independent_final_pipeline_repetitions_established=0)
    write(SUPPLEMENT/'independent_fixed_stage_audit.json', result)
    with (SUPPLEMENT/'independent_fixed_stage_results.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('FIXED_STAGE_AUDIT', len(records), 'stages;', len(rows), 'case checks; final repetitions not claimed')


if __name__ == '__main__':
    main()
