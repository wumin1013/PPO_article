"""Explicit validation replay plus byte-provenance and physical-output comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import prepare_frozen_replay as prepared

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_RESULT_FIELDS = {'inference_median_us', 'inference_p95_us', 'evaluation_wall_s'}
EXCLUDED_TRACE_COLUMNS = {'control_status__direct_mapping_us', 'control_status__direct_observation_features_us'}


def compare_results(original, replay):
    keys = (set(original) | set(replay)) - EXCLUDED_RESULT_FIELDS
    different = sorted(key for key in keys if key not in original or key not in replay or original[key] != replay[key])
    return dict(checked_top_level_fields=len(keys),
        non_timing_metrics_equal=not different,
        different_fields=different,
        excluded_fields=sorted(EXCLUDED_RESULT_FIELDS))


def compare_traces(original_path, replay_path):
    with np.load(original_path,allow_pickle=False) as original, np.load(replay_path,allow_pickle=False) as replay:
        if not np.array_equal(original['columns'],replay['columns']) or original['data'].shape != replay['data'].shape:
            raise ValueError('Replay trace columns or sample count differ')
        columns=original['columns']; a,b=original['data'],replay['data']
        differences={}
        for index,name in enumerate(columns):
            if name in EXCLUDED_TRACE_COLUMNS:continue
            if not np.array_equal(a[:,index],b[:,index],equal_nan=True):
                finite=np.isfinite(a[:,index]) & np.isfinite(b[:,index])
                differences[str(name)]=dict(max_abs_difference=float(np.max(np.abs(a[finite,index]-b[finite,index]))) if finite.any() else None,
                    finite_mask_equal=bool(np.array_equal(np.isfinite(a[:,index]),np.isfinite(b[:,index]))))
        return dict(samples=len(a), total_columns=len(columns),
            checked_non_timing_columns=sum(name not in EXCLUDED_TRACE_COLUMNS for name in columns),
            excluded_timing_columns=sorted(set(columns) & EXCLUDED_TRACE_COLUMNS),
            non_timing_trace_exactly_equal=not differences,different_columns=differences)


def validate(replay_root, execute=False, run_key='formal/direct_seed11'):
    replay_root=Path(replay_root).resolve()
    prepared.verify(replay_root)
    manifest=json.loads((replay_root/'replay_manifest.json').read_text(encoding='utf-8'))
    if run_key not in manifest['copied_model_runs']:raise ValueError('Run absent from prepared frozen models')
    original=ROOT/prepared.STUDY/run_key/'evaluation_validation_nominal'
    output=replay_root/prepared.STUDY/run_key/'evaluation_validation_nominal'
    evidence=replay_root/'replay_validation_check';evidence.mkdir(exist_ok=True)
    original_summary=json.loads((original/'summary.json').read_text(encoding='utf-8'))
    ids=[row['path_id'] for row in original_summary['rows']]
    original_files=[ROOT/'PPO_project/main.py',original/'summary.json'] + [original/f'{path_id}.{suffix}' for path_id in ids for suffix in ('json','npz')]
    before={path.relative_to(ROOT).as_posix():prepared.sha(path.read_bytes()) for path in original_files}
    command=['python','PPO_project/tools/preview_direct_explore_resume.py','evaluate','--run',
        (prepared.STUDY/run_key).as_posix(),'--split','validation']
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(), replay_root=str(replay_root),
        run=run_key,split='validation',command=command,working_directory=str(replay_root),
        interpretation='One deterministic replay of an existing trained seed, not a new independent replicate or test-set experiment',
        new_training_steps=0,source_hash_normalization=False,original_workspace_rollback=False,
        original_files_before=before,executed_now=execute,
        validator_sha256_at_execution=prepared.sha(Path(__file__).read_bytes()))
    if execute:
        if output.exists() and list(output.glob('*.json')):
            raise FileExistsError('Preserve existing replay evaluation; compare without --execute or prepare a new isolated root')
        if (evidence/'execution.json').exists():raise FileExistsError('Preserve the original replay execution record')
        (evidence/'execution.json').write_bytes(prepared.encoded(record))
        with (evidence/'console.log').open('w',encoding='utf-8') as log:
            process=subprocess.run([sys.executable,*command[1:]],cwd=replay_root,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONIOENCODING':'utf-8'},check=False)
        record['returncode']=process.returncode
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
        (evidence/'execution.json').write_bytes(prepared.encoded(record))
        if process.returncode:raise RuntimeError(f'Replay failed; inspect {evidence}/console.log')
    else:
        if not (evidence/'execution.json').exists():
            raise ValueError('No recorded explicit replay execution found')
        execution=json.loads((evidence/'execution.json').read_text(encoding='utf-8'))
        if execution['run'] != run_key or execution.get('returncode') != 0:
            raise ValueError('Saved explicit execution belongs to another or failed replay')
        before={name.replace('\\','/'):value for name,value in execution['original_files_before'].items()}
    replay_summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    if {row['path_id'] for row in replay_summary['rows']} != set(ids):
        raise ValueError('Replay did not evaluate the same complete validation path set')
    comparisons=[]
    for path_id in ids:
        old=json.loads((original/f'{path_id}.json').read_text(encoding='utf-8'))
        new=json.loads((output/f'{path_id}.json').read_text(encoding='utf-8'))
        comparisons.append(dict(path_id=path_id,strict_success=new['success'],elapsed_time_s=new['elapsed_time_s'],
            metrics=compare_results(old,new),trace=compare_traces(original/f'{path_id}.npz',output/f'{path_id}.npz')))
    after={path.relative_to(ROOT).as_posix():prepared.sha(path.read_bytes()) for path in original_files}
    result=dict(run=run_key,split='validation',paths=comparisons,original_files_unchanged=before==after,
        all_non_timing_metrics_equal=all(row['metrics']['non_timing_metrics_equal'] for row in comparisons),
        all_non_timing_trace_values_exactly_equal=all(row['trace']['non_timing_trace_exactly_equal'] for row in comparisons),
        verified_frozen_source_sha256=prepared.verify(replay_root)['runtime_source_sha256'],
        current_main_sha256=after['PPO_project/main.py'],new_training_steps=0,new_independent_replicates=0,
        boundary='Elapsed interpolation time and every physical/geometry/reward/action state are compared exactly; only actor latency, measured environment feature/mapping latency, and evaluation wall time are excluded.')
    (evidence/'comparison.json').write_bytes(prepared.encoded(result))
    script=Path(__file__);(evidence/script.name).write_bytes(script.read_bytes())
    hashes=dict(original_files=after,replay_files={str(path.relative_to(replay_root)):prepared.sha(path.read_bytes())
        for path in [output/'summary.json']+[output/f'{path_id}.{suffix}' for path_id in ids for suffix in ('json','npz')]},
        evidence_files={path.name:prepared.sha(path.read_bytes()) for path in evidence.iterdir() if path.is_file() and path.name!='hash_manifest.json'},
        original_files_unchanged=before==after)
    (evidence/'hash_manifest.json').write_bytes(prepared.encoded(hashes))
    if not (result['original_files_unchanged'] and result['all_non_timing_metrics_equal'] and result['all_non_timing_trace_values_exactly_equal']):
        raise ValueError('Replay differs; original and replay outputs retained for diagnosis')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-root',type=Path,default=prepared.DEFAULT_TARGET)
    parser.add_argument('--run',default='formal/direct_seed11')
    parser.add_argument('--execute',action='store_true',help='Explicitly execute the frozen validation runner; otherwise compare saved replay')
    args=parser.parse_args()
    print(json.dumps(validate(args.replay_root,args.execute,args.run),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
