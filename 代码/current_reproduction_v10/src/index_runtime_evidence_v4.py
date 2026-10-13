"""Index every prescribed runtime-subtask record without success filtering."""
import json
from common import ROOT,sha,write_json,utc

def read(path):return json.loads(path.read_text('utf-8'))

registrations={
    'historical_compatibility':ROOT/'results/strong_checkpoint_v4/registration.json',
    'no_target_override':ROOT/'results/v4b_no_target_override/registration.json',
    'fresh_fit_v4b':ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4b/registration.json',
    'fresh_fit_v4d':ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d/registration.json',
    'fresh_fit_v4e':ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4e/registration.json',
    'historical_half_ms_v4e':ROOT/'results/strong_checkpoint_v4/v4e_resolution_control/registration.json',
    'ppo_smoke_v4e':ROOT/'results/strong_ppo_v4/smoke_seed101/adaptive/evaluation_v4e/registration.json',
    'stage2_fit_v4e':ROOT/'results/final_function_fit_v4/stage2_seed101/adaptive/evaluation_v4e/registration.json',
}
expected=[]
for key,registration in registrations.items():
    reg=read(registration);folder=registration.parent
    if key=='historical_compatibility':
        expected.extend((key,folder/'v3'/(r['id']+'.json')) for r in reg['cases'])
        expected.extend((key,folder/'v1'/(name+'.json')) for name in ('original','dense'))
    elif key=='no_target_override':
        expected.extend((key,folder/'strong'/(r['id']+'.json')) for r in reg['cases'])
        expected.extend((key,folder/'teacher'/(r['id']+'.json')) for r in reg['cases'] if r['historical_split'] in ('train','validation'))
    elif key=='fresh_fit_v4b':expected.extend((key,folder/(r['id']+'.json')) for r in reg['cases'])
    elif key=='fresh_fit_v4d':
        expected.extend((key,folder/(r['id']+f'_dt{dt}us.json')) for r in reg['cases'] if r['id'].startswith('final_') for dt in (1000,500))
        expected.extend([(key,folder/'original_dt1000us.json'),(key,folder/'dense_dt500us.json')])
    elif key=='fresh_fit_v4e':
        expected.extend((key,folder/(r['id']+'_dt500us.json')) for r in reg['cases'] if r['id'].startswith('final_') or r['id']=='dense')
        expected.append((key,folder/'train_sine_dt1000us.json'))
    elif key=='historical_half_ms_v4e':expected.extend((key,folder/(name+'_dt500us.json')) for name in ('dense','original'))
    elif key=='ppo_smoke_v4e':expected.extend((key,folder/(r['id']+'_dt1000us.json')) for r in reg['cases'])
    elif key=='stage2_fit_v4e':
        expected.extend((key,folder/(r['id']+'_dt1000us.json')) for r in reg['cases'])
        expected.append((key,folder/'dense_dt500us.json'))
records=[]
for group,file in expected:
    row=dict(group=group,result_file=file.relative_to(ROOT).as_posix(),exists=file.exists())
    if file.exists():
        result=read(file);trace=file.with_suffix('.npz');digest=sha(trace)
        if digest!=result['trace_sha256']:raise ValueError('Recorded trajectory digest differs: '+str(file))
        row.update(success=result['success'],complete_time_s=result['complete_time_s'],elapsed_prefix_s=result['elapsed_prefix_s'],
            dt_s=result['dt_s'],path=result.get('path'),method=result.get('method'),runtime_version=result['runtime_version'],
            model_sha256=result.get('model_sha256'),result_sha256=sha(file),trace_file=trace.relative_to(ROOT).as_posix(),trace_sha256=digest)
    records.append(row)
summary=dict(created_utc=utc(),scope='Runtime development/replay subtask only; no final-test claims.',
    expected_runs=len(expected),completed_records=sum(r['exists'] for r in records),
    successful_records=sum(r.get('success',False) for r in records),
    failed_records=sum(r['exists'] and not r['success'] for r in records),
    registrations={k:dict(file=v.relative_to(ROOT).as_posix(),sha256=sha(v)) for k,v in registrations.items()},
    records=records,
    omissions='Fresh-fit original 0.5 ms was not launched because its preregistered dense success condition failed; raw failed dense remains. Parent-agent terminal/guide diagnostics and training receipts are separate evidence.',
    interpretation='Counts describe complete attempted runs, not independent seed replicates. Success requires the stored independent full trajectory audit; prefix duration is not completion time.')
write_json(ROOT/'audit/runtime_evidence_index_v4.json',summary)
print(json.dumps({key:summary[key] for key in ('expected_runs','completed_records','successful_records','failed_records')},indent=2))
