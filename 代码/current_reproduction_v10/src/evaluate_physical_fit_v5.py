"""Fixed nine-run development pilot matrix; long path is unconditional."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import time
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from strong_policy_v4 import load_strong_policy
from replay_strong_checkpoint_v4 import paths
from runtime_physical_time_v5 import install,VERSION
from evaluate import rollout

MODEL=ROOT/'results/physical_fit_v5/pilot_seed101/adaptive/final.pt'
OUT=MODEL.parent/'evaluation_v5'


def register():
    path=OUT/'registration.json'
    if path.exists():return json.loads(path.read_text('utf-8'))
    receipt=json.loads((MODEL.parent/'receipt.json').read_text('utf-8'))
    assert receipt['status']=='completed' and receipt['offline_iterations']==24000
    all_cases=paths()
    runs=[dict(case=row['id'],dt_s=.001,wall_s=240.) for row in all_cases if row['historical_split'] in ('train','validation')]
    runs.extend([dict(case='dense',dt_s=.001,wall_s=240.),dict(case='dense',dt_s=.0005,wall_s=300.),
                 dict(case='original',dt_s=.001,wall_s=1800.)])
    assert len(runs)==9
    names=['evaluate_physical_fit_v5.py','strong_policy_v4.py','runtime_physical_time_v5.py',
        'runtime_resolution_fixed_v4e.py','terminal_completion_v4d.py','runtime_no_target_override_v4b.py',
        'replay_no_target_override_v4b.py','runtime_bootstrap.py','runtime_configuration.py',
        'corridor_v3.py','execution_settings.py','evaluate.py','metrics.py','replay_strong_checkpoint_v4.py']
    reg=dict(created_utc=utc(),scope='Development pilot only',runtime_version=VERSION,
        identity='Fresh seed101 physical-state supervised24000 adaptive actor; not historical c16d45 main',
        seed=101,arm='adaptive',model_sha256=sha(MODEL),model_updated=False,
        fitting_receipt_sha256=sha(MODEL.parent/'receipt.json'),cases=all_cases,runs=runs,
        selection='Fixed final24000 actor on every registered run',
        long_rule='Always run original1ms regardless of all short/dense outcomes; full common path and1800s wall budget',
        retention='All successes, failed prefixes, exceptions and wall timeouts retained; failed prefixes have no completion time',
        source_sha256={name:sha(ROOT/'src'/name) for name in names},runtime_manifest_sha256=sha(ROOT/'runtime/source_manifest.json'))
    write_json(path,reg);return reg


def main():
    torch.set_num_threads(1)
    reg=register();results=[];started=time.perf_counter()
    for run in reg['runs']:
        name=run['case']+f"_dt{round(run['dt_s']*1e6)}us"
        output=OUT/(name+'.npz')
        if output.with_suffix('.json').exists():
            result=json.loads(output.with_suffix('.json').read_text('utf-8'))
            assert result['model_sha256']==reg['model_sha256']
            results.append(result);continue
        assert sha(MODEL)==reg['model_sha256']
        assert all(sha(ROOT/'src'/n)==v for n,v in reg['source_sha256'].items())
        row=next(row for row in reg['cases'] if row['id']==run['case'])
        assert sha(ROOT/row['reference'])==row['reference_sha256']
        setup=install(dt=run['dt_s'],tolerance=row['evaluation_tolerance_mm'])
        policy,_=load_strong_policy(MODEL)
        reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
        result=rollout(policy,reference,output,dt=run['dt_s'],tolerance=row['evaluation_tolerance_mm'],
            wall_s=run['wall_s'],runtime_version='v3')
        assert sha(MODEL)==reg['model_sha256']
        result.update(path=run['case'],scope='Development pilot only',method='fresh_seed101_physical_state_supervised24000_adaptive',
            seed=101,arm='adaptive',runtime_version=VERSION,executor_setup=setup,
            model_sha256=reg['model_sha256'],model_updated=False,
            reference_sha256=row['reference_sha256'],nominal_tolerance_mm=row['nominal_tolerance_mm'],
            arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],registration_sha256=sha(OUT/'registration.json'),
            action_source='Fresh seed101 adaptive actor final24000; common finite terminal brake when triggered')
        write_json(output.with_suffix('.json'),result);results.append(result)
        print('PHYSICAL_PILOT',run['case'],run['dt_s'],result['success'],result.get('complete_time_s'),result.get('elapsed_prefix_s'),result['reason'],flush=True)
    report=dict(created_utc=utc(),status='all_registered_runs_recorded',runs=len(results),
        successful_runs=sum(r['success'] for r in results),model_sha256=reg['model_sha256'],
        total_matrix_wall_s=time.perf_counter()-started,
        results=[dict(path=r['path'],dt_s=r['dt_s'],success=r['success'],complete_time_s=r['complete_time_s'],
            elapsed_prefix_s=r['elapsed_prefix_s'],reason=r['reason'],error=r['error']) for r in results],
        no_five_seed_expansion=True,scope='Same single adaptive development pilot; no independent final test claim')
    write_json(OUT/'summary.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
