"""Check publication numbers, model identities and complete recorded study membership."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, write_json
from frozen_policy import FrozenPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=ROOT/'evidence/frozen_main_v6'
    m=json.loads((data/'manifest.json').read_text('utf-8'))
    expected={(c,method,dt) for c in ('original','dense') for method in
        ('JNN','FIXED','S2','S2R','SCMO','COPP','NNC') for dt in (.001,.0005)}
    keys=[(r['case'],r['method'],r['dt_s']) for r in m['runs']]
    assert set(keys)==expected and len(keys)==len(expected)
    for name,digest in m['files'].items():assert sha(data/name)==digest,name
    assert m['model_sha256']=='c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
    assert sha(data/m['main_model'])==m['model_sha256']
    assert sha(ROOT/'shared_manuscript/frozen_main_v6/figure8.pdf')==m['figure8_sha256']
    policies={'JNN':FrozenPolicy(data/m['main_model']),'FIXED':FrozenPolicy(data/'fixed_two.pt')}
    torch.set_num_threads(1)
    rows=[]
    for run in m['runs']:
        metrics=run['metrics'];file=data/run['trace'];assert sha(file)==run['trace_sha256']
        assert metrics['limits']==[100.,2000.,62500.]
        with np.load(file) as z:
            p=z['positions'];reference=z['reference'];dt=run['dt_s']
            assert np.array_equal(reference,np.loadtxt(data/'programs'/run['case']/'reference.csv',delimiter=',',skiprows=1))
            assert np.array_equal(p[0],reference[0])
            v=np.diff(p,axis=0)/dt;a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
            j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
            peaks=[np.linalg.norm(x,axis=1).max(initial=0) for x in (v,a,j)]
            assert np.allclose(peaks,metrics['vector_peaks'],rtol=1e-8,atol=2e-4)
            assert abs((len(p)-1)*dt-metrics['elapsed_prefix_s'])<1e-8
            if metrics['success']:
                assert metrics['kinematic_pass'] and metrics['rest_pass']
                assert metrics['geometry']['geometry_complete'] and metrics['geometry']['corridor_inside']
                assert abs((len(p)-1)*dt-metrics['complete_time_s'])<1e-8
            else:assert metrics['complete_time_s'] is None
            action_error=None
            if run['method'] in policies:
                policy=policies[run['method']];obs=z['observations']
                actual=np.concatenate([policy.mean(batch).numpy() for batch in np.array_split(obs,max(1,len(obs)//500))])
                if run['method']=='JNN':
                    action_error=float(np.max(np.abs(actual-z['actions'])))
                else:
                    physical=np.concatenate([policy.decoded(batch) for batch in np.array_split(obs,max(1,len(obs)//500))])
                    action_error=float(np.max(np.abs(physical-z['actions'])))
                assert action_error<2e-4,(run['id'],action_error)
        rows.append(dict(id=run['id'],complete=metrics['success'],time_s=metrics['complete_time_s'],
            command_intervals=len(p)-1,action_error=action_error,trace_sha256=run['trace_sha256']))
    generalization=json.loads((ROOT/'audit/frozen_main_generalization_v6_report.json').read_text('utf-8'))
    assert generalization['complete_grid'] and len(generalization['rows'])==10
    for row in generalization['rows']:
        assert sha(ROOT/row['source_trace'])==row['source_trace_sha256']
        assert sha(ROOT/row['source_record'])==row['source_record_sha256']
        assert row['success']==row['independent_audit']['success']
    result=dict(status='scoped_evidence_verified',main_model_sha256=m['model_sha256'],
        check_scope='Fresh position-difference and action checks; geometry and terminal acceptance are hash-bound to the retained independent complete-interval audits.',
        manifest_sha256=sha(data/'manifest.json'),main_comparison_cells=len(rows),
        main_completed=sum(r['complete'] for r in rows),main_rows=rows,
        additional_cells=len(generalization['rows']),additional_completed=generalization['successful_cells'],
        additional_by_resolution=generalization['successful_by_resolution'],
        scope='One frozen controller; two development programs and all five registered synthetic transfer programs. No independent-seed population claim.',
        figure8_unchanged=True,five_seed_training_cancelled=True,
        publication_decision='Scientific numbers checked for this stated scope; author and journal-format decisions are separate.',
        retained_gaps=['Historical intermediate training inputs not fully recovered; final model, final-fit input and reported trajectories retained.',
            'No hardware motion, HIL or cutting evidence.',
            'JMP current guide unavailable (HTTP 403); JIM separate data statement placement needs submission-format resolution.'])
    write_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('status','main_comparison_cells','main_completed','additional_cells','additional_completed')}))


if __name__=='__main__':main()
