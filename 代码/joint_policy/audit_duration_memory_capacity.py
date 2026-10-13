"""Locate missing temporal response capacity from saved physical and policy histories."""
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features, OUTPUT_SCALE
from localize_joint_shape_residual import characterize
from duration_joint_metrics import guard_names
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_duration_guarded_memory_learning_20260920'
    source=BASE/'canonical_gate_band_coupled_peak_memory_learning_20260920'
    assert read(stage/'receipt.json')['status'].startswith('completed')
    result=read(stage/'state_joint/results.json')
    baseline=result['candidates'][0]
    new=result['candidates'][302:]
    nearest=min(new,key=lambda c:np.square(np.minimum(c['measured_inequalities'],0.)).sum())
    selected=min(new,key=lambda c:c['score'])
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    reports=[]
    for c in (baseline,nearest,selected):
        md['coefficients']=np.asarray(c['coefficients']).reshape(-1,3)
        replay=collect_data(md,md['parent'],c['records'])
        f=features(md,replay['observations'],replay['innovations'])
        coef=md['coefficients'].reshape(3,3,3)
        memory=np.zeros((2,3,3));previous=np.zeros((3,3));a=np.asarray(md['memory_decays'])[:,None,None]
        rows=[]
        for t,fi in zip(replay['decisions'],f):
            memory=a*memory+(1.-a)*previous;previous=fi.copy()
            response=np.concatenate([fi[None],memory])*coef*OUTPUT_SCALE
            if 260<=t*20<=540:
                rows.append(dict(ms=int(t*20),instantaneous_kernel=fi[:,0].tolist(),
                    per_kernel_correction=response.sum(0).tolist(),total_correction=response.sum((0,1)).tolist()))
        names=['incoming','outgoing','reverse','energy','peak','apex_speed','time_old','cruise_old',
               'first_turn','corridor','lateral']+guard_names()
        reports.append(dict(tag=c['tag'],score=c['score'],coefficients=c['coefficients'],
            failed_guards=[dict(name=n,value=v) for n,v in zip(names,c['measured_inequalities']) if v < -1e-8],
            actual_decoded_actions_bitwise=True,support=rows,record=c['records'][0],
            localization=characterize(c['records'][0]['trace'])))
    write_json(stage/'memory_capacity_audit.json',dict(records=reports,new_physical_rollouts=0,
        trace_bindings=[dict(tag=r['tag'],sha256=sha(r['record']['trace'])) for r in reports],
        interpretation='Exact saved-state/history replay, with no XY reintegration. '
        'Instantaneous support is separated from decaying causal memory. No global '
        'capacity or PPO-estimator conclusion follows from this local failed optimization.'))
    for r in reports:
        print('DURATION_CAPACITY',r['tag'],r['failed_guards'],
              [(s['ms'],s['instantaneous_kernel']) for s in r['support']],flush=True)


if __name__=='__main__':main()
