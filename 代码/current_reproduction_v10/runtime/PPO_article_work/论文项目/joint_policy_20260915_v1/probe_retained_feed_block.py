"""Single-factor offline parameter intervention to locate measured cruise loss."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from learn_single_decision_credit import review
from learn_canonical_credit_step import compare
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    restored=BASE/'canonical_gate_band_measured_segment_restoration_20260917'
    retained=BASE/'canonical_gate_band_shape_cruise_statewise_half_full_20260917'
    data=read(restored/'results.json');names=data['metric_names']
    stage=register('canonical_gate_band_retained_feed_intervention_20260917',
        [__file__,restored/'candidate.pt',restored/'results.json',restored/'receipt.json',
         retained/'candidate_0.5.pt',retained/'results.json',retained/'receipt.json',
         BASE/'canonical_gate_band_joint_regret_round_20260917/complete_cruise_comparison.json',
         BASE/'kernel_local_feedback_policy.py',BASE/'canonical_gate_state.py',BASE/'train_pilot.py'],
        dict(new_parameter_interventions=1,training_paths=64,evaluation_actor_calls=6000,
             evaluation_physics_ticks=160000,wall_s=180),
        'All20ms of restored-policy cruise loss relative to retained4067 occur on pure straights. '
        'Only replace the learned residual feed coefficient block by the retained feed block; '
        'keep steering, preview, base actor and runtime bitwise unchanged. Evaluate all64 training paths '
        'with complete ordered coverage, V/A/J, corridor, braking and rest checks. On straights '
        'compare exact XY/velocity histories with retained4067, not just threshold durations. '
        'Report interactions on curved paths separately. This is a finite offline causal intervention, '
        'not a newly trained policy, not adopted, and cannot establish final joint-learning success. '
        'Do not optimize the replacement, repeat its variants, inflate margins, change constraints or access sealed tests.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    base=torch.load(restored/'candidate.pt',map_location='cpu',weights_only=False)
    other=torch.load(retained/'candidate_0.5.pt',map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(base['policy'])
    with torch.no_grad():p.residual_coefficients[1].copy_(other['policy']['residual_coefficients'][1])
    for key,value in p.state_dict().items():
        if key!='residual_coefficients':assert torch.equal(value,base['policy'][key])
    assert torch.equal(p.residual_coefficients[[0,2]],base['policy']['residual_coefficients'][[0,2]])
    save_torch(stage/'diagnostic_only.pt',dict(base,policy=p.state_dict(),diagnostic_only=True,
        intervention='retained_feed_parameter_block',not_a_newly_trained_policy=True,
        source_model_sha256=sha(restored/'candidate.pt'),feed_model_sha256=sha(retained/'candidate_0.5.pt')))
    paths=[np.asarray(r) for r in read(BASE/'learning_preparation_v4/paths.json')['training']]
    rs=evaluate(p,paths,stage,'full',None,costs,start+180,checkpoint_environment_factory(base['architecture']))
    rs=[dict(r,original_path=i) for i,r in enumerate(rs)];traces=[stage/'full'/f'{i}.npz' for i in range(64)]
    summary=review(rs,traces);full=dict(summary=summary,metrics=metric_rows(rs,traces,summary,names),results=rs,
                                       traces=[str(t.relative_to(BASE)) for t in traces])
    full['checks_vs_restored']=compare(full,data['full_candidate'],names,required_reduction=0.,energy_scope='aggregate')
    full['checks_vs_anchor']=compare(full,data['full_parent'],names,required_energy_reduction=.01,energy_scope='aggregate')
    old=read(retained/'results.json')['full_candidate'];identities=[]
    for i,path in enumerate(paths):
        if len(path)!=2:continue
        j=next(j for j,r in enumerate(old['results']) if r['original_path']==i)
        with np.load(BASE/old['traces'][j]) as z:a=z['rows']
        with np.load(traces[i]) as z:b=z['rows']
        identities.append(dict(path=i,same_length=len(a)==len(b),
            exact_xy_velocity_acceleration=bool(a.shape==b.shape and np.array_equal(a[:,:6],b[:,:6])),
            maximum_xy_velocity_acceleration_error=float(abs(a[:,:6]-b[:,:6]).max()) if a.shape==b.shape else None))
    assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=160000
    write_json(stage/'results.json',dict(full_intervention=full,straight_identities=identities,
        replaced_head_only=True,not_a_newly_trained_policy=True,goal_complete=False))
    receipt(stage,dict(status='completed_retained_feed_causal_intervention',successes=summary['successes'],
        full_time_s=summary['full_time_s'],cruise99_s=summary['complete_cruise_s']['99'],
        straight_physical_identity=all(r['exact_xy_velocity_acceleration'] for r in identities),
        costs=costs,wall_s=time.perf_counter()-start,diagnostic_only=True,goal_complete=False))
    print('FEED_INTERVENTION',summary['successes'],summary['full_time_s'],summary['complete_cruise_s']['99'],
          full['checks_vs_anchor']['eligible'],full['checks_vs_anchor']['regressions'],identities,flush=True)


if __name__=='__main__':main()
