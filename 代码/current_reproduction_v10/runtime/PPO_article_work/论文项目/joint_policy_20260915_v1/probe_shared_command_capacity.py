"""Whole-feedback test of an algebraically feasible saved-command fit."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from audit_current_command_coordinates import command_signals
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    source=BASE/'canonical_gate_band_actor_head_command_capacity_20260918'
    prepared=BASE/'canonical_gate_band_current_actor_head_support_20260918'
    data=read(parent/'results.json');selected=data['chosen'];model=parent/selected['model']
    fit=read(source/'results.json');assert fit['parent_sha256']==sha(model)
    assert next(r for r in fit['arms'] if r['arm']=='half_degree_support')['verified_feasible']
    stage=register('canonical_gate_band_shared_command_closed_loop_probe_20260918',
        [__file__,model,parent/'results.json',parent/'receipt.json',source/'results.json',source/'receipt.json',
         source/'half_degree_support.npz',prepared/'support.npz',BASE/'train_pilot.py',
         BASE/'canonical_gate_state.py',BASE/'audit_current_command_coordinates.py'],
        dict(scales=[.25,1.],screen_paths=[r['original_path'] for r in selected['results']],
             actor_decisions=6000,physics_ticks=140000,wall_s=120,automatic_adoption=False),
        'The kernel-only half-degree fit is infeasible. Releasing the existing96 '
        'steering output weights makes the same frozen-path intervals feasible, '
        'but needs large coefficient changes. Execute that bounded-support fit '
        'at predeclared0.25/1 scales on all29 screen paths with the original '
        'full state feedback and frozen executor. Feed/preview PARAMETERS remain '
        'unchanged, while their policy outputs respond normally to altered states. '
        'Keep every failure and raw1ms trace; incomplete prefixes have no time '
        'benefit. Measure actual combined/filtered/physical reverse headings, '
        'XY entry/exit, curvature, cruise, time and constraints. These are '
        'diagnostic policy interventions, not accepted learned candidates. No '
        'runtime angle clamp, look-up trajectory, larger scale or test access.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);factory=checkpoint_environment_factory(md['architecture'])
    with np.load(source/'half_degree_support.npz') as z:
        direction=z['direction'].copy();output=z['actor_output_delta'].copy()
    with np.load(prepared/'support.npz') as z:output_scale=float(z['extra_weight_scale'])
    ids=[r['original_path'] for r in selected['results']]
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];candidates=[]
    for scale in (.25,1.):
        p.load_state_dict(md['policy'])
        with torch.no_grad():
            p.residual_coefficients.add_(torch.tensor(scale*direction,dtype=torch.float32))
            p.actor[-1].weight[0].add_(torch.tensor(scale*output_scale*output,dtype=torch.float32))
        name=f'scale{scale:g}.pt';tag=f'scale{scale:g}'
        save_torch(stage/name,dict(md,policy=p.state_dict(),diagnostic_only=True,
            parent_sha256=sha(model),learning_method='saved_command_capacity_intervention'))
        results=evaluate(p,[np.asarray(paths[i]) for i in ids],stage,tag,None,costs,start+120,factory)
        results=[dict(r,original_path=i) for r,i in zip(results,ids)]
        traces=[stage/tag/f'{j}.npz' for j in range(len(ids))]
        summary=review(results,traces);commands=[]
        for r,t in zip(results,traces):
            if not r['success'] or r['original_path']>=16:continue
            with np.load(t) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            signal=command_signals(ref,rows,actions)
            apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
            pre=(np.arange(len(rows))<apex)&(rows[:,12]==0)&(np.linalg.norm(rows[:,2:4],axis=1)>5.)
            commands.append(dict(path=r['original_path'],
                combined_negative_target_ms=int((pre&(signal['combined_deg']<-.05)).sum()),
                physical_negative_heading_ms=int((pre&(signal['physical_heading_deg']<-.05)).sum())))
        item=dict(step=scale,model=name,summary=summary,results=results,
            traces=[str(t.relative_to(BASE)) for t in traces],commands=commands,
            metrics=metric_rows(results,traces,summary,data['metric_names']))
        candidates.append(item)
        write_json(stage/'results.json',dict(parent_sha256=sha(model),baseline=selected,
            candidates=candidates,metric_names=data['metric_names'],costs=costs,diagnostic_only=True,goal_complete=False))
        print('COMMAND_CAPACITY_PHYSICS',scale,{k:summary[k] for k in
            ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s')},
            'negative_ms',sum(r['physical_negative_heading_ms'] for r in commands),flush=True)
        assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=140000
    receipt(stage,dict(status='completed_shared_command_capacity_physical_probe',costs=costs,
        wall_s=time.perf_counter()-start,diagnostic_only=True,learned_candidate_adopted=False,goal_complete=False))


if __name__=='__main__':main()
