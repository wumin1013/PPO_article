"""Isolate feedback amplification of the failed shared compensation field."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from train_pilot import evaluate
from learn_single_decision_credit import review
from audit_current_command_coordinates import command_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


class FrozenAdditionalCorrection:
    def __init__(self,base,corrections):
        self.base=base;self.corrections=corrections;self.index=0

    def act(self,observation,deterministic=True):
        action,logp,value=self.base.act(observation,deterministic)
        delta=self.corrections[self.index] if self.index<len(self.corrections) else np.zeros(3)
        self.index+=1
        return action.astype(float)+delta,logp,value


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    failed=BASE/'canonical_gate_band_shared_command_closed_loop_probe_20260918'
    support=BASE/'canonical_gate_band_current_actor_head_support_20260918'
    data=read(parent/'results.json');model=parent/data['chosen']['model']
    candidate=next(c for c in read(failed/'results.json')['candidates'] if c['step']==1.)
    altered=failed/candidate['model'];paths=[6,8,14,15]
    reuse=BASE/'canonical_gate_band_compensation_feedback_cause_20260918'
    stage=register('canonical_gate_band_compensation_feedback_cause_completed_20260918',
        [__file__,model,altered,parent/'results.json',failed/'results.json',failed/'receipt.json',
         support/'support.npz',support/'receipt.json',BASE/'probe_coordinated_suffix.py',BASE/'train_pilot.py',
         reuse/'receipt.json',reuse/'path6/0.npz',reuse/'path6/0.json'],
        dict(paths=paths,new_complete_paths=3,reused_corrected_paths=1,reused_modified_paths=4,wall_s=60,
             actor_calls=2500,physics_ticks=40000,saved_prefix_geometry_ticks=10000),
        'The full output-head compensation passes all frozen command intervals but '
        'fails8/29 closed-loop paths. On two failed corners6/8 and two successful '
        'controls14/15, measure the exact ADDITIONAL actor correction on the actual '
        'failed/successful states. Compare against its values on the saved parent '
        'states. Then intervene only in this additional correction: replay its '
        'parent-state per-decision values, while the original399a policy continues '
        'normal three-action feedback on the real new state. Guide and executor '
        'remain unchanged. After the saved parent decision support, use zero '
        'additional correction and finish normally. Reuse full-feedback variants '
        'and parent paths. This isolates correction-state feedback, not all policy '
        'feedback, and is an offline causal diagnostic only. No frozen sequence '
        'can be deployed or called learned success. Keep failures, complete time '
        'and actual XY/heading/curvature evidence; no candidate adoption.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,
        actor_calls=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False)
    base=CanonicalGatePolicy(256);base.load_state_dict(md['policy'])
    modified=CanonicalGatePolicy(256);modified.load_state_dict(torch.load(altered,map_location='cpu',weights_only=False)['policy'])
    factory=checkpoint_environment_factory(md['architecture'])
    with np.load(support/'support.npz') as z:obs=z['observation'].copy();path_id=z['state_path'].copy()
    geometry=BASE/'canonical_gate_band_current_hold_geometry_20260918'
    with np.load(geometry/'geometry.npz') as z:nominal_latent=z['latent'].copy()
    results=[]
    for path in paths:
        ids=np.flatnonzero(path_id==path);corrections=[];nominal_change=[]
        for index in ids:
            old=base.act(obs[index],True)[0].astype(float);new=modified.act(obs[index],True)[0].astype(float)
            costs['actor_calls']+=2;assert np.array_equal(old,nominal_latent[index])
            corrections.append(new-old);nominal_change.append(90*(np.tanh(new[0])-np.tanh(old[0])))
        corrections=np.asarray(corrections);assert np.max(abs(corrections[:,1:]))==0.
        j=next(j for j,r in enumerate(candidate['results']) if r['original_path']==path)
        with np.load(BASE/candidate['traces'][j]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        # Reconstruct only completed ordinary20ms decisions, not terminal brake rows.
        ordinary=int(np.flatnonzero(rows[:,12])[0]) if np.any(rows[:,12]) else len(rows)
        frame=max(0,((ordinary-1)//20)*20);states=[]
        env=reconstruct(ref,rows,actions,frame,costs,state_snapshots=states,env_factory=factory);states.append(env.state())
        real_changes=[];correction_latents=[]
        for state in states:
            env.restore(state);o=env.observation()
            old=base.act(o,True)[0].astype(float);new=modified.act(o,True)[0].astype(float)
            costs['actor_calls']+=2
            real_changes.append(90*(np.tanh(new[0])-np.tanh(old[0])))
            correction_latents.append(new-old)
        wrapper=FrozenAdditionalCorrection(base,corrections)
        if path==6:
            trace=reuse/'path6/0.npz';new=[read(reuse/'path6/0.json')]
        else:
            new=evaluate(wrapper,[ref],stage,f'path{path}',None,costs,start+60,factory)
            trace=stage/f'path{path}'/'0.npz'
        new[0]['original_path']=path
        summary=review(new,[trace])
        item=dict(path=path,original_full_feedback_result=candidate['results'][j],
            frozen_additional_correction_result=new[0],summary=summary,trace=str(trace.relative_to(BASE)),
            max_saved_state_extra_steering_deg=float(np.max(np.abs(nominal_change))),
            max_actual_state_extra_steering_deg=float(np.max(np.abs(real_changes))),
            actual_extra_over_half_degree_decisions=int((np.abs(real_changes)>.5+1e-6).sum()))
        np.savez_compressed(stage/f'path{path}_extra_feedback.npz',
            saved_extra_latent=corrections,actual_extra_latent=correction_latents,
            saved_extra_steering_deg=nominal_change,actual_extra_steering_deg=real_changes)
        results.append(item)
        write_json(stage/'results.json',dict(parent_sha256=sha(model),altered_model_sha256=sha(altered),
            records=results,costs=costs,diagnostic_only=True,goal_complete=False))
        print('EXTRA_FEEDBACK_CAUSE',path,item['original_full_feedback_result']['success'],
            new[0]['success'],'saved/actual extra',item['max_saved_state_extra_steering_deg'],
            item['max_actual_state_extra_steering_deg'],flush=True)
        assert costs['actor_calls']+costs['evaluation_actor_calls']<=2500 and costs['evaluation_physics_ticks']<=40000
        assert costs['saved_prefix_geometry_ticks']<=10000
    receipt(stage,dict(status='completed_additional_correction_feedback_causal_probe',
        costs=costs,wall_s=time.perf_counter()-start,diagnostic_only=True,goal_complete=False))


if __name__=='__main__':main()
