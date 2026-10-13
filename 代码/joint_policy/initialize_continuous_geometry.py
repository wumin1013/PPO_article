"""Training-state-only distillation to isolate a continuous geometry encoding."""
import time
import numpy as np
import torch
from continuous_tangent_policy import ContinuousTangentPolicy, interval_geometry
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from learn_single_decision_credit import SCREEN, review
from joint_runtime_v5 import environment_factory
from reflection_policy import reflect_observation
from search_complete_return import read
from study_io import BASE, register, receipt, sha, write_json
from train_pilot import evaluate, save_torch


def main():
    torch.set_num_threads(1);torch.manual_seed(917041)
    parent=BASE/'exit_local_credit_parameter_learning_20260916'
    support=BASE/'matched_entry_interaction_feedback_20260916/window_training_data.npz'
    event=BASE/'preswing_sampled_geometry_event_audit_20260917'
    stage=register('continuous_interval_geometry_initialization_20260917',
        [__file__,BASE/'continuous_tangent_policy.py',parent/'candidate_1.pt',parent/'receipt.json',
         support,event/'results.json',BASE/'preswing_sampled_geometry_causal_probe_20260917/receipt.json',
         BASE/'local_geometry_feedback_policy.py',BASE/'frame_feedback_policy.py',
         BASE/'reflection_policy.py',BASE/'learn_single_decision_credit.py',BASE/'train_pilot.py',
         BASE/'learning_preparation_v4/paths.json'],
        dict(saved_training_states=4177,new_probe_decisions=0,optimizer_steps=1000,
             checkpoints=[100,300,1000],learning_rate=.001,seed=917041,
             teacher_latent_scales=[.02,.2,.2],evaluation_actor_calls=10000,
             evaluation_physics_ticks=240000,wall_s=240),
        'Mechanism-directed representation initialization, not final task learning. Replace '
        'discontinuous point tangent observations with continuous interval mean displacement '
        'derived from unchanged reference point samples. Keep same72 input slots,96x96 actor, '
        'reflection/heading encoding, other inputs and physical executor for this bounded '
        'representation study; no dimensional-minimality or full-Markov sufficiency claim. '
        'The last chord is repeated, never normalized into a discontinuous direction. '
        'Distill only retained actor means on4177 saved training states with path-balanced '
        'normalized squared loss. No new target shape, reward or hand turn rule enters. '
        'Check reflection and disappearance of the identified geometric feature jumps. '
        'Evaluate three fixed checkpoints on21 training paths; choose at most one64-path '
        'initialization if21/21 success, each entry maximum no worse than parent+1e-6mm, '
        'exit maximum/full reversal sum no worse, full time<=102% parent, and improved '
        'independent complete return. No development access in this initialization study. '
        'If none qualifies, stop and analyze; fitting error is not task success. The encoding '
        'does not remove gate-progress or endpoint-validity events. No probe schedule is deployed.')
    started=time.perf_counter();costs=dict(optimizer_steps=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    state=torch.load(parent/'candidate_1.pt',map_location='cpu',weights_only=False)['policy']
    teacher=LocalGeometryFeedbackPolicy(mask_endpoint=True);teacher.load_state_dict(state)
    policy=ContinuousTangentPolicy(mask_endpoint=True);policy.load_state_dict(state)
    with np.load(support) as z:
        x=torch.tensor(z['observation'][:4177]);w=torch.tensor(z['weight'][:4177]*3,dtype=torch.float32)
    assert abs(float(w.sum())-1.)<1e-6
    assert torch.allclose(interval_geometry(reflect_observation(x)),reflect_observation(interval_geometry(x)),atol=1e-7,rtol=0.)
    with torch.no_grad():target=teacher.distribution(x).mean
    scale=torch.tensor([.02,.2,.2])
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=.001)
    examples=read(event/'results.json')['geometric_limit_probes']
    before=torch.tensor([r['before']['observation'] for r in examples],dtype=torch.float32)
    after=torch.tensor([r['after']['observation'] for r in examples],dtype=torch.float32)
    with torch.no_grad():teacher_jumps=(teacher.distribution(after).mean-teacher.distribution(before).mean).numpy()
    snapshots=[]
    for step in range(1,1001):
        assert time.perf_counter()-started<240
        predicted=policy.distribution(x).mean
        loss=((((predicted-target)/scale)**2).mean(1)*w).sum()
        optimizer.zero_grad();loss.backward();optimizer.step();costs['optimizer_steps']+=1
        if step not in (100,300,1000):continue
        with torch.no_grad():
            error=(policy.distribution(x).mean-target)/scale
            rms=torch.sqrt((error.square()*w[:,None]).sum(0))
            jumps=(policy.distribution(after).mean-policy.distribution(before).mean).numpy()
        name=f'policy_{step}.pt'
        save_torch(stage/name,dict(policy=policy.state_dict(),architecture='continuous_interval_tangent',
            mask_endpoint=True,observation_dim=72,frame='heading',history_independent=True,
            parent_sha256=sha(parent/'candidate_1.pt'),initialization='training_state_teacher_distillation',steps=step))
        row=dict(step=step,model=name,normalized_rms_by_head=rms.tolist(),
            max_normalized_error=float(error.abs().max()),latent_event_jumps=jumps.tolist(),
            max_abs_latent_event_jump=float(abs(jumps).max()))
        snapshots.append(row)
        write_json(stage/'fit.json',dict(checkpoints=snapshots,teacher_event_jumps=teacher_jumps.tolist(),costs=costs))
        print('CONTINUOUS_FIT',step,rms.tolist(),'event_jump',row['max_abs_latent_event_jump'],flush=True)
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(p) for p in paths['training']]
    base_results=[dict(r,original_path=i) for i,r in enumerate(read(parent/'training_1/summary.json')['results'])]
    base_traces=[parent/'training_1'/f'{i}.npz' for i in range(64)]
    baseline=review([base_results[i] for i in SCREEN],[base_traces[i] for i in SCREEN])
    candidates=[];evaluations={}
    for snapshot in snapshots:
        policy.load_state_dict(torch.load(stage/snapshot['model'],map_location='cpu',weights_only=False)['policy'])
        tag=f'screen_{snapshot["step"]}'
        results=evaluate(policy,[training[i] for i in SCREEN],stage,tag,None,costs,started+240,environment_factory)
        results=[dict(r,original_path=i) for r,i in zip(results,SCREEN)]
        traces=[stage/tag/f'{i}.npz' for i in range(len(SCREEN))]
        value=dict(review(results,traces),model=snapshot['model'])
        good=value['comparable_complete_pool']
        old={s['path']:s for s in baseline['shapes']}
        value['eligible']=bool(good and all(s['in_edge_outward_mm']<=old[s['path']]['in_edge_outward_mm']+1e-6 for s in value['shapes'])
            and value['max_exit_mm']<=baseline['max_exit_mm']+1e-6
            and value['full_reverse_sum_deg']<=baseline['full_reverse_sum_deg']+1e-6
            and value['full_time_s']<=1.02*baseline['full_time_s']
            and value['independent_zero_return']>baseline['independent_zero_return'])
        candidates.append(value);evaluations[snapshot['model']]=(results,traces)
        write_json(stage/'screen_results.json',dict(baseline=baseline,candidates=candidates,costs=costs))
        print('CONTINUOUS_SCREEN',snapshot['step'],value['successes'],'eligible',value['eligible'],flush=True)
    valid=[r for r in candidates if r['eligible']];selected=None
    if valid:
        selected=max(valid,key=lambda r:r['independent_zero_return'])
        write_json(stage/'training_selection.json',dict(selected=selected,training_only=True))
        policy.load_state_dict(torch.load(stage/selected['model'],map_location='cpu',weights_only=False)['policy'])
        remaining=[i for i in range(64) if i not in SCREEN]
        results=evaluate(policy,[training[i] for i in remaining],stage,'remaining',None,costs,started+240,environment_factory)
        results=[dict(r,original_path=i) for r,i in zip(results,remaining)]
        traces=[stage/'remaining'/f'{i}.npz' for i in range(len(remaining))]
        old_results,old_traces=evaluations[selected['model']]
        selected=dict(selected,full_training=review(old_results+results,old_traces+traces))
    assert costs['evaluation_actor_calls']<=10000 and costs['evaluation_physics_ticks']<=240000
    write_json(stage/'results.json',dict(candidates=candidates,selected=selected,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_continuous_geometry_initialization',costs=costs,
        screen_eligible=len(valid),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
