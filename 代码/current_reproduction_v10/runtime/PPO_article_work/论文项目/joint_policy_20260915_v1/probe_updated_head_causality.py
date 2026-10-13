"""Factorial ablation of a failed learned update, with exact endpoint reuse."""
import copy
import itertools
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from learn_exit_local_credit import summary
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def compose(parent,changed,bits):
    state=copy.deepcopy(parent)
    for head,bit in enumerate(bits):
        if bit:
            for key in ('actor.4.weight','actor.4.bias'):state[key][head]=changed[key][head].clone()
    return state


def main():
    torch.set_num_threads(1)
    a=BASE/'exit_local_credit_parameter_learning_20260916';b=BASE/'matched_entry_interaction_feedback_20260916'
    am=a/'candidate_1.pt';bm=b/'window_0.5.pt'
    indices=[0,1,7,12,14,15,16,28,31,40,41]
    stage=register('updated_head_causality_probe_20260916',
        [__file__,am,bm,a/'receipt.json',b/'receipt.json',b/'failure_action_audit.json',
         BASE/'local_geometry_feedback_policy.py',BASE/'learn_exit_local_credit.py',
         BASE/'learning_preparation_v4/paths.json'],
        dict(new_head_combinations=6,screening_paths=indices,full_training_candidates_at_most=2,
             development_candidates_at_most=1,new_actor_calls=20000,new_physics_ticks=450000,wall_s=300),
        'Exact3-output-head parameter ablation of the failed half-step feedback fit. Parent000 and '
        'changed111 are already physically evaluated on all64 training paths and are reused. Hidden '
        'network, observation, guider and executor are identical. Six remaining combinations first '
        'evaluate all six previous failure paths plus five representative corners. Only complete '
        'screen candidates with entry<=0.05mm, no larger maximum exit or full reversal sum, and '
        'lower time+two-edge cost than000 may enter remaining53 training paths, at most best two; '
        'reuse their already evaluated11 paths. A full64 candidate needs median exit<=80% parent '
        'with all these other criteria before at most one development evaluation. Failed-prefix '
        'shape aggregates remain null. All outputs are learned state-feedback functions; no new '
        'trajectory rule, live search, teacher schedule or sealed test. This is a causal component '
        'ablation/model-selection experiment, not independent seed evidence.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    parent=torch.load(am,map_location='cpu',weights_only=False)['policy']
    changed=torch.load(bm,map_location='cpu',weights_only=False)['policy']
    for k in parent:
        if k not in ('actor.4.weight','actor.4.bias'):assert torch.equal(parent[k],changed[k]),k
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(p) for p in paths['training']]
    model=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    with np.load(b/'window_training_data.npz') as z:obs=torch.tensor(z['observation'])
    with torch.no_grad():
        model.load_state_dict(parent);ma=model.distribution(obs).mean
        model.load_state_dict(changed);mb=model.distribution(obs).mean
    rows=[];evaluations={}
    for bits,folder in [((0,0,0),a/'training_1'),((1,1,1),b/'window_training_0.5')]:
        original=read(folder/'summary.json')['results']
        results=[dict(original[i],original_path=i) for i in indices];traces=[folder/f'{i}.npz' for i in indices]
        row=summary(folder,results,traces);row.update(bits=list(bits),reused=True,
            path_success=[dict(path=i,success=r['success']) for i,r in zip(indices,results)])
        rows.append(row);evaluations[bits]=(results,traces)
    baseline=rows[0]
    def screen(row,base,median=False):
        return bool(row['successes']==(64 if median else len(indices)) and
            row['max_entry_mm']<=.05 and row['max_exit_mm']<=base['max_exit_mm'] and
            row['full_reverse_sum_deg']<=base['full_reverse_sum_deg'] and
            row['time_plus_shape_s']<base['time_plus_shape_s'] and
            (not median or row['median_exit_mm']<=.8*base['median_exit_mm']))
    for bits in itertools.product((0,1),repeat=3):
        if bits in evaluations:continue
        model.load_state_dict(compose(parent,changed,bits))
        with torch.no_grad():
            expected=torch.stack([mb[:,h] if bits[h] else ma[:,h] for h in range(3)],-1)
            error=float((model.distribution(obs).mean-expected).abs().max());assert error<1e-6
        name='candidate_'+''.join(map(str,bits))+'.pt'
        save_torch(stage/name,dict(policy=model.state_dict(),observation_dim=72,architecture='local_geometry_feedback',
            mask_endpoint=True,frame='heading',history_independent=True,learning_method='learned_output_head_causal_ablation',
            bits=list(bits),parents=[sha(am),sha(bm)]))
        tag='screen_'+''.join(map(str,bits))
        results=evaluate(model,[training[i] for i in indices],stage,tag,None,costs,started+300,environment_factory)
        for row,i in zip(results,indices):row['original_path']=i
        traces=[stage/tag/f'{i}.npz' for i in range(len(indices))];evaluations[bits]=(results,traces)
        row=summary(stage/tag,results,traces);row.update(bits=list(bits),model=name,reused=False,
            head_identity_error=error,path_success=[dict(path=i,success=r['success']) for i,r in zip(indices,results)])
        row['eligible_screen']=screen(row,baseline);rows.append(row)
        write_json(stage/'screening.json',dict(records=rows,costs=costs))
        print('HEAD_CAUSE',bits,'success',row['successes'],'failures',[r['path'] for r in row['path_success'] if not r['success']],
              'medianexit',row['median_exit_mm'],'eligible',row['eligible_screen'],flush=True)
    eligible=sorted([r for r in rows if r.get('eligible_screen')],key=lambda r:r['time_plus_shape_s'])[:2]
    write_json(stage/'screening_selection.json',dict(selected=[r['bits'] for r in eligible],training_only=True))
    base_full=summary(a/'training_1',read(a/'training_1/summary.json')['results']);full=[]
    for row in eligible:
        bits=tuple(row['bits']);model.load_state_dict(compose(parent,changed,bits))
        remaining=[i for i in range(64) if i not in indices];tag='remaining_'+''.join(map(str,bits))
        results=evaluate(model,[training[i] for i in remaining],stage,tag,None,costs,started+300,environment_factory)
        for result,i in zip(results,remaining):result['original_path']=i
        mapping={i:(r,t) for i,r,t in zip(indices,*evaluations[bits])}
        mapping.update({i:(r,stage/tag/f'{j}.npz') for j,(i,r) in enumerate(zip(remaining,results))})
        ordered=[mapping[i][0] for i in range(64)];traces=[mapping[i][1] for i in range(64)]
        merged=summary(stage,ordered,traces);merged.update(bits=list(bits),model=row['model'],
            reused_paths=indices,new_paths=remaining,trace_paths=[p.relative_to(BASE).as_posix() for p in traces])
        merged['eligible_full']=screen(merged,base_full,median=True);full.append(merged)
        write_json(stage/'full_training_results.json',dict(baseline=base_full,candidates=full,costs=costs))
        print('HEAD_FULL',bits,merged['successes'],'medianexit',merged['median_exit_mm'],'eligible',merged['eligible_full'],flush=True)
    candidates=[r for r in full if r['eligible_full']]
    selected=min(candidates,key=lambda r:r['time_plus_shape_s']) if candidates else None
    write_json(stage/'selection.json',dict(selected=selected,training_only=True,goal_complete=False))
    development=None
    if selected:
        model.load_state_dict(torch.load(stage/selected['model'],map_location='cpu',weights_only=False)['policy'])
        result=evaluate(model,[np.asarray(p) for p in paths['validation']],stage,'development',None,costs,started+300,environment_factory)
        development=summary(stage/'development',result)
    assert costs['evaluation_actor_calls']<=20000 and costs['evaluation_physics_ticks']<=450000
    write_json(stage/'results.json',dict(screening=rows,full_training=full,selected=selected,
        development=development,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_updated_head_causal_ablation',costs=costs,
                       selected_model=selected['model'] if selected else None,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
