"""Saved-state audit of nonmonotone closed-loop parameter updates."""
import time
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from joint_runtime import SENSOR_DISTANCES
from learn_single_decision_credit import SCREEN
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    small=BASE/'full_feedback_geometry_fixed_composition_20260917'
    large=BASE/'full_feedback_geometry_fixed_learning_20260917'
    selection=read(parent/'training_selection.json')
    sm=read(small/'results.json')['candidates'][1]
    screen=SCREEN+[38,39,42,43,44,45,46,47]
    models={0.:parent/selection['model'], .125:small/'parameter_+0.125.pt',
            .5:large/'candidate_0.5.pt', 1.:large/'candidate_1.pt'}
    stage=register('large_step_feedback_event_audit_20260917',
        [__file__,parent/'training_selection.json',small/'results.json',large/'results.json',
         BASE/'probe_coordinated_suffix.py',BASE/'joint_runtime.py',
         BASE/'kernel_local_feedback_policy.py',*models.values()],
        dict(paths=[8,15],steps=list(models),new_physics_ticks=0,actor_calls=2500,
             saved_prefix_geometry_ticks=30000,wall_s=90),
        'Locate nonmonotone regressions using existing complete traces only. Reconstruct '
        'observations and gate states exactly, verify saved actions, distinguish direct '
        'parameter effects at parent states from closed-loop state feedback and discrete '
        'sensor/gate timing. No causal claim from temporal association alone; choose a '
        'bounded follow-up intervention only after locating the earliest divergence. '
        'No repeated candidate evaluation, training, or development/sealed access.')
    start=time.perf_counter();costs=dict(actor_calls=0,saved_prefix_geometry_ticks=0,new_physics_ticks=0)
    policies={}
    for step,model in models.items():
        p=KernelLocalFeedbackPolicy(256)
        p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        policies[step]=p
    records=[]
    for path in (8,15):
        traces={0.:BASE/selection['traces'][path],
                .125:BASE/sm['traces'][[r['original_path'] for r in sm['results']].index(path)],
                .5:large/'screen_0.5'/f'{screen.index(path)}.npz',
                1.:large/'screen_1'/f'{screen.index(path)}.npz'}
        cache={}
        for step,trace in traces.items():
            with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            obs=[];states=[]
            env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,obs,states)
            obs.append(env.observation());states.append(env.state());obs=np.asarray(obs)
            actual=np.asarray([policies[step].act(x,True)[0] for x in obs]);costs['actor_calls']+=len(obs)
            decoded=np.asarray([env.decode_action(x) for x in actual])
            assert np.array_equal(decoded,actions)
            front=[];membership=[];progress=[]
            for state in states:
                env.restore(state);s,ix=env.progress()
                front.append(ix);progress.append(s)
                membership.append(np.minimum(np.searchsorted(env.corridor.arcs,
                    np.clip(s+SENSOR_DISTANCES,0,env.corridor.arcs[-1]),side='right')-1,
                    len(env.corridor.lengths)-1))
            front=np.asarray(front);membership=np.asarray(membership)
            ni=np.array([-(ref[1]-ref[0])[1],(ref[1]-ref[0])[0]])
            turn=np.cross(ref[1]-ref[0],ref[2]-ref[1]);ni*=np.sign(turn)/np.linalg.norm(ni)
            no=np.array([-(ref[2]-ref[1])[1],(ref[2]-ref[1])[0]])*np.sign(turn)/np.linalg.norm(ref[2]-ref[1])
            speed2=np.sum(rows[:,2:4]**2,axis=1)
            omega=np.sign(turn)*np.cross(rows[:,2:4],rows[:,4:6])/np.maximum(speed2,1e-30)
            reverse=np.maximum(-omega,0)*(speed2>25)*.001*180/np.pi
            incoming=(rows[:,:2]-ref[1])@ni;outgoing=(rows[:,:2]-ref[1])@no
            cache[step]=dict(obs=obs,rows=rows,actions=actions,front=front,membership=membership,
                progress=np.asarray(progress),reverse=reverse,reference=ref)
            transitions=[dict(decision=int(j+1),sensor_distances_mm=SENSOR_DISTANCES[
                membership[j+1]!=membership[j]].tolist(),frontier_before=int(front[j]),
                frontier_after=int(front[j+1]),progress_mm=float(progress[j+1]),
                action_change=(actions[j+1]-actions[j]).tolist())
                for j in range(len(actions)-1) if np.any(membership[j+1]!=membership[j]) or front[j+1]!=front[j]]
            cache[step]['summary']=dict(step=step,trace=str(trace.relative_to(BASE)),
                trace_sha256=sha(trace),transitions=transitions,
                in_outward_mm=float(max(0,-incoming.min())),out_outward_mm=float(max(0,-outgoing.min())),
                out_peak_tick=int(np.argmin(outgoing)),reverse_deg=float(reverse.sum()),
                reverse_by_decision_deg=[float(reverse[j*20:(j+1)*20].sum()) for j in range(len(actions))],
                complete_time_s=len(rows)*.001)
        base=cache[0.]
        comparisons=[]
        for step in (.125,.5,1.):
            item=cache[step];n=min(len(base['actions']),len(item['actions']))
            at_base=np.asarray([policies[step].act(x,True)[0] for x in base['obs'][:n]])
            costs['actor_calls']+=n
            at_base=np.asarray([env.decode_action(x) for x in at_base])
            direct=at_base-base['actions'][:n]
            state=item['actions'][:n]-at_base
            mismatch=np.any(item['membership'][:n]!=base['membership'][:n],axis=1)
            gate=item['front'][:n]!=base['front'][:n]
            excess=[]
            for j in range(n):
                dr=float(item['reverse'][j*20:(j+1)*20].sum()-base['reverse'][j*20:(j+1)*20].sum())
                excess.append(dict(decision=j,reverse_change_deg=dr,
                    sensor_mismatch=bool(mismatch[j]),gate_mismatch=bool(gate[j]),
                    direct_action_delta=direct[j].tolist(),state_feedback_action_delta=state[j].tolist(),
                    xy_delta_mm=(item['rows'][j*20,:2]-base['rows'][j*20,:2]).tolist()))
            np.savez_compressed(stage/f'path{path}_step{step:g}.npz',
                observations=item['obs'],frontier=item['front'],membership=item['membership'],
                direct_action_delta=direct,state_feedback_action_delta=state,
                reverse=item['reverse'],parent_reverse=base['reverse'])
            comparisons.append(dict(step=step,sensor_mismatch_decisions=np.flatnonzero(mismatch).tolist(),
                gate_mismatch_decisions=np.flatnonzero(gate).tolist(),
                largest_reverse_increases=sorted(excess,key=lambda x:-x['reverse_change_deg'])[:8],
                largest_feedback_changes=sorted(excess,key=lambda x:-np.linalg.norm(
                    np.asarray(x['state_feedback_action_delta'])/[.02,5.,.5]))[:8]))
        records.append(dict(path=path,reference=ref.tolist(),traces=[cache[s]['summary'] for s in models],
            comparisons=comparisons))
        print('SAVED_EVENT_AUDIT',path,[(s,cache[s]['summary']['out_outward_mm'],cache[s]['summary']['reverse_deg']) for s in models],
            [(c['step'],c['sensor_mismatch_decisions'],c['gate_mismatch_decisions']) for c in comparisons],flush=True)
    assert costs['actor_calls']<=2500 and costs['saved_prefix_geometry_ticks']<=30000
    assert time.perf_counter()-start<90
    write_json(stage/'results.json',dict(records=records,costs=costs,association_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_saved_large_step_event_audit',costs=costs,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
