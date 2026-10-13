"""Saved evidence for remaining composition error after point-sensor repair."""
import argparse
import time
import numpy as np
import torch
from continuous_sensor_runtime import environment_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',default='continuous_sensor_matched_function_transfer_20260917')
    parser.add_argument('--arm',default='updated')
    parser.add_argument('--probe',default='continuous_sensor_complete_composition_20260917')
    parser.add_argument('--stage',default='continuous_sensor_composition_event_audit_20260917')
    parser.add_argument('--paths',type=int,nargs='+',default=[8,13,15])
    args=parser.parse_args()
    torch.set_num_threads(1)
    parent=BASE/args.parent
    selected=next(r for r in read(parent/'results.json')['candidates']
        if r.get('arm',r.get('tag'))==args.arm or r.get('model')==args.arm)
    parent_model=parent/selected['model']
    metadata=torch.load(parent_model,map_location='cpu',weights_only=False)
    policy_class=KernelLocalFeedbackPolicy;current_factory=environment_factory
    if metadata['architecture'] in ('canonical_gate_kernel','gate_band_canonical_kernel'):
        from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
        policy_class=CanonicalGatePolicy;current_factory=checkpoint_environment_factory(metadata['architecture'])
    probe=BASE/args.probe;data=read(probe/'results.json')
    stage=register(args.stage,
        [__file__,parent/'results.json',parent_model,probe/'results.json',probe/'comparison.json',
         probe/'step+0.125.pt',probe/'step-0.125.pt',BASE/'continuous_sensor_runtime.py',
         BASE/'probe_coordinated_suffix.py',BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py'],
        dict(paths=args.paths,saved_geometry_ticks=15000,new_physics_ticks=0,actor_calls=1200,wall_s=60,
            checkpoint_architecture=metadata['architecture']),
        'Locate the remaining current small-step composition discrepancy without '
        'repeating physics. Reconstruct continuous observations for baseline and '
        'both signs; check saved actions, gate-frontier timing, endpoint-validity '
        'masks, and direct versus feedback action changes. Point-tangent membership '
        'is not treated as a discontinuity in the new encoder. Compare real reverse '
        'increments to test whether the remaining path15 regression coincides with '
        'an uncorrected discrete event or smooth nonlinear composition. This is '
        'association evidence only and cannot by itself establish a new cause.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,actor_calls=0,new_physics_ticks=0)
    cases=[]
    for path in args.paths:
        j=next(i for i,r in enumerate(selected['results']) if r['original_path']==path)
        variants=[(0.,parent_model,BASE/selected['traces'][j])]
        for candidate in data['candidates']:
            variants.append((candidate['step'],probe/f"step{candidate['step']:+g}.pt",BASE/candidate['traces'][j]))
        cache={}
        for step,model,trace in variants:
            p=policy_class(256);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
            with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            states=[];original=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,state_snapshots=states,env_factory=current_factory)
            states.append(original.state());env=current_factory(ref)
            xs=[];front=[];progress=[];heading=[];target=[]
            for index,state in enumerate(states):
                env.restore(state);x=env.observation();xs.append(x);front.append(env.tracker.frontier);progress.append(env.progress()[0])
                heading.append(env.heading);target.append(env.guidance_target())
                actual=p.act(x,True)[0];costs['actor_calls']+=1
                assert np.array_equal(env.decode_action(actual),actions[index])
            u,v=np.diff(ref,axis=0);sign=np.sign(u[0]*v[1]-u[1]*v[0])
            speed2=np.sum(rows[:,2:4]**2,axis=1)
            omega=sign*(rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4])/np.maximum(speed2,1e-30)
            rev=np.maximum(-omega,0.)*(speed2>25)*.001*180/np.pi
            cache[step]=dict(policy=p,x=np.asarray(xs),front=np.asarray(front),progress=np.asarray(progress),
                reverse=rev,actions=actions,rows=rows,trace=str(trace.relative_to(BASE)))
            np.savez_compressed(stage/f'path{path}_saved_step{step:+g}.npz',observations=xs,
                frontier=front,progress=progress,heading=heading,target=target,actions=actions,rows=rows)
        base=cache[0.];comparison=[]
        for step in (-.125,.125):
            q=cache[step];n=min(len(base['x']),len(q['x']))
            direct=np.asarray([env.decode_action(q['policy'].act(x,True)[0]) for x in base['x'][:n]])-base['actions'][:n]
            costs['actor_calls']+=n;feedback=q['actions'][:n]-base['actions'][:n]-direct
            gate=q['front'][:n]!=base['front'][:n]
            valid=np.any(q['x'][:n,62:72]!=base['x'][:n,62:72],axis=1)
            rows=[]
            for a in range(n):
                rows.append(dict(anchor=a,reverse_delta_deg=float(q['reverse'][a*20:(a+1)*20].sum()-base['reverse'][a*20:(a+1)*20].sum()),
                    direct_delta=direct[a].tolist(),feedback_delta=feedback[a].tolist(),
                    state_change_max=float(abs(q['x'][a]-base['x'][a]).max()),
                    changed_valid=bool(valid[a]),changed_frontier=bool(gate[a])))
            comparison.append(dict(step=step,frontier_mismatch=np.flatnonzero(gate).tolist(),
                validity_mismatch=np.flatnonzero(valid).tolist(),
                full_reverse_change_deg=float(q['reverse'].sum()-base['reverse'].sum()),
                largest_reverse_increases=sorted(rows,key=lambda r:-r['reverse_delta_deg'])[:5],
                largest_feedback=sorted(rows,key=lambda r:-np.linalg.norm(np.asarray(r['feedback_delta'])/[.005,.1,.1]))[:5]))
            np.savez_compressed(stage/f'path{path}_step{step:+g}.npz',observations=q['x'],parent_observations=base['x'],
                direct_action_delta=direct,feedback_action_delta=feedback,reverse=q['reverse'],parent_reverse=base['reverse'])
        cases.append(dict(path=path,comparisons=comparison))
        print('CONTINUOUS_EVENT_AUDIT',path,[(c['step'],c['frontier_mismatch'],c['validity_mismatch'],c['full_reverse_change_deg']) for c in comparison],flush=True)
    assert costs['saved_prefix_geometry_ticks']<=15000 and costs['actor_calls']<=1200 and time.perf_counter()-start<60
    write_json(stage/'results.json',dict(cases=cases,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_continuous_sensor_saved_event_audit',costs=costs,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
