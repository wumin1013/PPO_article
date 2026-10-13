"""Bidirectional one-decision gate-support eligibility intervention."""
import time
from unittest.mock import patch
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,environment_factory,canonical_observation
from continuous_gate_guidance import active_gate_transitions
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import score_complete
from audit_xy_curvature_resolution import curvature_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_waveform_learned_update_20260917'
    probe=BASE/'canonical_recentered_waveform_corrected_composition_20260917'
    data=read(parent/'training_selection.json');candidate=read(probe/'results.json')['candidates'][1]
    models=[parent/data['model'],probe/'step+0.125.pt'];path=11;anchor=25
    traces=[]
    for source in (data,candidate):
        j=next(i for i,r in enumerate(source['results']) if r['original_path']==path)
        traces.append(BASE/source['traces'][j])
    stage=register('canonical_gate_lateral_support_causal_probe_20260917',
        [__file__,*models,*traces,probe/'receipt.json',
         BASE/'canonical_recentered_path11_event_audit_20260917/results.json',
         BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py',
         BASE/'probe_coordinated_suffix.py',BASE/'validate_actions.py'],
        dict(path=path,anchor=anchor,suffixes=2,actor_calls=250,new_physics_ticks=8000,
            saved_prefix_geometry_ticks=1000,wall_s=60),
        'At the observed500ms support boundary, parent projection distance1.24662mm '
        'activates state blending, candidate1.25112mm disables it. Both actual '
        'positions and complete traces are feasible. Test that Boolean observation '
        'support decision alone: candidate receives active blending computed at '
        'its OWN physical state, parent receives inactive blending at its OWN '
        'state. Change only this one policy call, then restore each unmodified '
        'policy and runtime through full braking/rest. No guide, physical state, '
        'coverage, limits or later feedback override. Require reciprocal rescue '
        'and impairment for the proposed causal explanation. Diagnostic only; '
        'no deployed intervention or learned-success claim.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    records=[]
    for index,(model,trace) in enumerate(zip(models,traces)):
        p=CanonicalGatePolicy(256);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,anchor*20,costs,env_factory=environment_factory)
        state=env.state();original=env.observation();latent=p.act(original,True)[0];costs['actor_calls']+=1
        assert np.array_equal(env.decode_action(latent),actions[anchor])
        active=active_gate_transitions(env,2.);assert bool(active)==(index==0)
        gate=env.corridor.gates[0];distance=float((env.position-gate.point)@gate.normal)
        replacement=[] if index==0 else [(gate,distance)]
        with patch('canonical_gate_state.active_gate_transitions',return_value=replacement):
            altered=canonical_observation(env)
        assert env.tracker.frontier==state['tracker']['frontier']
        changed=p.act(altered,True)[0];costs['actor_calls']+=1
        held_delta=env.decode_action(changed)-env.decode_action(latent)
        chunks=[];held=[]
        while not env.done:
            assert time.perf_counter()-start<60 and costs['actor_calls']<250 and costs['new_physics_ticks']<5880
            if chunks:changed=p.act(env.observation(),True)[0];costs['actor_calls']+=1
            _,_,_,info=env.step(changed);chunks.append(info['rows']);held.append(info['held_action'])
            costs['new_physics_ticks']+=len(info['rows'])
        full=np.vstack([rows[:anchor*20],*chunks]);all_actions=np.vstack([actions[:anchor],held])
        assert np.array_equal(full[:anchor*20],rows[:anchor*20])
        audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
        name=f'recipient{index}.npz';np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions)
        record=dict(recipient='parent' if index==0 else 'candidate',model_sha256=sha(model),
            original_active=bool(active),intervened_active=bool(replacement),held_action_delta=held_delta.tolist(),
            success=success,reason=env.reason,trace=name,audit=audit,geometry=env.tracker.snapshot(1.25),
            baseline=score_complete(ref,rows,actions),**score_complete(ref,full,all_actions,env),
            curvature=curvature_metrics(ref,full),full_time_s=len(full)*.001 if success else None)
        records.append(record);write_json(stage/'results.json',dict(records=records,costs=costs,diagnostic_only=True,goal_complete=False))
        print('GATE_SUPPORT_CAUSE',record['recipient'],success,held_delta.tolist(),
            'reverse',record['baseline']['shape']['full_physical_reverse_deg'],record['shape']['full_physical_reverse_deg'],
            'overshoot',record['baseline']['shape']['heading_overshoot_deg'],record['shape']['heading_overshoot_deg'],flush=True)
    receipt(stage,dict(status='completed_bidirectional_gate_support_causal_probe',
        suffixes=2,complete_suffixes=sum(r['success'] for r in records),costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
