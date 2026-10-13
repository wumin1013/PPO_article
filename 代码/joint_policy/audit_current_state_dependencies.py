"""Audit omitted state fields on saved ordinary transitions; no new physics."""
import time
from datetime import datetime, timezone
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from continuous_gate_guidance import active_gate_transitions
from geometry_spec import NUMERIC_MM
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_shape_cruise_statewise_half_full_20260917'
    data=read(parent/'results.json');candidate=data['full_candidate']
    model=parent/data['chosen']['model']
    stage=register('canonical_gate_band_current_state_dependencies_20260917',
        [__file__,model,parent/'results.json',parent/'receipt.json']+
        [BASE/n for n in ('canonical_gate_state.py','gate_band_state.py',
         'closed_loop_transition_credit.py','joint_runtime.py','geometry_spec.py',
         'probe_coordinated_suffix.py','kernel_local_feedback_policy.py')],
        dict(saved_paths=64,new_physics_ticks=0,new_training_decisions=0,
             maximum_actor_queries=16000,saved_geometry_ticks=100000,wall_s=90),
        'Check whether omitted action history or discrete tracker fields explain '
        'remaining shape errors. Reconstruct all saved ordinary states exactly; '
        'audit actual pending-negative activation and gate-boundary proximity. '
        'At saved gate-band states compare both neighboring frontier encodings '
        'and their learned outputs at the identical physical state. Test history '
        'masking with changed held commands. Clocks remain equal at the same '
        'decision index; terminal branches are outside the ordinary7D model. '
        'Absence on these traces is not proof of global state sufficiency.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,actor_queries=0)
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture']);records=[]
    with torch.no_grad():
        for result,trace in zip(candidate['results'],candidate['traces']):
            assert result['success']
            with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            states=[];ordinary=int(np.flatnonzero(rows[:,12])[0])
            env=reconstruct(ref,rows,actions,ordinary,costs,state_snapshots=states,env_factory=factory)
            pending=sum(s['tracker']['pending_negative'] for s in states)
            history_error=0.;frontier_observation_error=0.;frontier_action_error=0.;band_queries=0
            # Every macro state is already represented by the7 continuous fields,
            # a fixed clock and the actual nominal discrete mode.
            for k,state in enumerate(states):
                assert state['ticks']==20*k and state['brake'] is None
                env.restore(state);obs=env.observation()
                # Both policy branches must be independent of these three inputs.
                changed=obs.copy();changed[8:11]=np.array([-.91,.97,2.3])
                means=policy.distribution(torch.tensor(np.stack([obs,changed]))).mean.numpy()
                costs['actor_queries']+=2
                history_error=max(history_error,float(abs(means[1]-means[0]).max()))
                active=active_gate_transitions(env,2.,require_channel=False)
                if active:
                    assert len(active)==1
                    gate,_=active[0];saved_frontier=env.tracker.frontier;pair=[]
                    for frontier in (gate.vertex_index-1,gate.vertex_index):
                        env.tracker.frontier=frontier;pair.append(env.observation())
                    env.tracker.frontier=saved_frontier
                    frontier_observation_error=max(frontier_observation_error,float(abs(pair[1]-pair[0]).max()))
                    means=policy.distribution(torch.tensor(np.stack(pair))).mean.numpy()
                    costs['actor_queries']+=2;band_queries+=1
                    frontier_action_error=max(frontier_action_error,float(abs(means[1]-means[0]).max()))
                assert costs['actor_queries']<=16000 and costs['saved_prefix_geometry_ticks']<=100000
                assert time.perf_counter()-start<90
            gate_distances=[abs((rows[:ordinary,:2]-g.point)@g.normal).min() for g in env.corridor.gates]
            records.append(dict(path=result['original_path'],ordinary_decisions=len(states),
                pending_negative_decisions=int(pending),minimum_gate_plane_distance_mm=float(min(gate_distances)) if gate_distances else None,
                gate_boundary_numeric_mm=NUMERIC_MM,history_action_error=history_error,
                gate_band_queries=band_queries,frontier_observation_error=frontier_observation_error,
                frontier_action_error=frontier_action_error))
    out=dict(parent_sha256=sha(model),records=records,costs=costs,
        ordinary_clock_equal_at_same_decision=True,terminal_credit_is_separately_conditional=True,
        history_mask_verified=all(r['history_action_error']==0 for r in records),
        frontier_band_equivalence_verified=all(r['frontier_action_error']==0 for r in records),
        pending_negative_present=any(r['pending_negative_decisions'] for r in records),
        global_markov_sufficiency_claim=False,goal_complete=False)
    write_json(stage/'results.json',out)
    receipt(stage,dict(status='completed_saved_state_dependency_audit',costs=costs,
        wall_s=time.perf_counter()-start,new_physics_ticks=0))
    status=read(BASE/'当前执行状态.json')
    status['current'].update(state='shape_priority_mechanism_comparison_preparing',
        training_running=False,physical_diagnostic_running=False,diagnostic_stage=None,
        working_candidate=str(model.relative_to(BASE)),working_candidate_sha256=sha(model),
        candidate_final_accepted=False)
    status['running']=None;status['active_experiment']=None
    status['current_state_dependency_audit']=str(stage.relative_to(BASE))
    status['updated_utc']=datetime.now(timezone.utc).isoformat()
    write_json(BASE/'当前执行状态.json',status)
    print('STATE_DEPENDENCIES', {k:v for k,v in out.items() if k!='records'},flush=True)


if __name__=='__main__':main()
