"""Model the unchanged brake-trigger branch before using smooth trajectory credit."""
import argparse
import time
import numpy as np
from joint_runtime import terminal
from canonical_gate_state import checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_closed_loop_tangent_credit import STATE_SCALE,shift_state
from closed_loop_transition_credit import ACTION_SCALE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json
from terminal_duration_credit import fixed_duration_jacobian, TerminalDurationGuard


def state_responses(local):
    response=np.zeros((7,768));out=[]
    for k,sj in enumerate(local['state_jacobian']):
        out.append(response.copy());direct=np.zeros((3,768))
        for h in range(3):direct[h,h*256:(h+1)*256]=local['features'][k,h]/ACTION_SCALE[h]
        response=sj[:,:7]@response+sj[:,7:]@direct
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection',default='canonical_gate_band_complete_transition_credit_20260917')
    parser.add_argument('--reference',default='canonical_gate_band_coherent_parameter_credit_20260917')
    parser.add_argument('--coverage',default='canonical_gate_band_transition_response_coverage_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_terminal_trigger_boundary_model_20260917')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    source=BASE/args.collection
    probes=BASE/args.reference
    coverage=BASE/args.coverage
    data=read(source/'results.json');physical=read(probes/'results.json');xy=read(coverage/'results.json')
    metadata=torch.load(BASE/data['parent_model'],map_location='cpu',weights_only=False)
    factory=checkpoint_environment_factory(metadata['architecture'])
    stage=register(args.stage,
        [__file__,source/'results.json',source/'receipt.json',probes/'results.json',
         probes/'receipt.json',probes/'basis.npz',coverage/'results.json',coverage/'receipt.json',
         BASE/'probe_coordinated_suffix.py',BASE/'joint_runtime.py',BASE/'transition_waveform_model.py',
         BASE/'terminal_duration_credit.py',terminal.__file__],
        dict(new_actor_decisions=0,new_execution_ticks=0,terminal_planner_queries=5000,
             saved_geometry_ticks=65000,wall_s=180,trigger_reserve_fraction=.5),
        'The earlier-center full-XY audit failed and remains unchanged. Path7 positive '
        'feed delays the brake initiation840->860ms: its planned stop-endpoint '
        'distance at840ms changes0.968873->1.057057mm across the unchanged1mm '
        'trigger. Model that discrete boundary explicitly at the bound CURRENT center. Reconstruct saved '
        'decision states and evaluate the original brake planner near the final '
        'endpoint, without executing new trajectories. Differentiate trigger '
        'distance minus1mm analytically WITHIN the selected frozen cubic duration. '
        'Protect selected-duration feasibility, nonreversal and an infeasibility '
        'witness for every shorter duration. Chain into current shared '
        'parameters. Require half the nominal signed trigger margin to remain '
        'at every modeled decision. This is a LOCAL LEARNING restriction, not '
        'a runtime brake delay, new planner or relaxed stopping condition. Check '
        'all previously measured policy variants: every changed initiation must '
        'be rejected by these new event constraints; only unchanged-initiation '
        'and unchanged-duration variants can validate the continuous XY model. Do not relabel excluded '
        'variants as successful predictions or assert guarantees elsewhere.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,terminal_planner_queries=0)
    records=[];constraints=[];bounds=[];labels=[];event_details=[];duration_data={}
    def plan(state):
        assert time.perf_counter()-start<175 and costs['terminal_planner_queries']<5000
        costs['terminal_planner_queries']+=1
        return terminal.plan_planar_terminal_brake(state['position'],state['velocity'],state['acceleration'],
                                                   dt=.001,limits=terminal.LIMITS)
    with np.load(probes/'basis.npz') as z:directions=z['basis']*z['amplitudes'][:,None]
    try:
        for j,record in enumerate(data['records']):
            path=record['path'];assert physical['baseline']['results'][j]['original_path']==path
            with np.load(source/record['file']) as z:local={k:z[k].copy() for k in z.files}
            ref,rows,actions=local['reference'],local['rows'],local['actions'];states=[]
            env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
                            state_snapshots=states,env_factory=factory)
            states.append(env.state());responses=state_responses(local)
            local_constraints=[];local_duration_guards=[];nominal_terminal_count=None
            for k,state in enumerate(states):
                env.restore(state);remaining=env.corridor.arcs[-1]-env.progress()[0]
                if env.tracker.frontier<len(env.corridor.gates) or remaining>13.:continue
                item=dict(path=path,decision=k,remaining_mm=float(remaining),terminal=k==len(states)-1)
                try:brake_plan=plan(state)
                except ValueError as exc:
                    item.update(mode='no_nominal_brake_plan',error=str(exc));event_details.append(item);continue
                columns=brake_plan['columns'];point=np.array([columns['x_mm'][-1],columns['y_mm'][-1]])
                vector=point-ref[-1];distance=np.linalg.norm(vector);q=float(distance-1.)
                if remaining>12. and q<0:
                    item.update(mode='remaining_distance_guard',trigger_value=q);event_details.append(item);continue
                assert (q<=0)==item['terminal'],(path,k,q,item)
                count=brake_plan['braking_steps']
                endpoint_jacobian=fixed_duration_jacobian(count,len(columns['time_s'])-1)[-1,:2,:7]
                derivative=(vector/max(distance,1e-30))@endpoint_jacobian
                state6=np.r_[state['position'],state['velocity'],state['acceleration']]
                guard=TerminalDurationGuard(state6,responses[k],count)
                local_duration_guards.append(guard)
                guard_label=f'path{path}_decision{k}'
                duration_data[guard_label+'_state']=state6
                duration_data[guard_label+'_response']=responses[k]
                duration_data[guard_label+'_count']=count
                if item['terminal']:nominal_terminal_count=count
                gradient=derivative@responses[k]
                # Stop must remain eligible; earlier stops must remain ineligible.
                row=gradient if item['terminal'] else -gradient
                bound=.5*abs(q)
                label=f'path{path}_decision{k}_'+('retain_stop' if item['terminal'] else 'retain_no_stop')
                constraints.append(row);bounds.append(bound);labels.append(label)
                local_constraints.append(len(bounds)-1)
                item.update(mode='modeled',trigger_value=q,derivative_modes=['exact_fixed_duration']*6,
                            braking_steps=count,constraint=label)
                event_details.append(item)
            assert local_constraints and event_details[-1]['terminal']
            baseline_tick=int(np.flatnonzero(rows[:,12])[0]);checks=[]
            for candidate in physical['candidates']:
                with np.load(BASE/candidate['traces'][j]) as z:actual=z['rows'].copy()
                tick=int(np.flatnonzero(actual[:,12])[0]);direction=candidate['sign']*directions[candidate['direction']]
                violation=np.asarray(constraints)[local_constraints]@direction-np.asarray(bounds)[local_constraints]
                duration_violation=max(value for guard in local_duration_guards for value,_,_,_ in guard.evaluate(direction))
                # Every saved successful tail contains the original 23 hold ticks.
                count=len(actual)-tick-23
                error=next(e for e in xy['records'] if e['path']==path)
                e=next(e for e in error['errors'] if e['direction']==candidate['direction'] and e['sign']==candidate['sign'])
                checks.append(dict(direction=candidate['direction'],sign=candidate['sign'],
                    baseline_initiation_tick=baseline_tick,actual_initiation_tick=tick,
                    initiation_changed=tick!=baseline_tick,duration_changed=count!=nominal_terminal_count,
                    baseline_braking_steps=nominal_terminal_count,actual_braking_steps=count,
                    event_constraint_rejects=bool(violation.max()>1e-8 or duration_violation>1e-10),
                    largest_duration_guard_violation=float(duration_violation),
                    largest_predicted_event_violation_mm=float(violation.max()),xy_error_mm=e['max_error_mm'],
                    continuous_xy_margin_check=bool(e['max_error_mm']<.5*error['corridor_margin_mm'])))
            records.append(dict(path=path,checks=checks))
        a=np.asarray(constraints);b=np.asarray(bounds)
        conditional=all((c['event_constraint_rejects'] if c['initiation_changed'] or c['duration_changed'] else c['continuous_xy_margin_check'])
                        for r in records for c in r['checks'])
        switches=[dict(path=r['path'],**c) for r in records for c in r['checks'] if c['initiation_changed'] or c['duration_changed']]
        np.savez_compressed(stage/'constraints.npz',matrix=a,bound=b,labels=np.array(labels))
        np.savez_compressed(stage/'duration_guards.npz',**duration_data)
        write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],events=event_details,
            records=records,observed_switches=switches,conditional_measured_coverage_passed=conditional,
            original_unconditional_coverage_failed=not xy['measured_coverage_passed'],actual_new_update_guaranteed=False,goal_complete=False))
        receipt(stage,dict(status='completed_terminal_trigger_boundary_model',constraints=len(b),
            conditional_measured_coverage_passed=conditional,costs=costs,wall_s=time.perf_counter()-start))
        print('TERMINAL_BOUNDARY_MODEL',conditional,'constraints',len(b),'switches',switches,flush=True)
    except Exception as exc:
        receipt(stage,dict(status='failed_terminal_trigger_boundary_model',error=str(exc),
            costs=costs,wall_s=time.perf_counter()-start))
        raise


if __name__=='__main__':main()
