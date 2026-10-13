"""Bounded exact-predictor backtracking after terminal branches stabilize."""
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from event_adaptive_terminal_prediction import predict_terminal
from transition_waveform_model import chain_responses
from cruise_update_geometry import CruiseUpdateGeometry
from audit_ordinary_joint_errors import metrics
from audit_function_constraint_interference import command_segment_margin
from solve_complete_branch_joint_proposal import SHAPE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    solve=BASE/'canonical_gate_band_complete_branch_joint_proposal_20260918'
    source=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    reference=BASE/'canonical_gate_band_shape_priority_current_reference_20260918'
    geometry_source=BASE/'canonical_gate_band_current_hold_geometry_20260918'
    data,fit,physical=read(source/'results.json'),read(solve/'results.json'),read(reference/'results.json')
    assert not fit['eligible_for_actual_probe'] and not fit['history'][-1]['changed_terminal_modes']
    assert fit['history'][-1]['conditional_model_feasible']
    parent=BASE/data['parent_model'];assert sha(parent)==fit['parent_sha256']==physical['parent_sha256']
    geometry=CruiseUpdateGeometry(geometry_source,'physical',sha(parent))
    factory=checkpoint_environment_factory(torch.load(parent,map_location='cpu',weights_only=False)['architecture'])
    scales=(.999,.995,.99,.975,.95)
    stage=register('canonical_gate_band_complete_branch_interior_20260918',
        [__file__,solve/'direction.npz',solve/'results.json',solve/'receipt.json',solve/'error_model.json',
         source/'results.json',source/'receipt.json',reference/'results.json',parent,
         BASE/'event_adaptive_terminal_prediction.py',BASE/'transition_waveform_model.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=180,
             exact_backtracking_scales=list(scales),terminal_planner_queries=3500),
        'The three-round conditional solve ended with stable terminal modes and '
        'one remaining exact-predictor reverse-turn violation of2.872e-7 relative '
        'to its original ceiling. Do NOT loosen that ceiling or declare tolerance '
        'success. Check five predeclared decreasing scales against the unchanged '
        'exact full hybrid predictor, all original per-corner ceilings, measured '
        'error coefficients, half corridor margins, support and one-percent '
        'aggregate shape requirement. Select the first genuinely passing '
        'proposal. This is a bounded interior selection without optimizer '
        'updates or new physics; actual complete-policy validation is still required.')
    started=time.perf_counter();costs=dict(new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=0)
    with np.load(solve/'direction.npz') as z:direction=z['direction'].copy().ravel()
    rates={(r['path'],r['metric']):r['rate'] for r in read(solve/'error_model.json')['coefficients']}
    base={};margin={}
    for r in physical['baseline']['metrics']:
        v=dict(zip(physical['metric_names'],r['values']));margin[r['path']]=.5*(1.25-v['segment_max_distance_mm'])
        if r['path']<16:base[r['path']]={k:v[k] for k in v if k not in ('segment_max_distance_mm','complete_time_s')}
    totals={k:sum(v[k] for v in base.values()) for k in SHAPE}
    cached=[]
    for item in data['records']:
        with np.load(source/item['file']) as z:local={k:z[k].copy() for k in z.files}
        rows=local['rows'];stop=int(np.flatnonzero(rows[:,12])[0])
        cached.append((item['path'],local['reference'],rows,stop,chain_responses(local,direction[None,:])[:,:,0]))
    def budget():assert time.perf_counter()-started<175 and costs['terminal_planner_queries']<3500
    trials=[];selected=None
    for scale in scales:
        x=direction*scale;norm2=float(x@geometry.metric@x);current=dict.fromkeys(SHAPE,0.);records=[];ok=True
        for path,ref,rows,stop,response in cached:
            budget();p=predict_terminal(ref,rows[:,:6]+scale*response,stop,factory,costs,budget)
            r=dict(path=path,supported=p['supported'])
            if not p['supported']:
                ok=False;records.append(r);continue
            distance=command_segment_margin(ref,p['rows'][:,:2])['max_distance_upper_mm']
            r.update(initiation_tick=p['initiation_tick'],braking_steps=p['braking_steps'],
                time_s=p['total_ticks']*.001,prefix_inside=p['predicted_prefix_inside'],
                corridor_upper_mm=distance,violations=[])
            ok &= p['predicted_prefix_inside'] and distance<=1.25-margin[path]+1e-8
            if path in base:
                raw=metrics(ref,p['rows']);corrected={k:v+rates.get((path,k),0.)*norm2 for k,v in raw.items()}
                for key,value in corrected.items():
                    violation=(value-base[path][key])/max(base[path][key],1e-3)
                    if violation>1e-7:ok=False;r['violations'].append(dict(metric=key,scaled_violation=violation))
                for key in SHAPE:current[key]+=corrected[key]
            records.append(r)
        ratios={k:current[k]/v for k,v in totals.items() if v>1e-8}
        support,_=geometry.support_constraint(x)
        ok &= all(v<=.99+1e-7 for v in ratios.values()) and support<=1e-7 and norm2<=.1**2+1e-10
        trial=dict(scale=scale,eligible=bool(ok),records=records,corrected_shape_fractions=ratios,
                   metric_norm=float(np.sqrt(norm2)),support_violation=support)
        trials.append(trial)
        print('BRANCH_INTERIOR',scale,bool(ok),[(r['path'],r.get('violations')) for r in records if r.get('violations')],flush=True)
        if ok:
            selected=trial;np.savez_compressed(stage/'direction.npz',direction=x.reshape(3,256));break
    write_json(stage/'results.json',dict(parent_sha256=sha(parent),parent_model=data['parent_model'],
        eligible_for_actual_probe=bool(selected),selection=selected,trials=trials,costs=costs,
        prospective_error_bound=False,goal_complete=False))
    receipt(stage,dict(status='completed_exact_branch_interior_selection',costs=costs,
        eligible_for_actual_probe=bool(selected),wall_s=time.perf_counter()-started,goal_complete=False))


if __name__=='__main__':main()
