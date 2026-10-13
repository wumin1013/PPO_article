"""Validate a branch-aware predictor on already completed independent responses."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from transition_waveform_model import chain_responses
from event_adaptive_terminal_prediction import predict_terminal
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def compare(actual,predicted):
    length=max(len(actual),len(predicted))
    def pad(rows):return np.vstack([rows,np.repeat(rows[-1:],length-len(rows),axis=0)])
    error=np.linalg.norm(pad(actual)[:,:2]-pad(predicted)[:,:2],axis=1)
    return dict(maximum_xy_error_mm=float(error.max()),rms_xy_error_mm=float(np.sqrt(np.mean(error**2))),
        worst_tick=int(error.argmax()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--collection',default='canonical_gate_band_segment_cruise_credit_20260918')
    parser.add_argument('--reference',default='canonical_gate_band_segment_cruise_reference_20260918')
    parser.add_argument('--stage',default='canonical_gate_band_event_adaptive_terminal_prediction_20260918')
    parser.add_argument('--skip-oracle-controls',action='store_true')
    args=parser.parse_args()
    source=BASE/args.collection
    reference=BASE/args.reference
    feed_probe=BASE/'canonical_gate_band_segment_cruise_probe_20260918'
    source_data=read(source/'results.json');physical=read(reference/'results.json')
    parent=BASE/source_data['parent_model'];md=torch.load(parent,map_location='cpu',weights_only=False)
    assert sha(parent)==physical['parent_sha256']==source_data['parent_sha256']
    factory=checkpoint_environment_factory(md['architecture'])
    stage=register(args.stage,
        [__file__,BASE/'event_adaptive_terminal_prediction.py',BASE/'terminal_duration_credit.py',
         BASE/'transition_waveform_model.py',source/'results.json',source/'receipt.json',
         reference/'results.json',reference/'basis.npz',reference/'receipt.json',
         feed_probe/'results.json',feed_probe/'receipt.json',
         BASE/'canonical_gate_band_terminal_mode_time_lock_20260918/results.json',parent],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=150,
             terminal_planner_queries=2500,saved_reference_variants=116,
             earlier_stop_oracle_controls=0 if args.skip_oracle_controls else 4),
        'Fixed initiation and duration locks complete time. Test an offline hybrid '
        'predictor that retains current affine ordinary closed-loop credit, replays '
        'ordered geometry on its predicted XY, and queries the UNCHANGED exact '
        'brake planner at each eligible20ms decision. Its selected duration and '
        'initiation may change; integrate the exact selected cubic tail. Never '
        'continue ordinary motion beyond supplied credit support, and mark missing '
        'later support explicitly. Validate nominal identity on29 saved paths and '
        'compare all four stored parameter responses, including prior failures, '
        'against the fixed-branch prediction. Separately verify event logic on '
        'the four existing feed probes using their actual ordinary prefixes; '
        'these oracle-prefix identities do not validate unknown ordinary prediction. '
        'No executed trajectory, brake/stop requirement, policy or physical limit '
        'changes. Prediction improvement alone cannot certify a learned candidate.')
    start=time.perf_counter();costs=dict(new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=0)
    def budget():
        assert time.perf_counter()-start<145 and costs['terminal_planner_queries']<2500
    with np.load(reference/'basis.npz') as z:directions=z['basis']*z['amplitudes'][:,None]
    records=[];nominal_identity=[]
    for j,item in enumerate(source_data['records']):
        with np.load(source/item['file']) as z:local={k:z[k].copy() for k in z.files}
        rows=local['rows'];ref=local['reference'];stop=int(np.flatnonzero(rows[:,12])[0])
        nominal=predict_terminal(ref,rows[:,:6],stop,factory,costs,budget)
        assert nominal['supported'] and len(nominal['rows'])==len(rows)
        identity=float(abs(nominal['rows']-rows[:,:6]).max());assert identity<1e-7
        nominal_identity.append(dict(path=item['path'],maximum_state_error=identity))
        responses=chain_responses(local,directions)
        for c in physical['candidates']:
            with np.load(BASE/c['traces'][j]) as z:actual=z['rows'].copy()
            assert physical['baseline']['results'][j]['original_path']==item['path']
            affine=rows[:,:6]+c['sign']*responses[:,:,c['direction']]
            prediction=predict_terminal(ref,affine,stop,factory,costs,budget)
            actual_stop=int(np.flatnonzero(actual[:,12])[0]);actual_count=len(actual)-actual_stop-23
            record=dict(path=item['path'],direction=c['direction'],sign=c['sign'],
                actual_initiation_tick=actual_stop,actual_braking_steps=actual_count,
                fixed_branch_error=compare(actual[:,:6],affine),hybrid_supported=prediction['supported'])
            if prediction['supported']:
                record.update(hybrid_error=compare(actual[:,:6],prediction['rows']),
                    predicted_initiation_tick=prediction['initiation_tick'],
                    predicted_braking_steps=prediction['braking_steps'],
                    predicted_prefix_inside=prediction['predicted_prefix_inside'],
                    actual_total_ticks=len(actual),predicted_total_ticks=prediction['total_ticks'],
                    mode_matches=prediction['initiation_tick']==actual_stop and prediction['braking_steps']==actual_count)
            else:record['unsupported_reason']=prediction['reason']
            records.append(record)
        write_json(stage/'results.json',dict(parent_sha256=sha(parent),records=records,
            nominal_identity=nominal_identity,costs=costs,goal_complete=False))
    oracles=[]
    for item in ([] if args.skip_oracle_controls else read(feed_probe/'results.json')['records']):
        with np.load(feed_probe/item['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        stop=int(np.flatnonzero(rows[:,12])[0])
        predicted=predict_terminal(ref,rows[:,:6],stop,factory,costs,budget)
        assert predicted['supported'] and len(predicted['rows'])==len(rows)
        error=float(abs(predicted['rows']-rows[:,:6]).max());assert error<1e-7
        oracles.append(dict(path=item['path'],initiation_tick=stop,braking_steps=predicted['braking_steps'],
            actual_prefix_identity_error=error,predictive_ordinary_credit_verified=False))
    supported=[r for r in records if r['hybrid_supported']]
    result=dict(parent_sha256=sha(parent),records=records,nominal_identity=nominal_identity,
        actual_prefix_oracle_controls=oracles,supported_responses=len(supported),
        exact_terminal_mode_matches=sum(r['mode_matches'] for r in supported),
        terminal_or_ordinary_remaining_errors=True,costs=costs,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_event_adaptive_terminal_prediction_audit',costs=costs,
        supported_responses=len(supported),exact_terminal_mode_matches=result['exact_terminal_mode_matches'],
        wall_s=time.perf_counter()-start,goal_complete=False))
    print('HYBRID_TERMINAL_PREDICTION',len(supported),result['exact_terminal_mode_matches'],
        [(r['path'],r['direction'],r['fixed_branch_error']['maximum_xy_error_mm'],r['hybrid_error']['maximum_xy_error_mm'])
         for r in supported if r['hybrid_error']['maximum_xy_error_mm']<.5*r['fixed_branch_error']['maximum_xy_error_mm']],flush=True)


if __name__=='__main__':main()
