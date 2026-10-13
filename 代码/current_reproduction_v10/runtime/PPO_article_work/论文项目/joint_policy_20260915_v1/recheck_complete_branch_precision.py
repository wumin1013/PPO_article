"""Tighten a numerical distance enclosure without changing model or acceptance."""
import argparse
import shutil
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from transition_waveform_model import chain_responses
from event_adaptive_terminal_prediction import predict_terminal
from audit_function_constraint_interference import command_segment_margin
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--round-index',type=int)
    args=parser.parse_args();source=BASE/args.source
    credit=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    reference=BASE/'canonical_gate_band_shape_priority_current_reference_20260918'
    data=read(source/'results.json');local_data=read(credit/'results.json');ref=read(reference/'results.json')
    last=data['history'][args.round_index-1] if args.round_index is not None else data['history'][-1]
    if args.round_index is not None:assert last['outer_round']==args.round_index and args.round_index>=1
    direction_path=source/(f'round{args.round_index}_direction.npz' if args.round_index is not None else 'direction.npz')
    old=last['exact_branch_checks']
    parent=BASE/data['parent_model'];assert sha(parent)==data['parent_sha256']==local_data['parent_sha256']
    assert last['conditional_model_feasible'] and all(r['supported'] for r in old['records'])
    stage=register(args.stage,[__file__,source/'results.json',source/'receipt.json',direction_path,
        parent,credit/'results.json',credit/'receipt.json',reference/'results.json',
        BASE/'audit_function_constraint_interference.py',BASE/'event_adaptive_terminal_prediction.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,terminal_planner_queries=600,
             distance_enclosure_tolerances_mm=[1e-8,1e-10],unchanged_acceptance_slack_mm=1e-8,
             selected_saved_round=args.round_index),
        'The independent distance audit always adds its numerical tolerance1e-6mm '
        'to the sampled lower bound, whereas the conditional capsule model has '
        'a direct convex interval certificate. Test whether marginal failures '
        'are an enclosure-resolution artifact: reconstruct the same exact hybrid '
        'branches and tighten only the distance computation tolerance to1e-8 then '
        '1e-10mm. Keep the half-corridor-margin ceiling and1e-8 acceptance slack '
        'unchanged; no new fit, parameter, sample, physical limit or outcome is '
        'altered. Preserve old ineligible results. Physical feedback remains '
        'mandatory even if the refined predictor passes.')
    start=time.perf_counter();costs=dict(new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=0)
    def budget():
        assert time.perf_counter()-start<55 and costs['terminal_planner_queries']<600
    factory=checkpoint_environment_factory(torch.load(parent,map_location='cpu',weights_only=False)['architecture'])
    with np.load(direction_path) as z:direction=z['direction'].copy().ravel()
    ceilings={r['path']:.625+.5*dict(zip(ref['metric_names'],r['values']))['segment_max_distance_mm']
              for r in ref['baseline']['metrics']}
    prior={r['path']:r for r in old['records']};records=[]
    for item in local_data['records']:
        budget()
        with np.load(credit/item['file']) as z:local={k:z[k].copy() for k in z.files}
        stop=int(np.flatnonzero(local['rows'][:,12])[0])
        response=chain_responses(local,direction[None,:])[:,:,0]
        predicted=predict_terminal(local['reference'],local['rows'][:,:6]+response,stop,factory,costs,budget)
        original=prior[item['path']]
        assert predicted['supported']
        assert (predicted['initiation_tick'],predicted['braking_steps'])==(original['initiation_tick'],original['braking_steps'])
        measures=[command_segment_margin(local['reference'],predicted['rows'][:,:2],tolerance=e) for e in (1e-8,1e-10)]
        updated=dict(original,corridor_upper_mm=measures[-1]['max_distance_upper_mm'],
            previous_corridor_upper_mm=original['corridor_upper_mm'],corridor_ceiling_mm=ceilings[item['path']],
            refined_distance_enclosures=measures)
        updated['corridor_pass']=updated['corridor_upper_mm']<=ceilings[item['path']]+1e-8
        records.append(updated)
    eligible=all(r['corridor_pass'] and r['prefix_inside'] and not r['shape_violations'] for r in records)
    eligible &= old['saved_support_violation']<=1e-7 and last['metric_norm']<=.1+1e-9
    eligible &= all(v<=.99+1e-7 for v in old['corrected_shape_fractions'].values())
    checked=dict(old,records=records,eligible_for_actual_probe=bool(eligible))
    last=dict(last,exact_branch_checks=checked)
    shutil.copy2(direction_path,stage/'direction.npz')
    assert sha(stage/'direction.npz')==sha(direction_path)
    write_json(stage/'results.json',dict(data,history=[last],eligible_for_actual_probe=bool(eligible),
        source_stage=source.name,unchanged_direction_sha256=sha(direction_path),metric_norm=last['metric_norm'],
        distance_precision_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_tighter_distance_enclosure',eligible_for_actual_probe=bool(eligible),
        costs=costs,wall_s=time.perf_counter()-start,goal_complete=False))
    print('DISTANCE_PRECISION',eligible,[(r['path'],r['previous_corridor_upper_mm']-r['corridor_ceiling_mm'],
        r['corridor_upper_mm']-r['corridor_ceiling_mm']) for r in records if r['previous_corridor_upper_mm']>r['corridor_ceiling_mm']+1e-8])


if __name__=='__main__':main()
