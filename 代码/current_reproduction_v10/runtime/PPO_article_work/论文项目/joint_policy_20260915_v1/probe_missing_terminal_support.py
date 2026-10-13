"""Bounded complete physical checks of missing ordinary terminal support."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from train_pilot import evaluate,save_torch
from transition_waveform_model import chain_responses
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='canonical_gate_band_complete_branch_interior_solve_20260918')
    parser.add_argument('--stage',default='canonical_gate_band_missing_terminal_support_probe_20260918')
    args=parser.parse_args()
    source=BASE/args.source
    credit=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    solved=read(source/'results.json');data=read(credit/'results.json')
    last=solved['history'][-1]
    unsupported=[r['path'] for r in last['exact_branch_checks']['records'] if not r['supported']]
    assert 1<=len(unsupported)<=2 and last['conditional_model_feasible']
    assert all(not r.get('shape_violations') for r in last['exact_branch_checks']['records'])
    model=BASE/solved['parent_model'];assert sha(model)==data['parent_sha256']==solved['parent_sha256']
    stage=register(args.stage,
        [__file__,model,source/'results.json',source/'receipt.json',source/'direction.npz',
         credit/'results.json',credit/'receipt.json',BASE/'train_pilot.py',BASE/'joint_runtime.py',
         BASE/'canonical_gate_state.py',BASE/'gate_band_state.py'],
        dict(paths=unsupported,full_policy_rollouts=len(unsupported),evaluation_actor_calls=800*len(unsupported),
             evaluation_physics_ticks=18000*len(unsupported),wall_s=60,automatic_adoption=False),
        'Only the explicitly registered paths lack an eligible predicted brake within available ordinary '
        'support; all other exact-model shape checks passed for this explicitly '
        'bound proposal. Do not invent later ordinary states or silently remove '
        'these failures. Execute this one shared learned-parameter candidate on '
        'the actual complete training paths with the unchanged runtime, guide, '
        'planner and physical limits. Retain its failure or full brake/hold '
        'and compare actual ordinary state at the support boundary. Diagnostic '
        'only; if successful, reuse this exact trace for a subsequent complete '
        '29-route screen rather than repeat it. No physical success inferred '
        'from a conditional model or failed prefix.')
    start=time.perf_counter();torch.set_num_threads(1)
    md=torch.load(model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(md['policy'])
    with np.load(source/'direction.npz') as z:direction=z['direction'].copy()
    with torch.no_grad():p.residual_coefficients.add_(torch.tensor(direction,dtype=torch.float32))
    save_torch(stage/'diagnostic_candidate.pt',dict(md,policy=p.state_dict(),diagnostic_only=True,
        source_direction_sha256=sha(source/'direction.npz'),parent_sha256=sha(model)))
    costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    records=[]
    for path in unsupported:
        item=next(r for r in data['records'] if r['path']==path)
        with np.load(credit/item['file']) as z:local={k:z[k].copy() for k in z.files}
        tag=f'path{path}'
        result=evaluate(p,[local['reference']],stage,tag,None,costs,start+55,
            checkpoint_environment_factory(md['architecture']))[0]
        with np.load(stage/tag/'0.npz') as z:actual=z['rows'].copy()
        old_stop=int(np.flatnonzero(local['rows'][:,12])[0])
        response=chain_responses(local,direction.ravel()[None,:])[:,:,0]
        predicted=local['rows'][:,:6]+response
        actual_stops=np.flatnonzero(actual[:,12])
        detail=dict(path=path,result=dict(result,original_path=path),trace=str((stage/tag/'0.npz').relative_to(BASE)),
            old_ordinary_support_ticks=old_stop,actual_brake_start_tick=int(actual_stops[0]) if len(actual_stops) else None,
            strict_vector_limits=bool(np.all(np.asarray(result['audit']['peaks'])<=[100.,2000.,62500.])),
            diagnostic_only=True)
        if len(actual)>=old_stop:
            detail['support_boundary_state_error']=(actual[old_stop-1,:6]-predicted[old_stop-1]).tolist()
            detail['actual_boundary_state']=actual[old_stop-1,:6].tolist()
            detail['predicted_boundary_state']=predicted[old_stop-1].tolist()
        records.append(detail)
        print('MISSING_TERMINAL_SUPPORT',detail,flush=True)
    assert costs['evaluation_actor_calls']<=800*len(unsupported) and costs['evaluation_physics_ticks']<=18000*len(unsupported)
    write_json(stage/'results.json',dict(parent_sha256=sha(model),source_direction_sha256=sha(source/'direction.npz'),
        scale=1.,records=records,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_missing_terminal_support_physical_probe',costs=costs,
        complete_success=all(r['result']['success'] for r in records),wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
