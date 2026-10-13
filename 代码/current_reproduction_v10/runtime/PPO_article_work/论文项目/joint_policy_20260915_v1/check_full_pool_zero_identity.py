"""Check saved metrics and exact zero-update execution before further learning."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learn_full_pool_joint_credit import candidate_metadata
from canonical_gate_state import checkpoint_environment_factory
from learn_nonlinear_joint_feedback import rollout
from complete_corner_metrics import complete_corner_metrics
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    source=BASE/'canonical_gate_band_active_constraint_joint_learning_20260919'
    baseline=read(source/'baseline.json')['baseline']
    metadata=torch.load(parent,map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:directions=z['directions'].copy()
    stage=register('canonical_gate_band_current_zero_identity_check_20260919',
        [__file__,parent,source/'baseline.json',source/'parameter_space.npz',
         BASE/'learn_full_pool_joint_credit.py',BASE/'evaluate_learned_training_pool.py'],
        dict(wall_s=60,paths=[8,14],new_identity_routes=3),
        'Path8 peak guard discrepancies stayed near3e-6/mm while proposed update amplitude fell. '
        'Before attributing this to learning, recompute metrics from the actual saved rawXY and '
        'check exact parent execution on paths8/14, plus zero-added candidate execution on path8. '
        'This is a targeted consistency test prompted by non-scaling errors, not an experiment rerun '
        'or an adopted policy change. Save raw comparison and model/runtime bindings.')
    start=time.perf_counter();results=[]
    factory=checkpoint_environment_factory('gate_band_canonical_kernel')
    for path,zero in ((8,False),(8,True),(14,False)):
        old=next(r for r in baseline['records'] if r['path']==path)
        with np.load(old['trace']) as z:ref,rows,actions,obs=[z[k].copy() for k in ('reference','rows','actions','observations')]
        md=candidate_metadata(metadata,directions,np.zeros(len(directions))) if zero else metadata
        policy=load_credit_policy(md)
        saved_latent=np.array([policy.act(o,True)[0] for o in obs])
        decoded=np.array([factory(ref).decode_action(a) for a in saved_latent])
        saved_shape=complete_corner_metrics(ref,rows)
        costs=dict(actor_calls=0,physics_ticks=0)
        record,new_rows,new_actions,new_obs=rollout(policy,factory,ref,costs,start+58)
        tag=f'path{path}_'+('zero_added' if zero else 'parent')
        np.savez_compressed(stage/(tag+'.npz'),reference=ref,rows=new_rows,actions=new_actions,observations=new_obs)
        same_shape=new_rows.shape==rows.shape
        row_error=float(abs(new_rows-rows).max()) if same_shape else None
        first=None
        if same_shape:
            ids=np.flatnonzero(np.any(new_rows!=rows,axis=1));first=int(ids[0]+1) if len(ids) else None
        result=dict(tag=tag,success=record['success'],rows_bitwise=bool(same_shape and np.array_equal(new_rows,rows)),
            maximum_saved_decoded_action_error=float(abs(decoded-actions).max()),first_different_ms=first,
            maximum_row_error=row_error,old_peak=old['shape']['full_spatial_peak_curvature_mm_inv'],
            saved_raw_peak=saved_shape['full_spatial_peak_curvature_mm_inv'],
            actual_peak=record['shape']['full_spatial_peak_curvature_mm_inv'],record=record,
            model_sha256=sha(parent),saved_trace_sha256=sha(old['trace']))
        results.append(result);print('ZERO_IDENTITY', {k:v for k,v in result.items() if k!='record'},flush=True)
    write_json(stage/'results.json',dict(records=results,adopted=False,goal_complete=False))
    receipt(stage,dict(status='completed_current_zero_identity_check',wall_s=time.perf_counter()-start,
        all_bitwise=all(r['rows_bitwise'] for r in results),adopted=False,goal_complete=False))


if __name__=='__main__':main()
