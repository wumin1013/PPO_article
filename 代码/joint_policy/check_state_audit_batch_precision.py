"""Resolve tiny batched-output discrepancies in the saved-state audit."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_shape_cruise_statewise_half_full_20260917'
    data=read(parent/'results.json');model=parent/data['chosen']['model']
    stage=register('canonical_gate_band_state_audit_precision_check_20260917',
        [__file__,model,BASE/'canonical_gate_band_current_state_dependencies_20260917/results.json'],
        dict(paths=[21,63],maximum_actor_queries=2500,new_physics_ticks=0,wall_s=30),
        'Resolve0.24-0.48micro-unit differences between batch lanes in the '
        'previous history/frontier action checks. Compare scalar calls with '
        'changed masked history, and compare identical observations in both '
        'batch lanes. Preserve the prior exact-equality failures. This check '
        'cannot establish the sole cause of earlier derivative-scale errors.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,actor_queries=0)
    meta=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(meta['policy'])
    factory=checkpoint_environment_factory(meta['architecture']);records=[]
    with torch.no_grad():
        for path in (21,63):
            j=next(j for j,r in enumerate(data['full_candidate']['results']) if r['original_path']==path)
            with np.load(BASE/data['full_candidate']['traces'][j]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            states=[];ordinary=int(np.flatnonzero(rows[:,12])[0])
            env=reconstruct(ref,rows,actions,ordinary,costs,state_snapshots=states,env_factory=factory)
            scalar_max=0.;identical_batch_max=0.;history_vs_identical_max=0.
            for state in states:
                env.restore(state);obs=env.observation();other=obs.copy();other[8:11]=[-.91,.97,2.3]
                scalar=policy.distribution(torch.tensor(obs)).mean.numpy()
                changed=policy.distribution(torch.tensor(other)).mean.numpy()
                identical_batch=policy.distribution(torch.tensor(np.stack([obs,obs]))).mean.numpy()
                history_batch=policy.distribution(torch.tensor(np.stack([obs,other]))).mean.numpy()
                costs['actor_queries']+=6
                scalar_max=max(scalar_max,float(abs(changed-scalar).max()))
                identical_batch_max=max(identical_batch_max,float(abs(identical_batch[1]-identical_batch[0]).max()))
                history_vs_identical_max=max(history_vs_identical_max,float(abs(identical_batch-history_batch).max()))
                assert costs['actor_queries']<=2500 and time.perf_counter()-start<30
            records.append(dict(path=path,scalar_history_error=scalar_max,
                identical_batch_lane_error=identical_batch_max,history_vs_identical_batch_error=history_vs_identical_max))
    assert all(r['scalar_history_error']==0 and r['history_vs_identical_batch_error']==0 for r in records)
    write_json(stage/'results.json',dict(records=records,costs=costs,masked_history_dependence_found=False,
        original_exact_batch_lane_failures_explained=True,global_state_sufficiency_claim=False,
        earlier_derivative_scale_unique_cause_claim=False,goal_complete=False))
    receipt(stage,dict(status='verified_scalar_history_independence_and_batch_lane_roundoff',costs=costs,wall_s=time.perf_counter()-start))
    print('BATCH_PRECISION_CHECK',records,flush=True)


if __name__=='__main__':main()
