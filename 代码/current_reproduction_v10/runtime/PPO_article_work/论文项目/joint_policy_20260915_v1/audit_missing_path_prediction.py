"""Check the newly collected path19 derivative against an existing full rollout."""
import time
import numpy as np
from transition_waveform_model import chain_responses
from audit_completion_event_changes import terminal_query
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    credit=BASE/'canonical_gate_band_missing_path19_credit_20260918'
    feedback=BASE/'canonical_gate_band_shape_priority_repaired_half_feedback_20260918'
    proposal=BASE/'canonical_gate_band_shape_priority_nearest_feasible_repair_20260918'
    meta=read(credit/'results.json');candidate=read(feedback/'results.json')['full_candidate']
    item=next(r for r in meta['records'] if r['path']==19);local_file=credit/item['file']
    j=next(j for j,r in enumerate(candidate['results']) if r['original_path']==19)
    trace=BASE/candidate['traces'][j]
    stage=register('canonical_gate_band_missing_path19_prediction_audit_20260918',
        [__file__,credit/'results.json',credit/'receipt.json',local_file,trace,
         feedback/'results.json',feedback/'receipt.json',proposal/'direction.npz',
         BASE/'transition_waveform_model.py',BASE/'audit_completion_event_changes.py'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=30,terminal_planner_queries=1),
        'Before adding the missing path model to learning constraints, test it '
        'against the already completed rejected half-step. Predict only the '
        'ordinary interval supported by the399a trace; explicitly exclude its '
        'later candidate ordinary decision and brake from claimed prediction '
        'accuracy. Compare the predicted and actual position lag at the original '
        'brake tick and query the original planner there. This is a held saved '
        'response check for the new local model, not independent generalization '
        'or a uniform error bound. No duplicate rollout or derivative collection.')
    start=time.perf_counter()
    with np.load(local_file) as z:local={k:z[k].copy() for k in z.files}
    with np.load(proposal/'direction.npz') as z:direction=.5*z['direction'].copy().ravel()
    with np.load(trace) as z:actual=z['rows'].copy();assert np.array_equal(z['reference'],local['reference'])
    nominal=local['rows'];stop=int(np.flatnonzero(nominal[:,12])[0])
    response=chain_responses(local,direction[None,:])[:,:,0]
    predicted=nominal[:,:6]+response
    error=np.linalg.norm(predicted[:stop,:2]-actual[:stop,:2],axis=1)
    query=terminal_query(predicted[stop-1,:6],local['reference'][-1])
    result=dict(parent_sha256=meta['parent_sha256'],path=19,supported_ordinary_ticks=stop,
        max_ordinary_xy_error_mm=float(error.max()),terminal_state_xy_error_mm=float(error[-1]),
        predicted_terminal_position_change_mm=response[stop-1,:2].tolist(),
        actual_terminal_position_change_mm=(actual[stop-1,:2]-nominal[stop-1,:2]).tolist(),
        predicted_original_brake_query=query,later_tail_prediction_claimed=False,
        prospective_error_guarantee=False,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_missing_path_saved_prediction_check',new_physics_ticks=0,
        new_actor_decisions=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('MISSING_PATH_PREDICTION',result,flush=True)


if __name__=='__main__':main()
