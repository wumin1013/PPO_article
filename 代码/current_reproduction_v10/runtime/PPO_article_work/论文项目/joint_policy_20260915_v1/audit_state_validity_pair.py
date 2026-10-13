"""Matched first-round control for the proposed state-validity constraint."""
import time
import numpy as np
from transition_waveform_model import chain_responses
from probe_closed_loop_tangent_credit import STATE_SCALE,ACTION_SCALE
from joint_runtime import wrap
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    arms=[('without_state_cut','canonical_gate_band_completion_first_round_control_20260918',
           'canonical_gate_band_completion_first_round_feedback_20260918'),
          ('with_state_cut','canonical_gate_band_state_validity_proposal_checked_20260918',
           'canonical_gate_band_state_validity_feedback_20260918')]
    credit=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    data=read(credit/'results.json');item=next(r for r in data['records'] if r['path']==31)
    stage=register('canonical_gate_band_state_validity_matched_audit_20260918',
        [__file__,credit/item['file']]+[BASE/f/n for _,p,f in arms for n in ('results.json','receipt.json')]+
        [BASE/p/'direction.npz' for _,p,f in arms],
        dict(new_physics_ticks=0,new_actor_decisions=0,matched_candidates=2,wall_s=60),
        'Compare the first conditional solve with and without the one state-validity '
        'cut, at the same anchor/objective/completion witness and refined distance '
        'admission. Reuse the previously saved no-cut first-round direction, not '
        'the later third-round failure, to avoid attributing branch-relinearization '
        'effects to the new cut. Evaluate real shape, complete time/cruise and '
        'predicted-versus-actual state deviation/error. No new samples or adoption.')
    start=time.perf_counter()
    with np.load(credit/item['file']) as z:local={k:z[k].copy() for k in z.files}
    sj_response=np.zeros((7,768))
    for k in range(25):
        direct=np.zeros((3,768))
        for h in range(3):direct[h,h*256:(h+1)*256]=local['features'][k,h]/ACTION_SCALE[h]
        sj=local['state_jacobian'][k];sj_response=sj[:,:7]@sj_response+sj[:,7:]@direct
    results=[]
    for label,proposal,feedback in arms:
        record=read(BASE/feedback/'results.json')['candidates'][0]
        with np.load(BASE/proposal/'direction.npz') as z:direction=z['direction'].copy().ravel()
        j=next(j for j,r in enumerate(record['results']) if r['original_path']==31)
        with np.load(BASE/record['traces'][j]) as z:actual=z['rows'].copy()
        nominal=local['rows'][499]
        shift=np.r_[(actual[499,:6]-nominal[:6])/STATE_SCALE[:6],
            wrap(np.arctan2(actual[499,9],actual[499,8])-np.arctan2(nominal[9],nominal[8]))]
        response=chain_responses(local,direction[None,:])[:,:,0]
        length=min(int(np.flatnonzero(actual[:,12])[0]),int(np.flatnonzero(local['rows'][:,12])[0]))
        error=actual[:length,:2]-local['rows'][:length,:2]-response[:length,:2]
        results.append(dict(label=label,summary={k:record['summary'][k] for k in
            ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s','complete_cruise_s')},
            checks=record['checks'],actual_state_shift_norm=float(np.linalg.norm(shift)),
            predicted_state_shift_norm=float(np.linalg.norm(sj_response@direction)),
            path31_max_ordinary_xy_prediction_error_mm=float(np.linalg.norm(error,axis=1).max()),
            path31_last_ordinary_xy_prediction_error_mm=float(np.linalg.norm(error[-1]))))
    write_json(stage/'results.json',dict(records=results,
        conclusion='Both matched first-round candidates complete29/29. The state cut cannot be credited with restoring success relative to the unmatched third-round failure. Both still fail existing local shape criteria and add no99mm/s cruise. Do not expand this cut alone as a demonstrated solution.',goal_complete=False))
    receipt(stage,dict(status='completed_matched_state_validity_comparison',new_physics_ticks=0,
        new_actor_decisions=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('STATE_VALIDITY_MATCHED',results,flush=True)


if __name__=='__main__':main()
