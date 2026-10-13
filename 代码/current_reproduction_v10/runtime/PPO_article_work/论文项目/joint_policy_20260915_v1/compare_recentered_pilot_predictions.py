"""Compare old and recentered local credit on identical saved trajectory prefixes."""
import time
import numpy as np
import torch
from transition_waveform_model import chain_responses
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    fresh=BASE/'canonical_gate_band_shape_diagnostic_center_pilot_corrected_20260918'
    reference=BASE/'canonical_gate_band_shape_diagnostic_center_reference_20260918'
    old=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    failed=BASE/'canonical_gate_band_complete_completion_feedback_20260918'
    fresh_data,ref_data,old_data=[read(p/'results.json') for p in (fresh,reference,old)]
    assert read(fresh/'receipt.json')['status']=='completed_current_complete_transition_credit'
    assert sorted(r['path'] for r in fresh_data['records'])==[15,31]
    targets=ref_data['candidates'].copy();last=read(failed/'results.json')['candidates'][0]
    targets.append(dict(last,model=str((failed/last['model']).relative_to(BASE))))
    stage=register('canonical_gate_band_recentered_pilot_prediction_audit_20260918',
        [__file__,fresh/'results.json',fresh/'receipt.json',reference/'results.json',
         old/'results.json',old/'receipt.json',failed/'results.json',failed/'receipt.json',
         BASE/'transition_waveform_model.py']+[BASE/t['model'] for t in targets],
        dict(new_physics_ticks=0,new_actor_decisions=0,paths=[31,15],saved_policy_variants=5,wall_s=60),
        'Evaluate both original399a and new unadopted1919-center Jacobians against '
        'the identical saved third-policy ordinary prefixes. Bind exact checkpoint '
        'parameter differences; use only the common ordinary support, including '
        'the failed prefix as an error diagnostic and never completion. Report '
        'nominal399a separately because its own prediction is exactly centered. '
        'This measures whether relinearization addresses the observed state '
        'nonlinearity, not a prospective error bound, successful learning update '
        'or new complete-policy comparison. No model adoption.')
    start=time.perf_counter();records=[]
    center_policies={name:torch.load(BASE/d['parent_model'],map_location='cpu',weights_only=False)['policy']
                     for name,d in [('old399a',old_data),('new1919',fresh_data)]}
    for path in (31,15):
        locals={}
        for name,folder,d in [('old399a',old,old_data),('new1919',fresh,fresh_data)]:
            item=next(r for r in d['records'] if r['path']==path)
            with np.load(folder/item['file']) as z:locals[name]={k:z[k].copy() for k in z.files}
        assert np.array_equal(locals['old399a']['reference'],locals['new1919']['reference'])
        for target in targets:
            j=next(j for j,r in enumerate(target['results']) if r['original_path']==path)
            with np.load(BASE/target['traces'][j]) as z:truth=z['rows'].copy()
            brake=np.flatnonzero(truth[:,12]);length=min(int(brake[0]) if len(brake) else len(truth),
                *(int(np.flatnonzero(local['rows'][:,12])[0]) for local in locals.values()))
            policy=torch.load(BASE/target['model'],map_location='cpu',weights_only=False)['policy']
            row=dict(path=path,target_model=target['model'],target_path_complete=target['results'][j]['success'],
                compared_ordinary_ticks=length,predictions={})
            for name,local in locals.items():
                center=center_policies[name]
                assert set(center)==set(policy)
                assert all(torch.equal(center[k],policy[k]) for k in center if k!='residual_coefficients')
                direction=(policy['residual_coefficients'].double()-center['residual_coefficients'].double()).numpy().ravel()
                response=chain_responses(local,direction[None,:])[:,:,0]
                error=truth[:length,:6]-local['rows'][:length,:6]-response[:length]
                xy=np.linalg.norm(error[:,:2],axis=1)
                row['predictions'][name]=dict(max_xy_error_mm=float(xy.max()),rms_xy_error_mm=float(np.sqrt((xy*xy).mean())),
                    last_xy_error_mm=float(xy[-1]),max_velocity_error_mm_s=float(np.linalg.norm(error[:,2:4],axis=1).max()))
            records.append(row)
    assert time.perf_counter()-start<60
    write_json(stage/'results.json',dict(records=records,new_center_sha256=fresh_data['parent_sha256'],
        new_center_adopted=False,goal_complete=False))
    receipt(stage,dict(status='completed_matched_recentered_prediction_audit',new_physics_ticks=0,
        new_actor_decisions=0,wall_s=time.perf_counter()-start,goal_complete=False))
    for r in records:print('RECENTERED_PREDICTION',r,flush=True)


if __name__=='__main__':main()
