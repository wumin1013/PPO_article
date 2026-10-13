"""Relate measured shared-action tradeoffs to observed motion-state differences."""
import numpy as np
import torch
from joint_teacher_state_policy import features
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_early_joint_exploration_restoration_20260920'
    md=torch.load(stage/'unaccepted_exploration_candidate.pt',map_location='cpu',weights_only=False)
    with np.load(stage/'replayed_states.npz') as z:data={k:z[k].copy() for k in z.files}
    f=features(md,data['observations'],data['innovations']);speed=100*np.linalg.norm(data['observations'][:,2:4],axis=1)
    b={r['path']:r for r in read(stage/'baseline.json')['baseline']['records']}
    actual={r['path']:r for r in read(stage/'exploration_result.json')['best']['records']}
    rows=[]
    for path in (14,104,105,204,206,304):
        mask=data['path_ids']==path;local=f[mask,:,1];indices=np.argmax(local,axis=0)
        r=dict(path=path,maximum_kernel_by_center=np.max(local,axis=0).tolist(),
            corresponding_speed_mm_s=speed[mask][indices].tolist(),
            corresponding_forward_gate_mm=(30*data['observations'][mask,11][indices]).tolist(),
            corresponding_decision_ms=(20*data['decisions'][mask][indices]).tolist(),
            curvature_change=actual[path]['shape']['full_spatial_peak_curvature_mm_inv']-b[path]['shape']['full_spatial_peak_curvature_mm_inv'],
            lateral_change=actual[path]['shape']['reverse_lateral_total_mm']-b[path]['shape']['reverse_lateral_total_mm'])
        rows.append(r);print('EARLY_MOTION_CONTEXT',r,flush=True)
    write_json(stage/'motion_response_context_audit.json',dict(records=rows,physical_reruns=0,
        interpretation='Current radial heads share a fixed coefficient at all supported states. '
        'The saved action update has opposing curvature/lateral effects across different '
        'observed speeds. This motivates testing learned local affine state modulation, '
        'but does not prove speed alone explains the conflict or guarantees a feasible update.'))


if __name__=='__main__':main()
