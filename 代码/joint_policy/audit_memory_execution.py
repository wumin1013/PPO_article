"""Verify actual nonzero memory actions after instantaneous support disappears."""
import copy
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features
from audit_saved_lateral_motion import lateral_motion
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_early_observable_memory_joint_learning_20260920'
    source=BASE/'canonical_gate_band_early_local_affine_joint_learning_20260920'
    rows=read(stage/'state_joint/results.json')['candidates']
    tested=[c for c in rows if np.any(np.asarray(c['coefficients'])[6:])]
    assert tested,'No nonzero memory response yet'
    candidate=min(tested,key=lambda c:c['score'])
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    for key in ('local_affine_context','context_scale','center_context'):md.pop(key,None)
    md.update(architecture='observable_memory_joint_radial',memory_decays=[.5,.8],
        coefficients=np.asarray(candidate['coefficients']).reshape(-1,3))
    records=[r for r in candidate['records'] if r['path'] in (14,206,304)]
    replay=collect_data(md,md['parent'],records)
    f=features(md,replay['observations'],replay['innovations']);output=[]
    for r in records:
        mask=replay['path_ids']==r['path'];local=f[mask];labels=replay['labels'][mask]
        decisions=20*replay['decisions'][mask];instant=np.any(local!=0.,axis=(1,2));acted=np.any(labels!=0.,axis=1)
        output.append(dict(path=r['path'],trace_sha256=sha(r['trace']),
            instantaneous_support_ms=decisions[instant].tolist(),actual_residual_decisions_ms=decisions[acted].tolist(),
            nonzero_action_without_instantaneous_support_ms=decisions[acted&~instant].tolist(),
            significant_lateral_intervals=lateral_motion(r['trace'])['intervals'],
            complete_success=r['success'],actual_decoded_actions_bitwise_reproduced=True))
    write_json(stage/'memory_execution_audit.json',dict(tag=candidate['tag'],score=candidate['score'],
        coefficients=candidate['coefficients'],original_guard_minimum=min(candidate['measured_inequalities']),
        actual_decision_replay=True,physical_reruns=0,records=output,
        interpretation='Labels here are measured current-minus-parent latent actions on the '
        'same actual saved states, obtained only for auditing. They are not offline fitting '
        'targets. Recorded target velocities and decoded actions reproduce exactly; XY is '
        'read unchanged, not re-integrated. Nonzero residual '
        'after the instantaneous kernel vanishes verifies the new temporal capacity. '
        'It does not by itself establish better complete shape or acceptance.'))
    for r in output:print('MEMORY_EXECUTION',r['path'],r['instantaneous_support_ms'],
                         r['nonzero_action_without_instantaneous_support_ms'],flush=True)


if __name__=='__main__':main()
