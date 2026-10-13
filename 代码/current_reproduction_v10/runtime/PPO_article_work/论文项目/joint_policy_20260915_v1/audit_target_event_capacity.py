"""Measure the current actor's attainable action range at actual target-jump states."""
import numpy as np
import torch
from joint_teacher_state_policy import OUTPUT_SCALE
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_context_coherent_parameter_training_20260920'
    targets=read(stage/'current_target_change_audit.json')['records'];records=[]
    for mode in ('geometry','dynamic'):
        model_file=stage/(mode+'_training_policy.pt');md=torch.load(model_file,map_location='cpu',weights_only=False)
        with np.load(stage/(mode+'_actor_replay.npz')) as z:replay={k:z[k].copy() for k in z.files}
        proof=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
        for p,(f,memory,latents,held) in proof.paths.items():
            t=next(r for r in targets if r['arm']==mode and r['path']==p)
            event=max((r for r in t['largest_changes'] if r['decision_boundary']),key=lambda r:abs(r['target_change_deg']))
            j=event['tick']//20
            design=np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
            bound=.03*np.sum(abs(design[j]),axis=0)
            low=GateBandRuntime.decode_action(latents[j]-bound);high=GateBandRuntime.decode_action(latents[j]+bound)
            width=high-low
            r=dict(arm=mode,path=p,decision=j,target_jump_deg=event['target_change_deg'],
                observed_preview_jump_mm=event['preview_change_mm'],
                coefficient_box_decoded_width=[float(np.degrees(width[0])),float(width[1]),float(width[2])],
                max_coefficient_latent_amplitude=bound.tolist(),
                event_design_norm_fraction_of_path_peak=float(np.linalg.norm(design[j])/max(np.max(np.linalg.norm(design,axis=(1,2))),1e-30)))
            records.append(r);print('TARGET_EVENT_CAPACITY',mode,p,round(r['event_design_norm_fraction_of_path_peak'],6),
                np.round(r['coefficient_box_decoded_width'],6).tolist(),flush=True)
    write_json(stage/'target_event_capacity_audit.json',dict(records=records,new_physical_runs=0,
        model_sha256={m:sha(stage/(m+'_training_policy.pt')) for m in ('geometry','dynamic')},
        scope='Exact saved-state per-head box range only; simultaneous action-trust limits make it an upper bound, '
        'not proof of target-angle reachability or of an irreducible physical limit.'))


if __name__=='__main__':main()
