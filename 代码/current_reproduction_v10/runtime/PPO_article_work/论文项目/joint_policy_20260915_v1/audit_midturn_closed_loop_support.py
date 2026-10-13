"""Read actual feedback support after bounded joint parameter exploration."""
import copy
import time
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features
from localize_joint_shape_residual import characterize
from study_io import BASE,write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1);started=time.perf_counter()
    stage=BASE/'canonical_gate_band_bounded_midturn_joint_learning_20260920'
    model=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    history=read(stage/'state_joint/results.json')['candidates'];records=[]
    for c in history:
        md=copy.deepcopy(model);md['coefficients']=np.asarray(c['coefficients']).reshape(-1,3)
        replay=collect_data(md,md['parent'],c['records'],include_latents=True,deadline=started+55)
        np.savez_compressed(stage/(c['tag']+'_replayed_inputs.npz'),**replay)
        f=features(md,replay['observations'],replay['innovations']);r=c['records'][0]
        with np.load(r['trace']) as z:actions=z['actions'].copy()
        support=[]
        for k in (3,7):
            ids=np.flatnonzero(np.any(f[:,k]!=0.,axis=1))
            support.append(dict(kernel=k,active_ms=(replay['decisions'][ids]*20).tolist(),
                activation=abs(f[ids,k]).max(axis=1).tolist()))
        row=dict(tag=c['tag'],support=support,
            actions=[dict(time_ms=d*20,held=actions[d].tolist(),latent_residual=replay['labels'][d].tolist(),
                original_parent_latent=replay['parent_latents'][d].tolist()) for d in range(15,20)],
            shape=r['shape'],localization=characterize(r['trace']))
        records.append(row)
        print('ACTUAL_SUPPORT',c['tag'],support,flush=True)
    write_json(stage/'actual_support_audit.json',dict(records=records,decoded_actions_bitwise=True,
        new_physical_runs=0,wall_s=time.perf_counter()-started,
        interpretation='Measured support and parent feedback at saved actual states. '
        'A changed activation is a mechanism observation, not by itself an isolated causal proof.'))


if __name__=='__main__':main()
