"""Zero correction initialization of a matching-capacity two-action model."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from research_policy_loader import from_saved
from localized_two_action_policy import LocalizedTwoPolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--seed',type=int,default=922401);a=p.parse_args()
    assert Path(a.name).name==a.name
    verify_bound_inputs();torch.set_num_threads(1)
    stage=ROOT/a.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    parent=ROOT/'direct14_actual_two_action_ppo_round45/checkpoint_0008.pt'
    saved=torch.load(parent,map_location='cpu',weights_only=False);base=from_saved(saved)
    sources=[Path(__file__)]+[ROOT/n for n in ['localized_two_action_policy.py',
        'train_localized_two_action_ppo.py','train_mixed_resolution_ppo.py','localized_preview_head.py']]
    bindings={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    arrays=[];trace_hashes={}
    for case in ['dense','original']:
        for key in ['1ms','half_ms']:
            path=parent.parent/f'eval_0008_{case}_{key}.npz'
            with np.load(path) as d:x=d['observations'].copy()
            # Preserve all dense states; subsample repeated long-path cruise.
            arrays.append(x[::4] if case=='original' else x)
            trace_hashes[str(path)]=sha(path)
    x=np.vstack(arrays).astype(np.float32);scale=x.std(0).clip(.03);z=x/scale
    centers=[int(np.argmax(np.linalg.norm(z,axis=1)))];distance=((z-z[centers[0]])**2).sum(1)
    for _ in range(191):
        idx=int(np.argmax(distance));centers.append(idx);distance=np.minimum(distance,((z-z[idx])**2).sum(1))
    policy=LocalizedTwoPolicy(a.seed);policy.initialize(base)
    with torch.no_grad():
        policy.actor.centers.copy_(torch.tensor(x[centers]))
        policy.actor.feature_scale.copy_(torch.tensor(scale));policy.actor.radius.fill_(1.2)
        inp=torch.tensor(x);assert torch.equal(policy.actor(inp),base.actor(inp))
    for row in x[::257]:
        assert np.array_equal(policy.act(row,True)[0],base.act(row,True)[0])
    payload=dict(saved);payload.update(policy=policy.state_dict(),policy_class='LocalizedTwoPolicy',local_basis_count=192,
        interface=[14,2,a.seed],updates=0,new_ppo_updates=0,ppo_updates_this_round=0,
        parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),training_kind='Zero local-head initialization;no training',
        local_feature_scope='Same192basis structure as main steering/feed correction;centers from own observed execution states;preview action absent',
        observation_dimension=14,actor_outputs=2,policy_memory_scalars=0)
    torch.save(payload,stage/'initial.pt')
    np.savez_compressed(stage/'feature_initialization.npz',centers=x[centers],scale=scale)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Two-action corresponding-capacity initialization',parent=str(parent),parent_sha256=sha(parent),
        source_sha256=bindings,trace_sha256=trace_hashes,inputs=14,outputs=2,policy_memory_scalars=0,
        local_basis_count=192,radius=1.2,output_scale=[.01,.08],all_initial_actions_bitwise_equal=True,
        no_additional_preview_head='The main96basis preview-only layer has no steering/feed effect and is removed with the preview action.',
        trainable_actor='Own MLP and two-output local correction;all actor weights can update in subsequent PPO',
        intended_training='One conditional seed;48updates2048decisions,75percentdense,25percentoriginal,two physical dt;scheduled checks8updates;<=30min wall',
        fairness_scope='Corresponding steering/feed structure and further training opportunity;historical initialization and budget still differ,not a from-scratch matched-seed experiment.',
        new_ppo_updates=0,goal_complete=False,method_frozen=False))
    print('TWO_ACTION_CAPACITY_INITIALIZED',stage/'initial.pt','exact_identity',True,flush=True)


if __name__=='__main__':main()
