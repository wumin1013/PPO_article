"""Prepare an exact paired action-removal experiment from each own-seed actor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import shutil
from datetime import datetime,timezone
import numpy as np
import torch
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from replay_lowdim_training_stage import SUPPLEMENT,SEEDS
from research_policy_loader import from_saved
from direct_two_action_control import DirectTwoPolicy
from localized_two_action_policy import LocalizedTwoPolicy


def main():
    torch.set_num_threads(1)
    verify_bound_inputs()
    stage=SUPPLEMENT/'paired_action_ablation'
    stage.mkdir(exist_ok=False)
    source=ROOT/'direct14_dictionary02_resume_round73'
    binding={}
    xs=[]
    for case in ['original','dense']:
        for key in ['1ms','half_ms']:
            p=source/f'eval_0000_{case}_{key}.npz'
            binding[str(p)]=sha(p)
            with np.load(p) as d: xs.append(d['observations'][::4 if case=='original' else 1].copy())
    x=torch.tensor(np.vstack(xs),dtype=torch.float32)
    records=[]
    for seed in SEEDS:
        parent=SUPPLEMENT/'independent_training'/f'seed_{seed}/local_features/checkpoint_0004.pt'
        saved=torch.load(parent,map_location='cpu',weights_only=False)
        binding[str(parent)]=sha(parent)
        main_policy=from_saved(saved,seed)
        base=DirectTwoPolicy(seed)
        three_state={**main_policy.state_dict()}
        base_state=base.state_dict()
        for k in base_state:
            if k.startswith('actor.'):
                value=main_policy.actor.backbone.backbone.state_dict()[k[len('actor.'):]]
                if k in ['actor.4.weight','actor.4.bias']:value=value[:2]
            elif k=='log_std':value=main_policy.log_std[:2]
            else:value=three_state[k]
            assert value.shape==base_state[k].shape
            base_state[k]=value.clone()
        base.load_state_dict(base_state)
        two=LocalizedTwoPolicy(seed,count=192)
        two.initialize(base)
        with torch.no_grad():
            for name in ['centers','feature_scale','radius']:
                getattr(two.actor,name).copy_(getattr(main_policy.actor,name))
            two.actor.correction.weight.copy_(main_policy.actor.correction.weight[:2])
            assert torch.equal(main_policy.actor(x)[:,:2],two.actor(x))
        for row in x.numpy()[::257]:
            assert np.array_equal(main_policy.act(row,True)[0][:2],two.act(row,True)[0])
        paths={}
        for arm,policy in [('three',main_policy),('two',two)]:
            folder=stage/f'seed_{seed}'/arm
            folder.mkdir(parents=True)
            payload=dict(saved)
            payload.update(policy=policy.state_dict(),interface=[14,3 if arm=='three' else 2,seed],
                updates=0,new_ppo_updates=0,ppo_updates_this_round=0,
                actor_outputs=3 if arm=='three' else 2,
                parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),
                paired_arm=arm,training_kind='Exact paired action-removal initialization; zero new training')
            if arm=='two':
                payload.update(policy_class='LocalizedTwoPolicy',local_basis_count=192,preview_rule_mm=3.)
            path=folder/'initial.pt'
            torch.save(payload,path)
            paths[arm]=dict(path=str(path),sha256=sha(path))
        records.append(dict(seed=seed,parent=str(parent),parent_sha256=sha(parent),arms=paths,
            initial_steering_and_feed_bitwise_equal=True,verified_observations=len(x)))
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),seeds=list(SEEDS),
        source_sha256=binding,initializations=records,
        shared_inputs=14,policy_memory_scalars=0,action_outputs=[3,2],
        physical=dict(corridor_mm=1.25,vector_limits=[100,2000,62500],decision_s=.02,
                      integration_s=[.001,.0005],rest_s=.02,cruise_threshold_mm_s=99),
        fixed_two_action_preview_mm=3.,
        preview_selection='3mm was selected in the previous valid two-action development; fixed before this paired experiment. Previous stronger valid two-action records remain comparison references.',
        shared_contexts=str(source),shared_context_index=0,
        rounds=[dict(updates=24,batch_size=2048),dict(updates=24,batch_size=2048)],
        checkpoints_for_comparison=[0,24,48],
        decision_mix=dict(dense=.75,original=.25),
        actor_epochs=6,critic_epochs=32,minibatch=256,kl_limit=.01,
        auxiliary_target_loss_weight=0.,
        auxiliary_scope='Existing sampler may record initializer targets for auditing; weight is zero in both arms to permit adaptation after removing the action.',
        matched_actor_training='All remaining actor coefficients train in both arms. Remove only preview output and its96 preview-only features; corresponding192 steering/feed features and initial outputs are identical.',
        selection='Among endpoints0/24/48 passing all four complete physical checks, minimize the maximum normalized case time across both resolutions, using the same S2 denominators; tie-break by mean normalized time then earlier endpoint. Failures stay in the denominator. No failed-seed replacement.',
        independence_scope='Three paired continuations from independently initialized upstream students, conditional on shared historical corpus and fitted feature maps. It does not retroactively equalize historical method discovery.',
        purpose='Isolate action removal plus equal-budget adaptation, while assessing closed-loop stability of both arms.',
        adoption='All main seeds must pass complete constraints and unchanged all34/time criteria; compare with paired and stronger historical baselines. A failed two-action arm cannot establish a speedup percentage.',
        wall_limit_per_round_s=1800,no_generalization_claim=True,selected_paper_model_unchanged=True)
    write(stage/'protocol.json',protocol)
    shutil.copy2(__file__,stage/'prepare_paired_supplement.py')
    print('PAIRED_INITIALIZATIONS',records,flush=True)


if __name__=='__main__':main()
