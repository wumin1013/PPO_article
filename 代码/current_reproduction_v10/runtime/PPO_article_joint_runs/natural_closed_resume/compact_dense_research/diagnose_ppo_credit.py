"""Offline reward/credit diagnostic on saved on-policy batches; zero simulation."""
import json
from pathlib import Path
import numpy as np
import torch
from compact_control import CompactPolicy
from joint_policy import advantages

ROOT=Path(__file__).resolve().parent
stage=ROOT/'ppo14_action3_allow_seed922301_pilot'
total=float(np.linalg.norm(np.diff(np.loadtxt(ROOT/'dense_closed_case/reference.csv',delimiter=',',skiprows=1),axis=0),axis=1).sum())
torch.set_num_threads(1);records=[];previous_progress=0.;previous_done=False
for path in sorted(stage.glob('batch_*.npz')):
    index=int(path.stem.split('_')[-1])
    checkpoint=stage/('initial.pt' if index==1 else f'checkpoint_{index-1:04d}.pt')
    if not checkpoint.exists():break
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    policy=CompactPolicy();policy.load_state_dict(saved['policy'])
    b=np.load(path);x=torch.as_tensor(b['observation']);latent=b['latent']
    with torch.no_grad():mean=policy.actor(x).numpy();std=policy.distribution(x).scale.numpy()
    score=(latent-mean)/(std*std)
    adv,_=advantages(b['reward'],b['value'],b['next_value'],b['terminal'])
    progress=b['progress_mm'];done=b['terminal']
    before=np.r_[0. if previous_done else previous_progress, np.where(done[:-1],0.,progress[:-1])]
    delta=(~done)*(progress-total)-(before-total)
    shaped=b['reward']+(1/30.-1/100.)*delta
    alternative,_=advantages(shaped,b['value'],b['next_value'],done)
    centered=adv-adv.mean();alt_centered=alternative-alternative.mean()
    records.append(dict(batch=index,n=len(adv),reward_mean=float(b['reward'].mean()),reward_std=float(b['reward'].std()),
        advantage_std=float(adv.std()),actor_bias_score_gradient=(score*centered[:,None]/(adv.std()+1e-8)).mean(0).tolist(),
        potential30_offline_reward_mean=float(shaped.mean()),potential30_offline_reward_std=float(shaped.std()),
        potential30_offline_advantage_std=float(alternative.std()),
        potential30_offline_bias_gradient=(score*alt_centered[:,None]/(alternative.std()+1e-8)).mean(0).tolist(),
        progress_step_q=np.quantile(delta,[0,.01,.5,.99,1]).tolist(),
        requested_feed_mean=float(b['held_action'][:,1].mean()),
        mean_latent_feed=float(mean[:,1].mean()),stochastic_completed=int(done.sum())))
    previous_progress=progress[-1];previous_done=done[-1]
result=dict(kind='Offline on-policy credit diagnostic; no new physical rollout or optimization',batches=records,
    interpretation='Alternative potential uses30mm/s only as reward shaping. For every complete episode the summed reward differs by the same L*(1/30-1/100), hence preserves full-time ordering. Offline gradients use the old critic and do not establish improvement.',
    physics_ticks=0,ppo_updates=0)
(ROOT/'pilot_credit_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(records[-5:],indent=2))
