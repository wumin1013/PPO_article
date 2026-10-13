"""Independent local checks of the request likelihood and physical-time GAE."""
import json

import numpy as np
import torch

from common import ROOT, sha, utc, write_json
from safe_ppo_interface_v7 import from_fit, physical_time_gae, SampleRequest
from strong_policy_v4 import load_strong_policy


def main():
    torch.set_num_threads(1)
    source=ROOT/'results/recovery_training_v6/training_data/supervision.npz'
    with np.load(source) as data:
        observation=data['observations'][::max(1,len(data['observations'])//16)][:16].copy()
    x=torch.tensor(observation)
    report=dict(created_utc=utc(),scope='Independent interface software audit; no physical rollout or training',
        interface_sha256=sha(ROOT/'src/safe_ppo_interface_v7.py'), arms={})
    for arm in ('adaptive','fixed'):
        path=ROOT/f'results/recovery_training_v6/seed101_{arm}/final.pt'
        policy,_=from_fit(path)
        fitted,_=load_strong_policy(path)
        assert torch.equal(policy.mean(x),fitted.mean(x))
        distribution=policy.dist(x)
        request=(distribution.mean+.5*distribution.scale).detach()
        policy.zero_grad()
        logp=distribution.log_prob(request).sum()
        logp.backward()
        bias=policy.actor.base[-1].bias
        actual=float(bias.grad[1])
        original=bias.detach().clone()
        values=[];epsilon=.0005
        for delta in (-epsilon,epsilon):
            with torch.no_grad():
                bias.copy_(original);bias[1]+=delta
                values.append(float(policy.dist(x).log_prob(request).sum()))
        with torch.no_grad():bias.copy_(original)
        numerical=(values[1]-values[0])/(2*epsilon)
        relative=abs(actual-numerical)/max(1.,abs(actual),abs(numerical))
        assert relative<.01,(arm,actual,numerical,relative)
        sample,saved_logp,_=policy.act(observation[0])
        with torch.no_grad():
            replay_logp=float(policy.dist(x[0]).log_prob(torch.tensor(sample)).sum())
        assert replay_logp==saved_logp
        provider=SampleRequest(arm)
        provider.begin(observation[0],sample)
        request=provider.mean(observation[0]).numpy()
        assert np.array_equal(request[:policy.action_dim],sample)
        if arm=='fixed':
            assert abs((6.4+5.6*request[2])-3.)<1e-12
        rejected=False
        try:provider.mean(observation[0])
        except AssertionError:rejected=True
        assert rejected
        report['arms'][arm]=dict(deterministic_fit_mean_exact=True,original_request_likelihood_exact=True,
            sampled_request_dimension=policy.action_dim,executor_request_dimension=len(request),
            duplicate_request_rejected=True,gradient_feed_bias_autograd=actual,
            gradient_feed_bias_central_difference=numerical,gradient_relative_error=relative,
            checkpoint_sha256=sha(path))
    # Explicit hand-calculated episode with a 100 ms stop and 20 ms command.
    rows=[dict(path='a',episode=0,tick_before=0,tick_after=100,elapsed_s=.1,
              reward=-.1,value=1.,next_value=2.,terminated=False,truncated=False),
          dict(path='b',episode=0,tick_before=0,tick_after=20,elapsed_s=.02,
              reward=-.02,value=4.,next_value=6.,terminated=False,truncated=True),
          dict(path='a',episode=0,tick_before=100,tick_after=120,elapsed_s=.02,
              reward=-.02,value=2.,next_value=999.,terminated=True,truncated=False)]
    advantage,returns=physical_time_gae(rows)
    expected=np.array([.9+(.95**5)*(-2.02),1.98,-2.02])
    error=float(abs(advantage-expected).max())
    assert error<1e-6
    # Failure terminal has phi=0; shaping cancels over the whole episode.
    phi=[-2.,-.5,0.];times=[.1,.02];penalties=[0.,8.]
    reward=[-dt+phi[i+1]-phi[i]-penalties[i] for i,dt in enumerate(times)]
    assert abs(sum(reward)-(-sum(times)-phi[0]-sum(penalties)))<1e-12
    report.update(status='passed',physical_time_gae_max_error=error,
        terminal_bootstrap_ignored=True,external_truncation_bootstrap_retained=True,
        long_stop_eligibility_uses_physical_time=True,episodic_reward_telescopes=True,
        limitations='This software audit does not establish real execution completion, efficiency, or training benefit.')
    write_json(ROOT/'audit/safe_ppo_interface_v7_independent_review.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
