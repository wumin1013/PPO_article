"""Single sampling change: cap feed-logit exploration at a registered scale."""
import torch

from preview_positive_feed_ppo import PositiveFeedPolicy
import preview_coherent_distance as coherent


class BoundedFeedPolicy(PositiveFeedPolicy):
    def __init__(self,*args,feed_logit_std_cap=.3,**kwargs):
        if not 0 < feed_logit_std_cap <= 2:
            raise ValueError('Feed exploration cap must be in(0,2]')
        super().__init__(*args,**kwargs)
        self.feed_logit_std_cap=float(feed_logit_std_cap)

    def forward(self,states):
        mu,std=super().forward(states)
        return mu,torch.cat((std[:,:1],std[:,1:2].clamp(max=self.feed_logit_std_cap),std[:,2:]),dim=1)


def make_agent(cfg,env):
    agent=coherent.make_agent(cfg,env)
    cap=cfg['revision']['feed_exploration']['logit_std_cap']
    with torch.random.fork_rng(devices=[]):
        actor=BoundedFeedPolicy(agent.state_dim,agent.actor.shared_fc1.out_features,
            agent.action_dim,std_mode='std_fix',feed_logit_std_cap=cap).to(agent.device)
    actor.load_state_dict(agent.actor.state_dict())
    rate=agent.actor_optimizer.param_groups[0]['lr']
    agent.actor=actor
    agent.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=rate)
    return agent
