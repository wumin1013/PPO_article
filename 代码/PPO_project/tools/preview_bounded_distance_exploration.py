"""Cap normalized Gaussian D exploration while preserving all mean weights."""
import torch

import preview_bounded_feed_exploration as bounded


class BoundedDistancePolicy(bounded.BoundedFeedPolicy):
    def __init__(self, *args, distance_normalized_std_cap=.05, **kwargs):
        if not 0 < distance_normalized_std_cap <= 2:
            raise ValueError('Normalized distance exploration cap must be in(0,2]')
        super().__init__(*args, **kwargs)
        self.distance_normalized_std_cap = float(distance_normalized_std_cap)

    def forward(self, states):
        mu, std = super().forward(states)
        distance_std = std[:, 2:3].clamp(max=self.distance_normalized_std_cap)
        return mu, torch.cat((std[:, :2], distance_std), dim=1)


def make_agent(cfg, env):
    agent = bounded.make_agent(cfg, env)
    cap = cfg['revision']['distance_exploration']['normalized_std_cap']
    # Extra network construction must not change matched Gaussian RNG streams.
    with torch.random.fork_rng(devices=[]):
        actor = BoundedDistancePolicy(agent.state_dim, agent.actor.shared_fc1.out_features,
            agent.action_dim, std_mode='std_fix',
            feed_logit_std_cap=agent.actor.feed_logit_std_cap,
            distance_normalized_std_cap=cap).to(agent.device)
    actor.load_state_dict(agent.actor.state_dict())
    rate = agent.actor_optimizer.param_groups[0]['lr']
    agent.actor = actor
    agent.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rate)
    return agent
