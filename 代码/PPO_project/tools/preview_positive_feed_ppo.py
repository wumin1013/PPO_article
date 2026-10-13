"""Same PPO network, with a Gaussian feed logit executed through sigmoid.

Only feed sampling changes. PPO stores and scores its original Gaussian draw;
the environment transforms feed immediately before execution. Entropy and KL
are explicitly in latent coordinates. The old deterministic feed is preserved
when loading an earlier checkpoint; the physical sampling variance changes.
"""
import torch
import torch.nn.functional as F

from src.algorithms.preview_event_ppo import PreviewEventPolicy
from preview_macro_ppo import PreviewMacroPPO


class PositiveFeedPolicy(PreviewEventPolicy):
    def forward(self, x):
        x = self.ln1(F.elu(self.shared_fc1(x)))
        x = self.ln2(F.elu(self.shared_fc2(x)))
        x = self.ln3(F.elu(self.shared_fc3(x)))
        raw_mu = self.mu_head(x)
        feed_median = torch.sigmoid(raw_mu[:, 1:2])
        mu = torch.cat((torch.tanh(raw_mu[:, :1]), raw_mu[:, 1:2],
                        torch.sigmoid(raw_mu[:, 2:])), dim=1)
        original_std = torch.clamp(F.softplus(self.std_head(x)) + .001, max=2.)
        # First-order scale conversion at the median. The cap bounds logit
        # exploration; it does not impose a positive minimum physical feed.
        jacobian = (feed_median * (1. - feed_median)).clamp_min(1e-8)
        feed_std = (original_std[:, 1:2] / jacobian).clamp(max=2.)
        return mu, torch.cat((original_std[:, :1], feed_std, original_std[:, 2:]), dim=1)


class PositiveFeedPPO(PreviewMacroPPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            actor = PositiveFeedPolicy(self.state_dim, self.actor.shared_fc1.out_features,
                                       self.action_dim, std_mode='std_fix').to(self.device)
        actor.load_state_dict(self.actor.state_dict())
        rate = self.actor_optimizer.param_groups[0]['lr']
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=rate)

    def update(self, transition_dict):
        result = super().update(transition_dict)
        self.last_update_diagnostics.update(
            feed_sampling='Gaussian logit -> sigmoid at execution; stored actions remain Gaussian logits',
            entropy_coordinates='Latent Gaussian, shared by fixed and adaptive controls',
            feed_mu_and_std_coordinates='Logit; other action coordinates unchanged',
        )
        return result
