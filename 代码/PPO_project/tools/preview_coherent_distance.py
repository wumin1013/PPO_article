"""Temporally coherent learned preview with event-masked PPO likelihood.

Feed remains a20ms macro action. A new D is sampled every registered period
and then held by the existing physical environment. All fixed/adaptive actors
observe the same remaining-period phase in the formerly zero state slot21.
"""
from __future__ import annotations

import copy
import numpy as np
import torch

import preview_local_corner_study as local
from preview_positive_feed_ppo import PositiveFeedPPO
from src.algorithms.preview_event_ppo import PreviewEventPPO

PERIODS_MS = (20, 200)
PHASE_SLOT = 21
HELD_SLOT = 20


def configure(cfg, *, period_ms):
    if isinstance(period_ms, bool) or period_ms not in PERIODS_MS:
        raise ValueError('Registered D periods are20ms and200ms')
    cfg = copy.deepcopy(cfg)
    revision = cfg['revision']
    if revision['observation_dim'] != 26 or revision['macro_period_steps'] != 20:
        raise ValueError('Requires26D state and20 physical ticks per macro decision')
    if revision['active_action_dims'] not in ([1], [1, 2]):
        raise ValueError('Requires feed action and optionally learned D; no angular residual')
    revision.update(preview_period_steps=int(period_ms),
        state_semantics_version='local-corner-coherent-D-phase-v1',
        coherent_distance=dict(period_ms=int(period_ms), phase_slot=PHASE_SLOT,
            held_distance_slot=HELD_SLOT, macro_period_ms=20,
            phase='((period-physical_tick%period)%period)/period, for both fixed and adaptive',
            likelihood='D Gaussian scored only when state21 is zero; feed scored every macro; no inverse-event weighting',
            migration='Zero actor.shared_fc1.weight[:,21] and critic.fc1.weight[:,21]; fresh Adam'),
        preview_timing='Observe preceding held D and remaining phase; sample feed every20ms and D only at period event; KCM1ms; observe after full macro')
    return cfg


class CoherentDistanceEnvironment(local.LocalCornerEnvironment):
    def __init__(self, cfg, path_id, reference=None):
        period = cfg['revision']['coherent_distance']['period_ms']
        if period not in PERIODS_MS or cfg['revision']['preview_period_steps'] != period:
            raise ValueError('Coherent D config and physical holding period disagree')
        super().__init__(cfg, path_id, reference=reference)
        if self.inner.period != period:
            raise ValueError('Underlying environment did not install the D holding period')

    def _sense(self):
        super()._sense()
        period = self.cfg['revision']['coherent_distance']['period_ms']
        phase = ((period-self.inner.tick % period) % period)/period
        self.state[PHASE_SLOT] = phase
        self.trace[-1]['coherent_remaining_phase'] = float(phase)

    def step(self, action):
        start = len(self.trace)
        first_tick = self.inner.tick
        physics_before = self.physics_steps
        result = super().step(action)
        # The frozen macro tracer labels every20ms as a D event. Replace only
        # that diagnostic field using the actual physical holding schedule.
        physical_count = self.physics_steps-physics_before
        period = self.inner.period
        for index, row in enumerate(self.trace[start:]):
            event = float(index < physical_count and self.inner.adaptive
                          and (first_tick+index) % period == 0)
            row['preview_event'] = event
            row['control_status__preview_event'] = event
        return result


class CoherentDistancePPO(PositiveFeedPPO):
    @torch.no_grad()
    def take_action(self, state, return_info=False, deterministic=False):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError('Expected finite state vector')
        if not 0 <= state[PHASE_SLOT] < 1 or not 0 <= state[HELD_SLOT] <= 1:
            raise ValueError('Invalid remaining phase or held D')
        # Parent always consumes all three Gaussian draws, including inactive
        # coordinates; matched controls retain common random-number alignment.
        action, info = super().take_action(state, return_info=True, deterministic=deterministic)
        event = 2 in self.active_action_dims and state[PHASE_SLOT] == 0
        mask = np.asarray([0., 1., float(event)], dtype=np.float32)
        if 2 in self.active_action_dims and not event:
            action[2] = state[HELD_SLOT]
        mu = torch.as_tensor(info['mu'][None], device=self.device)
        std = torch.as_tensor(info['std'][None], device=self.device)
        latent = torch.as_tensor(action[None], device=self.device)
        score = self.latent_log_prob(mu,std,latent,torch.as_tensor(mask[None],device=self.device))
        info.update(action_mask=mask,old_log_prob=float(score[0,0]),preview_decision=bool(event))
        return (action,info) if return_info else action

    def _batch_masks(self, transitions, count):
        states = np.asarray(transitions['states'], dtype=np.float32)
        if states.shape != (count,self.state_dim) or not np.isfinite(states).all():
            raise ValueError('Invalid state batch')
        phase = states[:,PHASE_SLOT]
        if np.any((phase < 0) | (phase >= 1)):
            raise ValueError('Invalid remaining phase')
        expected = np.zeros((count,3),dtype=np.float32)
        expected[:,1] = 1.
        if 2 in self.active_action_dims:
            expected[:,2] = phase == 0
        if 'action_masks' not in transitions:
            raise ValueError('Explicit event masks required')
        supplied = np.asarray(transitions['action_masks'])
        if supplied.shape != expected.shape or not np.array_equal(supplied,expected):
            raise ValueError('Saved masks disagree with the observed D event phase')
        if 'preview_decisions' in transitions:
            decisions=np.asarray(transitions['preview_decisions']).reshape(-1)
            if decisions.shape != (count,) or not np.array_equal(decisions,expected[:,2]):
                raise ValueError('Preview decisions disagree with event masks')
        return torch.as_tensor(expected,device=self.device)

    def update(self, transitions):
        # Bypass only the macro subclass's constant-mask diagnostics; reuse
        # the original PPO objective, GAE, critic target and optimizer code.
        result = PreviewEventPPO.update(self,transitions)
        masks=np.asarray(transitions['action_masks'])
        self.last_update_diagnostics.update(
            macro_period_s=.020,decision_count_by_action=masks.sum(0).astype(int).tolist(),
            normalization='Mean over20ms macro decisions; no event-rate multiplier',
            likelihood='Joint Gaussian on newly sampled dimensions only; held D excluded',
            feed_sampling='Gaussian logit -> sigmoid at execution; original latent draw scored')
        return result


def make_agent(cfg, env):
    keys=('hidden_dim','actor_lr','critic_lr','lmbda','epochs','eps','gamma','ent_coef')
    agent=CoherentDistancePPO(state_dim=env.observation_dim,device=torch.device('cpu'),
        active_action_dims=cfg['revision']['active_action_dims'],
        fixed_preview=(cfg['revision']['fixed_distance_mm']-.8)/3.8,
        **{key:cfg['ppo'][key] for key in keys})
    path=local.core.old.ROOT/cfg['revision']['init_checkpoint']
    if local.core.old.sha(path) != cfg['revision']['init_checkpoint_sha256']:
        raise ValueError('Registered initializer changed')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    agent.actor.load_state_dict(payload['actor'])
    agent.critic.load_state_dict(payload['critic'])
    with torch.no_grad():
        agent.actor.shared_fc1.weight[:,PHASE_SLOT].zero_()
        agent.critic.fc1.weight[:,PHASE_SLOT].zero_()
    if agent.actor_optimizer.state or agent.critic_optimizer.state:
        raise ValueError('Initialization requires fresh Adam states')
    return agent
