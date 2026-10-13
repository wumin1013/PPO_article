"""Continue saved sensing policies with one positive-feed sampling change.

No test interface. All validation paths are evaluated. Earlier artifacts and
the frozen numerical sources are immutable. Fixed/adaptive receive the same
additional budget from their respective equal-budget sensing checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import preview_macro_study as core
import preview_sensor_study as sensor
from preview_positive_feed_ppo import PositiveFeedPPO

OUT = core.old.ROOT / '论文项目/revision_20260910_continued/positive_feed_v1'
BASE_TRAIN = core.train
PRIOR = sensor.OUT / 'pilot'
METHODS = ('fixed3', 'adaptive')


def execute_action(latent):
    action = np.asarray(latent, dtype=float).copy()
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError('Expected a finite three-dimensional latent action')
    action[1] = np.exp(-np.logaddexp(0., -action[1]))
    return action


class PositiveFeedEnvironment(sensor.SensorEnvironment):
    def step(self, latent):
        start = len(self.trace)
        result = super().step(execute_action(latent))
        for row in self.trace[start:]:
            row['feed_logit'] = float(latent[1])
            row['feed_executed_action'] = float(execute_action(latent)[1])
        return result


def configuration(candidate, method, seed):
    if candidate != 'preview_time' or method not in METHODS or seed not in (101, 901):
        raise ValueError('Registered continuation: preview_time, fixed3/adaptive, seed101 (smoke901)')
    cfg = sensor.configuration(candidate, method, seed)
    prior = PRIOR / f'preview_time_{method}_seed101'
    cfg['revision'].update(
        study='positive-feed-logit-continuation-v1',
        candidate_change='Feed Gaussian logit and sigmoid execution only; control D and sensing unchanged',
        initialization='Resume actor/critic from corresponding sensor seed101; fresh Adam states; same additional budget',
        init_checkpoint=prior.relative_to(core.old.ROOT).as_posix()+'/checkpoint.pt',
        init_checkpoint_sha256=core.old.sha(prior/'checkpoint.pt'),
        pretraining_macro_decisions=4096,
        likelihood='Gaussian latent logits retained in PPO; deterministic evaluation executes sigmoid(feed mean); latent entropy',
        feed_logit_std='min(2, old physical std/(sigmoid(raw_mean)*(1-sigmoid(raw_mean))))',
        additional_decisions=4096,
    )
    return cfg


def make_agent(cfg, env):
    keys = ('hidden_dim','actor_lr','critic_lr','lmbda','epochs','eps','gamma','ent_coef')
    agent = PositiveFeedPPO(state_dim=env.observation_dim, device=torch.device('cpu'),
        active_action_dims=cfg['revision']['active_action_dims'],
        fixed_preview=(cfg['revision']['fixed_distance_mm']-.8)/3.8,
        **{key: cfg['ppo'][key] for key in keys})
    prior = core.old.ROOT / cfg['revision']['init_checkpoint']
    if core.old.sha(prior) != cfg['revision']['init_checkpoint_sha256']:
        raise ValueError('Preserve registered initial checkpoint')
    checkpoint = torch.load(prior, map_location='cpu', weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    return agent


def source_state():
    value = sensor.source_state()
    for name in ('preview_positive_feed_ppo.py','preview_positive_feed_study.py'):
        path = Path(__file__).with_name(name)
        value['source_files'][path.relative_to(core.old.ROOT).as_posix()] = core.old.sha(path)
    value['source_sha256'] = hashlib.sha256(json.dumps(value['source_files'],sort_keys=True).encode()).hexdigest()
    return value


def prepare():
    core.old.check_protocol()
    priors = {}
    for method in METHODS:
        path = PRIOR / f'preview_time_{method}_seed101'
        summary = core.old.read(path/'training_summary.json')
        assert summary['decisions'] == 4096
        assert summary['checkpoint_sha256'] == core.old.sha(path/'checkpoint.pt')
        priors[method] = dict(checkpoint_sha256=summary['checkpoint_sha256'],
            summary_sha256=core.old.sha(path/'training_summary.json'),
            prior_physics_steps=summary['physics_steps'], prior_wall_s=summary['training_wall_s'])
    value = dict(version='positive-feed-logit-continuation-v1',source=source_state(),
        manifest_sha256=core.old.sha(core.old.DATA/'manifest.json'),priors=priors,
        methods=list(METHODS),seed=101,additional_macro_decisions=4096,
        change='Feed sampling only: raw logit Gaussian and sigmoid execution, physical median preserved at initialization',
        unchanged='Learned control D, fixed six geometry coordinates, 26D network, reward, 20ms/1ms periods, KCM and full rest/output audit',
        evaluation='All three existing validation paths after the registered budget; no cherry-picked trajectories',
        objective='Final deterministic success, motion/tolerance compliance and complete elapsed time; training success is diagnostic only',
        advance='Diagnose nearest-corner sensing after this isolated sampling comparison; no new test use until final design freeze',
        pretraining='4096 prior decisions per method plus 4096 continuation; both costs reported',
        test_access=False,machine_execution_authorized=False)
    target = OUT/'protocol.json'
    if target.exists() and core.old.read(target) != value:
        raise ValueError('Registered study changed; create another version')
    if not target.exists():
        core.old.write(target,value)
    return value


def install():
    core.OUT=OUT
    core.source_state=source_state
    core.configuration=configuration
    core.MacroEnvironment=PositiveFeedEnvironment
    core.make_agent=make_agent
    core.prepare=prepare
    return core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    p=sub.add_parser('train')
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--decisions',type=int,default=4096)
    p.add_argument('--stage',choices=('smoke','pilot'),default='pilot')
    p=sub.add_parser('evaluate')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path)
    args=parser.parse_args()
    install()
    if args.command=='prepare':
        prepare()
    elif args.command=='train':
        core.train('preview_time',args.method,args.seed,args.decisions,args.stage)
    else:
        core.evaluate(args.run,args.output)


if __name__=='__main__':
    main()
