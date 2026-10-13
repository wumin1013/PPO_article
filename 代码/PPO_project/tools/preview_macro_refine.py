"""Matched development fine-tuning from one validated fixed-preview policy.

The sole candidate change is common fixed-policy initialization. Network,
objective, action ranges, exploration, optimization and terminal audit remain
the macro preview_time configuration. Every fixed distance is independently
fine-tuned for the same additional budget. No test interface is exposed.
"""
from pathlib import Path

import torch

import preview_macro_study as core

BASE_OUT=core.OUT
OUT=BASE_OUT.parent/'macro_refine'
BASE_SOURCE=core.source_state
BASE_CONFIG=core.configuration
BASE_AGENT=core.make_agent
PRIOR=BASE_OUT/'pilot/preview_time_fixed3_seed101'


def source_state():
    import hashlib,json
    state=BASE_SOURCE()
    state['source_files'][Path(__file__).relative_to(core.old.ROOT).as_posix()]=core.old.sha(__file__)
    state['source_sha256']=hashlib.sha256(json.dumps(state['source_files'],sort_keys=True).encode()).hexdigest()
    return state


def configuration(candidate,method,seed):
    if candidate!='preview_time' or seed!=101:raise ValueError('Registered refinement is preview_time, development seed101 only')
    cfg=BASE_CONFIG(candidate,method,seed)
    cfg['revision'].update(study='macro-common-fixed-initialization-v1',
        initialization='Load actor and critic from independently trained preview_time_fixed3_seed101; new Adam states for every method',
        init_checkpoint=str(PRIOR/'checkpoint.pt'),init_checkpoint_sha256=core.old.sha(PRIOR/'checkpoint.pt'),
        pretraining_macro_decisions=4096,preview_initial_std=.15,
        physical_std_known_limitation='std_fix active; inherited floor warning does not apply',
        total_budget_note='4096 common fixed pretraining decisions plus 4096 independent fine-tuning decisions',
        refinement_change='Common actor/critic initialization only; third mean/std copied unchanged from inactive fixed head')
    return cfg


def make_agent(cfg,env):
    if core.old.sha(PRIOR/'checkpoint.pt')!=cfg['revision']['init_checkpoint_sha256']:
        raise ValueError('Pretraining checkpoint changed')
    agent=BASE_AGENT(cfg,env)
    checkpoint=torch.load(PRIOR/'checkpoint.pt',map_location='cpu',weights_only=False)
    agent.actor.load_state_dict(checkpoint['actor']);agent.critic.load_state_dict(checkpoint['critic'])
    return agent


def prepare():
    if core.old.read(PRIOR/'provenance.json')['source_sha256']!=BASE_SOURCE()['source_sha256']:
        raise ValueError('Prior macro source changed')
    training=core.old.read(PRIOR/'training_summary.json')
    if training['decisions']!=4096 or training['checkpoint_sha256']!=core.old.sha(PRIOR/'checkpoint.pt'):
        raise ValueError('Prior model not at the registered budget')
    validation=core.old.read(PRIOR/'validation/summary.json')['rows']
    expected={p['id'] for p in core.old.manifest()['paths'] if p['split']=='validation'}
    if len(validation)!=3 or {r['path_id'] for r in validation}!=expected or not all(r['success'] for r in validation):
        raise ValueError('Common initialization must pass every registered validation path')
    if any(r['checkpoint_sha256']!=training['checkpoint_sha256'] for r in validation):raise ValueError('Prior validation mismatch')
    value=dict(version='macro-common-fixed-initialization-v1',source=source_state(),
        prior_checkpoint_sha256=training['checkpoint_sha256'],prior_training_summary=training,
        prior_validation_sha256=core.old.sha(PRIOR/'validation/summary.json'),
        candidate='preview_time',methods=list(core.METHODS),seed=101,
        pretraining_macro_decisions=4096,additional_macro_decisions=4096,
        change='Only common initialization from valid fixed policy; new optimizer states for all methods',
        controls='Fixed1.2/3/4mm separately fine-tuned; select one uniform distance using validation, no per-path fixed choice',
        data_role='Single-seed development validation, no test access or independent-seed superiority claim',
        checkpoint_reuse='One common prior physically trained once; amortized and per-method attributed costs both reported',
        machine_execution_authorized=False)
    target=OUT/'protocol.json'
    if target.exists() and core.old.read(target)!=value:raise ValueError('Preserve refinement protocol')
    if not target.exists():core.old.write(target,value)
    return value


def install():
    core.OUT=OUT;core.source_state=source_state;core.configuration=configuration
    core.make_agent=make_agent;core.prepare=prepare
    return core


if __name__=='__main__':install().main()
