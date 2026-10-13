"""Direct PPO with an interior, trainable initial preview exploration scale.

Only the third latent-action standard deviation changes from the fine study:
0.01 to 0.05. Physical heads, full-distance policy mapping, shared observations,
reward/certificate, constraints and optimization budgets are unchanged.
Importing this module never activates adapters or evaluates paths.
"""
from pathlib import Path
import hashlib
import json
import math
import time

import torch

import preview_direct_fine as fine

study=fine.study
OUT=study.ROOT/'论文项目/revision_20260908/preview_direct_explore'
PREVIEW_INITIAL_STD=.05
BASE_CONFIG=fine.config_for
BASE_SOURCE=fine.source_state
BASE_PROTOCOL=fine.protocol_definition
BASE_MAKE_AGENT=study.make_agent
FIXED_NO_ANCHOR={name+'_no_anchor':distance for name,distance in study.DISTANCES.items()}
PILOT_VARIANTS=(*study.DISTANCES,'direct','direct_no_anchor',*FIXED_NO_ANCHOR)
FORMAL_VARIANTS=('selected_fixed','direct','selected_fixed_no_anchor','direct_no_anchor')


def config_for(variant,seed):
    if variant not in (*PILOT_VARIANTS,'selected_fixed','selected_fixed_no_anchor'):
        raise ValueError(f'Unknown exploration-study variant: {variant}')
    base_variant=variant.removesuffix('_no_anchor') if variant in FIXED_NO_ANCHOR else variant
    if variant=='selected_fixed_no_anchor':
        selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
        base_variant=selection['selected_fixed_no_anchor'].removesuffix('_no_anchor')
    cfg=BASE_CONFIG(base_variant,seed)
    cfg['revision'].update(study='direct-distance-shared-gain15-fine-witness-explore05-v1',
        variant=variant,
        preview_initial_std=PREVIEW_INITIAL_STD,
        initialization='fresh actor means [0,0.05,(3-0.8)/3.8], std [0.01,0.01,0.05]; no learned weights or supervision',
        preview_exploration_change_role='Only third-head initial std changes; fixed third head remains inactive',
        physical_std_known_limitation='The two unchanged physical std heads initialize below the float32 clamp floor and have zero std gradients at initialization')
    if variant in FIXED_NO_ANCHOR or variant=='selected_fixed_no_anchor':
        cfg['revision']['motion_anchor_beta']=0.
    return cfg


def make_agent(cfg,env):
    initial_std=cfg['revision']['preview_initial_std']
    if initial_std!=PREVIEW_INITIAL_STD:
        raise ValueError('The registered preview exploration candidate requires initial std 0.05')
    agent=BASE_MAKE_AGENT(cfg,env)
    with torch.no_grad():
        agent.actor.std_head.bias[2]=math.log(math.expm1(initial_std-.001))
    return agent


def source_state():
    state=BASE_SOURCE()
    for path in (Path(__file__),Path(__file__).with_name('preview_direct_explore_batch.py')):
        state['source_files'][path.relative_to(study.ROOT).as_posix()]=study.backend.sha(path)
    state['source_sha256']=hashlib.sha256(json.dumps(state['source_files'],sort_keys=True).encode()).hexdigest()
    return state


def protocol_definition():
    value=BASE_PROTOCOL()
    value.update(version='preview-direct-shared-gain15-fine-witness-explore05-v1',
        predecessor='preview_direct_fine: immutable fine-certificate candidate; no holdout data used',
        candidate_change='Only third latent preview initial std 0.01 to 0.05; fixed variants also initialize third std 0.05 but exclude that inactive action from likelihood and entropy',
        diagnostic_basis=['preview_direct/exploration_diagnostic/summary.json',
                          'preview_direct/exploration_diagnostic/std_initialization_probe.csv'],
        initialization='fresh means [0,0.05,(3-0.8)/3.8], std [0.01,0.01,0.05]; 3 mm from developer fixed-grid evidence, not pretraining',
        preview_exploration=dict(initial_std_u=PREVIEW_INITIAL_STD,initial_sigma_distance_mm=3.8*PREVIEW_INITIAL_STD,
            initial_mean_distance_mm=3.,bounds_mm=[.8,4.6],trainable=True,
            temporal_sampling='Independent latent Gaussian draw at every 1 ms interpolation step',
            mapping='D=0.8+3.8*clip(u,0,1); unchanged, no geometry control rule',
            gradient_diagnosis='float32 softplus(log(expm1(.01-.001)))+.001 is below clamp(min=.01); saved scalar autograd probe gives zero bias gradient at .01 and positive gradient at .05',
            fixed_control='Third dimension remains inactive and excluded from fixed PPO likelihood and entropy',
            physical_control='First two initial means/std and physical anchor unchanged; no extra pretraining or physical-head modification'),
        known_limitations=['The two unchanged physical std heads are initially in the hard-clamp zero-gradient region',
            'Increasing initial preview std does not itself prove useful state-dependent distance adaptation or performance gain',
            'Shared global gradient clipping and joint PPO likelihood couple active-head optimization'],
        unchanged='fine study shared heading gain 15/s, ordered witnesses 0.01 mm, physical reward, KCM, endpoint handling, all physical constraints, initial means, PPO lr/clip/entropy/epochs and budgets, train/validation/test split')
    value.update(pilot_variants=list(PILOT_VARIANTS),formal_variants=list(FORMAL_VARIANTS),
        total_pilot_steps=len(PILOT_VARIANTS)*study.PILOT_STEPS,
        total_formal_steps=len(FORMAL_VARIANTS)*len(study.SEEDS)*study.FORMAL_STEPS,
        selection='All ten pilots retained; uniform fixed distance selected separately within each anchor beta using validation only',
        matched_fixed_baselines={'direct':'selected_fixed','direct_no_anchor':'selected_fixed_no_anchor'},
        fairness_extension='Four independently trained beta-zero fixed-distance pilots and a beta-zero selected fixed formal baseline prevent confounding preview contribution with the physical-action anchor',
        test_policy='No holdout use until all twenty newly retrained formal models/configurations freeze; never automatic evaluation')
    return value


def select():
    study.check_protocol();target=OUT/'selection.json'
    if target.exists():raise FileExistsError('Preserve selected fixed distances')
    ranking=[]
    for variant in PILOT_VARIANTS:
        _,rows=study._validated_run('pilot',variant,101,study.PILOT_STEPS)
        ranking.append(study._rank(variant,rows))
    anchored=max((r for r in ranking if r['variant'] in study.DISTANCES),key=lambda r:r['score'])['variant']
    unanchored=max((r for r in ranking if r['variant'] in FIXED_NO_ANCHOR),key=lambda r:r['score'])['variant']
    study.backend.write_json(target,dict(selected_fixed=anchored,fixed_distance_mm=study.DISTANCES[anchored],
        selected_fixed_no_anchor=unanchored,fixed_distance_no_anchor_mm=FIXED_NO_ANCHOR[unanchored],
        rankings=ranking,selection_groups={'beta1':list(study.DISTANCES),'beta0':list(FIXED_NO_ANCHOR)},
        source_sha256=source_state()['source_sha256'],protocol_sha256=study.backend.sha(OUT/'protocol.json'),
        test_results_used_for_selection=False,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return target


def install():
    """Install only in this explicitly selected CLI process; safe to call twice."""
    fine.install()
    study.OUT=OUT
    study.config_for=config_for
    study.make_agent=make_agent
    study.PILOT_VARIANTS=PILOT_VARIANTS
    study.FORMAL_VARIANTS=FORMAL_VARIANTS
    study.select=select
    study.source_state=source_state
    study.protocol_definition=protocol_definition
    return study


if __name__=='__main__':install().main()
