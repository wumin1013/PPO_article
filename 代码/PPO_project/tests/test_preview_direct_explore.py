"""One exploration change, unchanged physical control, isolated protocol."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'PPO_project/tools'))
import preview_direct_explore as explore
from src.utils import revision_metrics as metrics


@pytest.fixture
def installed(monkeypatch):
    for obj,names in [(explore.study,['OUT','config_for','make_agent','source_state','protocol_definition','PILOT_VARIANTS','FORMAL_VARIANTS','select']),
                      (explore.study.v2,['make_env']),(metrics,['ordered_projection'])]:
        for name in names:monkeypatch.setattr(obj,name,getattr(obj,name))
    yield explore.install()


def setup_agents(installed,variant='direct'):
    cfg=installed.config_for(variant,101)
    env,obs=installed.make_env(cfg,'probe',reference=np.array([[0.,0.],[20.,0.]]))
    torch.manual_seed(41);baseline=explore.BASE_MAKE_AGENT(cfg,env)
    torch.manual_seed(41);candidate=installed.make_agent(cfg,env)
    return cfg,env,obs,baseline,candidate


def test_import_does_not_install_or_create_artifacts():
    code="""import sys
sys.path.insert(0,'PPO_project/tools')
import preview_direct_research as d
from src.utils import revision_metrics as m
before=(d.OUT,d.config_for,d.make_agent,d.source_state,d.protocol_definition,d.v2.make_env,m.ordered_projection)
import preview_direct_explore
import preview_direct_explore_batch
assert before==(d.OUT,d.config_for,d.make_agent,d.source_state,d.protocol_definition,d.v2.make_env,m.ordered_projection)
"""
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_preview_std_gradient_positive_physical_distributions_unchanged(installed):
    _,_,obs,baseline,candidate=setup_agents(installed)
    x=torch.tensor(np.repeat(obs[None,:],8,axis=0),dtype=torch.float32)
    mu0,std0=baseline.actor(x);mu1,std1=candidate.actor(x)
    torch.testing.assert_close(mu0,mu1,rtol=0,atol=0)
    torch.testing.assert_close(std0[:,:2],std1[:,:2],rtol=0,atol=0)
    assert float(std1[0,2].detach())==pytest.approx(.05)
    std1[:,2].sum().backward()
    assert candidate.actor.std_head.bias.grad[2]>0.
    assert torch.count_nonzero(candidate.actor.std_head.bias.grad[:2])==0
    std0[:,2].sum().backward()
    assert baseline.actor.std_head.bias.grad[2]==0.
    torch.manual_seed(74);a0=baseline.take_action(obs)
    torch.manual_seed(74);a1=candidate.take_action(obs)
    assert a0[:2]==a1[:2] and a0[2]!=a1[2]


@pytest.mark.parametrize('variant',['fixed_mid','fixed_mid_no_anchor'])
def test_inactive_fixed_preview_cannot_change_ppo_update(installed,variant):
    _,_,obs,baseline,candidate=setup_agents(installed,variant)
    rng=np.random.default_rng(21)
    states=np.repeat(obs[None,:],16,axis=0)+rng.normal(0,.001,(16,len(obs)))
    torch.manual_seed(52);actions0=[baseline.take_action(s) for s in states]
    torch.manual_seed(52);actions1=[candidate.take_action(s) for s in states]
    np.testing.assert_array_equal(np.array(actions0)[:,:2],np.array(actions1)[:,:2])
    common=dict(states=states,next_states=states+.001,rewards=np.linspace(-.01,.02,16),dones=[False]*15+[True])
    torch.manual_seed(83);loss0=baseline.update(dict(common,actions=actions0))
    torch.manual_seed(83);loss1=candidate.update(dict(common,actions=actions1))
    assert loss0==loss1
    for name,param in baseline.actor.state_dict().items():
        other=candidate.actor.state_dict()[name]
        if name=='std_head.bias':torch.testing.assert_close(param[:2],other[:2],rtol=0,atol=0)
        else:torch.testing.assert_close(param,other,rtol=0,atol=0)
    for p0,p1 in zip(baseline.critic.parameters(),candidate.critic.parameters()):
        torch.testing.assert_close(p0,p1,rtol=0,atol=0)
    assert candidate.active_action_dims==(0,1)
    assert candidate.last_update_diagnostics['epochs'][-1]['preview_mu_weight_gradient_norm']==0.


def test_shared_configs_budgets_and_full_direct_distance_are_preserved(installed):
    assert explore.install() is installed
    for variant in installed.PILOT_VARIANTS:
        new=installed.config_for(variant,101)
        base_variant=variant.removesuffix('_no_anchor') if variant in explore.FIXED_NO_ANCHOR else variant
        old=explore.BASE_CONFIG(base_variant,101)
        for key in new:
            if key!='revision':assert new[key]==old[key]
        assert new['revision']['preview_initial_std']==.05
        assert new['revision']['ordered_witness_spacing_mm']==.01
        assert new['revision']['active_action_dims']==([0,1,2] if variant.startswith('direct') else [0,1])
    protocol=installed.protocol_definition()
    assert protocol['formal_seeds']==[11,22,33,44,55]
    assert protocol['pilot_steps']==32768 and protocol['formal_steps']==131072
    assert protocol['geometry_distance_rule_used'] is False
    assert protocol['preview_exploration']['initial_std_u']==.05
    assert len(protocol['pilot_variants'])==10 and len(protocol['formal_variants'])==4
    assert protocol['total_formal_steps']==4*5*131072
    assert protocol['matched_fixed_baselines']['direct_no_anchor']=='selected_fixed_no_anchor'
    assert installed.direct_distance(0.)==.8 and installed.direct_distance(1.)==4.6
    assert 'PPO_project/tools/preview_direct_explore.py' in installed.source_state()['source_files']


def test_unregistered_exploration_scale_rejected(installed):
    cfg=installed.config_for('direct',101);cfg['revision']['preview_initial_std']=.1
    with pytest.raises(ValueError,match='initial std 0.05'):
        installed.make_agent(cfg,None)


def test_fixed_anchor_groups_selected_independently(installed,monkeypatch,tmp_path):
    monkeypatch.setattr(explore,'OUT',tmp_path)
    monkeypatch.setattr(installed,'OUT',tmp_path)
    (tmp_path/'protocol.json').write_text('{}',encoding='utf-8')
    monkeypatch.setattr(installed,'check_protocol',lambda:None)
    monkeypatch.setattr(installed,'_validated_run',lambda stage,variant,seed,budget:(None,[dict(variant=variant)]))
    scores={name:1 for name in explore.PILOT_VARIANTS}
    scores.update(fixed_mid=4,fixed_long_no_anchor=5,direct=10,direct_no_anchor=11)
    monkeypatch.setattr(installed,'_rank',lambda name,rows:dict(variant=name,score=[scores[name]]))
    path=explore.select()
    import json
    choice=json.loads(path.read_text())
    assert choice['selected_fixed']=='fixed_mid'
    assert choice['selected_fixed_no_anchor']=='fixed_long_no_anchor'
    cfg=installed.config_for('selected_fixed_no_anchor',11)
    assert cfg['revision']['motion_anchor_beta']==0.
    assert cfg['revision']['active_action_dims']==[0,1]
    assert cfg['reward_weights']['lookahead']['distance']==4.
    assert cfg['revision']['variant']=='selected_fixed_no_anchor'


def test_registered_freeze_requires_all_twenty_formal_models(installed,monkeypatch,tmp_path):
    import json
    monkeypatch.setattr(explore,'OUT',tmp_path)
    monkeypatch.setattr(installed,'OUT',tmp_path)
    installed.prepare();calls=[]
    def validated(stage,variant,seed,budget):
        calls.append((stage,variant,seed,budget))
        run=tmp_path/stage/f'{variant}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
        (run/'checkpoint.pt').write_bytes(f'{variant} {seed}'.encode())
        (run/'config.yaml').write_text(f'method: {variant}\n',encoding='utf-8')
        ok=variant in ('fixed_mid','fixed_center_no_anchor',*explore.FORMAL_VARIANTS)
        return run,[dict(success=ok,path_completed=ok,geometry_feasible=True,
                         ordered_progress=1. if ok else .2,elapsed_time_s=4.)]*3
    monkeypatch.setattr(installed,'_validated_run',validated)
    choice=json.loads(explore.select().read_text())
    assert choice['selected_fixed']=='fixed_mid'
    assert choice['selected_fixed_no_anchor']=='fixed_center_no_anchor'
    frozen=json.loads(installed.freeze().read_text())
    assert len(frozen['models'])==len(frozen['configs'])==20
    assert len(calls)==30 and all(c[-1]==131072 for c in calls[10:])
    assert set(c[1] for c in calls[10:])==set(explore.FORMAL_VARIANTS)
    with pytest.raises(FileExistsError):installed.freeze()
