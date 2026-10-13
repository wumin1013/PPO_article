"""Fair independent initialization, immutable evidence and phase isolation."""
import copy
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_formal_continuation as formal
from test_preview_full_batch import ShortEpisodes, agent_factory, assert_tree_equal


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_prior(tmp_path, monkeypatch, method='adaptive', seed=11):
    monkeypatch.setattr(formal, 'ROOT', tmp_path)
    monkeypatch.setattr(formal, 'OUT', tmp_path/'formal')
    run = formal.run_dir('pretraining',method,seed)
    run.mkdir(parents=True)
    (run/'checkpoint.pt').write_bytes(f'own model {method} {seed}'.encode())
    formal.old.write(run/'training_summary.json',dict(seed=seed,method=method,decisions=4096,
        checkpoint_sha256=formal.old.sha(run/'checkpoint.pt')))
    return run


@pytest.mark.parametrize('method',formal.METHODS)
def test_same_numerical_configuration_but_own_seed_checkpoint(tmp_path,monkeypatch,method):
    prior = make_prior(tmp_path,monkeypatch,method,22)
    before = formal.full.configuration('preview_time',method,101)
    after = formal.configuration('continuation',method,22)
    assert after['seed'] == 22
    assert after['revision']['init_checkpoint'] == (prior/'checkpoint.pt').relative_to(tmp_path).as_posix()
    assert 'seed101' not in after['revision']['init_checkpoint']
    for key in before:
        if key not in ('seed','revision'): assert before[key] == after[key]
    for key in ('active_action_dims','fixed_distance_mm','corner_observation','feed_logit_std',
                'training_batch_macro_decisions','terminal_batch_rule','macro_period_steps'):
        assert before['revision'][key] == after['revision'][key]


def test_wrong_method_seed_or_budget_prior_is_rejected(tmp_path,monkeypatch):
    run = make_prior(tmp_path,monkeypatch)
    path=run/'training_summary.json'; good=formal.old.read(path)
    for key,value in (('seed',101),('method','fixed3'),('decisions',256),('checkpoint_sha256','fake')):
        bad=copy.deepcopy(good);bad[key]=value;formal.old.write(path,bad)
        with pytest.raises(ValueError): formal.configuration('continuation','adaptive',11)
    formal.old.write(path,good)
    assert formal.configuration('continuation','adaptive',11)['seed'] == 11


def test_distinct_fresh_seeds_produce_distinct_trainable_trunks():
    cfg = formal.configuration('pretraining','adaptive',11)
    env = type('Env',(),{'observation_dim':26})()
    torch.manual_seed(11); a = formal.BASE_SENSOR_AGENT(cfg,env)
    torch.manual_seed(22); b = formal.BASE_SENSOR_AGENT(cfg,env)
    assert any(not torch.equal(a.actor.state_dict()[k],b.actor.state_dict()[k]) for k in a.actor.state_dict())
    torch.manual_seed(11); repeated = formal.BASE_SENSOR_AGENT(cfg,env)
    assert_tree_equal(a.actor.state_dict(),repeated.actor.state_dict())


@pytest.mark.parametrize('method,seed,stage', [('fixed1p2',11,'formal'),('adaptive',101,'formal'),
                                             ('fixed3',11,'smoke'),('adaptive',901,'test')])
def test_unregistered_requests_rejected(method,seed,stage):
    with pytest.raises(ValueError): formal.validate_request(method,seed,stage)


def phase_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(formal,'ROOT',tmp_path)
    monkeypatch.setattr(formal,'OUT',tmp_path/'formal')
    protocol=dict(sources={phase:dict(source_files={},source_sha256=phase+'-unit') for phase in ('pretraining','continuation')})
    formal.old.write(formal.OUT/'protocol.json',protocol)
    monkeypatch.setattr(formal,'prepare',lambda:protocol)
    return protocol


def test_phase_switch_restores_all_bindings_even_on_exception(tmp_path,monkeypatch):
    phase_fixture(tmp_path,monkeypatch)
    names=('OUT','source_state','configuration','MacroEnvironment','make_agent','prepare')
    before={name:getattr(formal.core,name) for name in names}
    for phase in ('pretraining','continuation','pretraining'):
        with pytest.raises(RuntimeError,match='deliberate'):
            with formal.phase_installed(phase):
                assert formal.core.MacroEnvironment is (formal.sensor.SensorEnvironment if phase=='pretraining' else formal.full.local.LocalCornerEnvironment)
                with pytest.raises(RuntimeError,match='nest'): 
                    with formal.phase_installed(phase): pass
                raise RuntimeError('deliberate')
        assert all(getattr(formal.core,name) is value for name,value in before.items())


def test_feed_patch_is_rejected_before_installation(tmp_path,monkeypatch):
    phase_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(formal.cnc_env,'apply_kinematic_constraints',lambda *args:None)
    with pytest.raises(RuntimeError,match='fresh process'):
        with formal.phase_installed('continuation'): pass


def test_selection_full_success_precedes_time_and_incomplete_ties_choose_earliest():
    incomplete=dict(step=0,successes=2,complete_validation_time_s=None)
    complete=dict(step=4096,successes=3,complete_validation_time_s=99.)
    assert min([incomplete,complete],key=formal.selection_key) is complete
    later=dict(step=256,successes=2,complete_validation_time_s=.001)
    assert min([later,incomplete],key=formal.selection_key) is incomplete
    initial=dict(step=0,successes=3,complete_validation_time_s=9.)
    assert min([initial,complete],key=formal.selection_key) is initial


def test_sensor_interrupted_suffix_is_preserved_and_reconciled(tmp_path):
    torch.save(dict(decisions=64,episode=1),tmp_path/'checkpoint.pt')
    updates=b'{"decisions":64}\n{"decisions":128}\n{partial'
    episodes=b'{"episode":0}\n{"episode":1}\n'
    (tmp_path/'updates.jsonl').write_bytes(updates);(tmp_path/'episodes.jsonl').write_bytes(episodes)
    formal.repair_sensor_logs(tmp_path)
    assert (tmp_path/'updates.jsonl').read_bytes()==b'{"decisions":64}\n'
    assert (tmp_path/'episodes.jsonl').read_bytes()==b'{"episode":0}\n'
    archived=[p.read_bytes() for p in (tmp_path/'recovery').iterdir()]
    assert updates in archived and episodes in archived


def test_no_committed_sensor_checkpoint_keeps_uncommitted_log_bytes(tmp_path):
    extra=b'{"decisions":64}\n';(tmp_path/'updates.jsonl').write_bytes(extra)
    formal.repair_sensor_logs(tmp_path)
    assert (tmp_path/'updates.jsonl').read_bytes()==b''
    assert any(p.read_bytes()==extra for p in (tmp_path/'recovery').iterdir())


def test_continuation_uses_original_trainer_and_protects_earlier_snapshot(tmp_path,monkeypatch):
    phase_fixture(tmp_path,monkeypatch)
    cfg=dict(seed=901,revision=dict(training_path_ids=['three','five'],variant='adaptive',pretraining_macro_decisions=64),ppo=dict(epochs=2))
    monkeypatch.setattr(formal,'configuration',lambda *args:copy.deepcopy(cfg))
    monkeypatch.setattr(formal.full.local,'LocalCornerEnvironment',ShortEpisodes)
    monkeypatch.setattr(formal.full.local,'make_agent',agent_factory)
    def provenance(run,cfg,protocol):
        run.mkdir(parents=True,exist_ok=True)
        (run/'config.yaml').write_text(formal.yaml.safe_dump(cfg),encoding='utf-8')
        for name in ('provenance.json','source.zip'): (run/name).write_text('unit',encoding='utf-8')
    monkeypatch.setattr(formal,'_ensure_continuation_provenance',provenance)
    initial=formal.train_continuation('adaptive',901,0,'smoke')
    initial_bytes=(initial/'checkpoint.pt').read_bytes()
    final=formal.train_continuation('adaptive',901,64,'smoke')
    checkpoint=torch.load(final/'checkpoint.pt',weights_only=False)
    assert len(checkpoint['updates'])==1 and checkpoint['optimized_decisions']==64
    assert len(checkpoint['episodes'])>1
    assert formal.train_continuation('adaptive',901,0,'smoke')==initial
    assert (initial/'checkpoint.pt').read_bytes()==initial_bytes
    (initial/'episodes.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        formal.train_continuation('adaptive',901,0,'smoke')


def test_freeze_requires_all_ten_complete_chains(tmp_path,monkeypatch):
    phase_fixture(tmp_path,monkeypatch)
    with pytest.raises(FileNotFoundError): formal.freeze()
    assert not (formal.OUT/'freeze.json').exists()


def test_validation_agent_construction_cannot_change_resumed_updates(tmp_path):
    cfg=dict(seed=11,revision=dict(training_path_ids=['three','five'],variant='adaptive'),ppo=dict(epochs=2))
    original=formal.full.FullBatchTrainer(cfg,tmp_path/'original',ShortEpisodes,agent_factory,'rng-source')
    original.advance(64)
    shutil.copytree(tmp_path/'original',tmp_path/'with_validation')
    original.advance(128)
    expected_rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
    # The validator constructs three agents, consuming Torch initialization
    # randomness; also contaminate NumPy/Python explicitly as a stronger check.
    for _ in range(3):
        formal.BASE_SENSOR_AGENT(formal.configuration('pretraining','adaptive',11),
                                type('Env',(),{'observation_dim':26})())
        torch.randn(137);np.random.normal(size=127);random.random()
    resumed=formal.full.FullBatchTrainer(cfg,tmp_path/'with_validation',ShortEpisodes,agent_factory,'rng-source')
    resumed.advance(128)
    for name in ('actor','critic','actor_optimizer','critic_optimizer'):
        assert_tree_equal(getattr(original.agent,name).state_dict(),getattr(resumed.agent,name).state_dict())
    assert_tree_equal(original.updates,resumed.updates)
    assert_tree_equal(original.episodes,resumed.episodes)
    assert_tree_equal(expected_rng,(torch.get_rng_state(),np.random.get_state(),random.getstate()))
