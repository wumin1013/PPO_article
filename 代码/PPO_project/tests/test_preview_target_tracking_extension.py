"""Exact source-only prefix migration, bounded budgets and evidence gates."""
import copy
from pathlib import Path
import sys
import threading
import time
import zipfile

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_target_tracking_extension as ext
from test_preview_full_batch import ShortEpisodes, agent_factory, assert_tree_equal, config


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def prefix(tmp_path,monkeypatch):
    """A real PPO update plus nonempty cross-episode pending policy buffer."""
    monkeypatch.setattr(ext,'ROOT',tmp_path)
    monkeypatch.setattr(ext,'OUT',tmp_path/'extension')
    parent=tmp_path/'frozen_parent';parent.mkdir()
    cfg=config()
    trainer=ext.BASE_TRAINER(cfg,parent,ShortEpisodes,agent_factory,'old-source')
    trainer.advance(69)
    payload=torch.load(parent/'checkpoint.pt',weights_only=False)
    (parent/'config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    ext.audit.write(parent/'training_summary.json',trainer.summary(payload))
    ext.audit.write(parent/'provenance.json',dict(source_sha256='old-source'))
    with zipfile.ZipFile(parent/'source.zip','x') as archive:archive.writestr('old.py','old source')
    ext.core.old.save_trace(parent/'training_final_partial.npz',trainer.env.trace)
    files={p.name:ext.audit.sha(p) for p in parent.iterdir() if p.is_file()}
    ext.audit.write(parent/'snapshot_manifest.json',dict(files=files))
    script=tmp_path/'new_driver.py';script.write_text('source only\n',encoding='utf-8')
    protocol=dict(source=dict(source_sha256='new-source',source_files={'new_driver.py':ext.audit.sha(script)}),
        parents={'adaptive':dict(snapshot='frozen_parent',config=cfg,
            checkpoint_sha256=ext.audit.sha(parent/'checkpoint.pt'),source_sha256='old-source',
            snapshot_manifest_sha256=ext.audit.sha(parent/'snapshot_manifest.json'),
            learning_state_sha256=ext.learning_digest(payload),parent_decisions=69)})
    ext.audit.write(ext.OUT/'protocol.json',protocol)
    return parent,tmp_path/'continued',protocol,payload


def perform_import(prefix):
    parent,run,protocol,_=prefix
    return ext._import_prefix(parent,run,protocol,'adaptive',ShortEpisodes,agent_factory)


def test_source_only_import_preserves_every_learning_field_and_parent_bytes(prefix):
    parent,run,protocol,original=prefix
    before={p.name:p.read_bytes() for p in parent.iterdir()}
    perform_import(prefix)
    imported=torch.load(run/'checkpoint.pt',weights_only=False)
    assert imported['source_sha256']=='new-source'
    assert ext.learning_digest(imported)==ext.learning_digest(original)
    for key in ext.RESUME_KEYS:assert_tree_equal(imported[key],original[key])
    assert len(imported['buffer']['states'])==5 and len(original['actions'])==2
    assert imported['actor_optimizer']['state'] and imported['critic_optimizer']['state']
    assert {p.name:p.read_bytes() for p in parent.iterdir()}==before
    assert {p.name:p.read_bytes() for p in (run/'origin_step1024').iterdir()}==before
    ledger=ext.audit.read(run/'prefix_import.json')
    assert ledger['new_training_decisions']==0
    assert ledger['replayed_current_episode_actions']==len(original['actions'])
    assert ledger['parent_files_unchanged']
    assert ledger['new_environment_replay_wall_s']>=0
    assert imported['replay_wall_s']>=original['replay_wall_s']


def test_import_continuation_crosses_update_boundary_exactly_and_is_idempotent(prefix,tmp_path):
    parent,run,protocol,_=prefix
    direct=ext.BASE_TRAINER(config(),tmp_path/'direct',ShortEpisodes,agent_factory,'old-source')
    direct.advance(128)
    expected=torch.load(tmp_path/'direct/checkpoint.pt',weights_only=False)
    perform_import(prefix)
    imported_bytes=(run/'checkpoint.pt').read_bytes()
    assert perform_import(prefix)==run
    assert (run/'checkpoint.pt').read_bytes()==imported_bytes
    resumed=ext.BASE_TRAINER(config(),run,ShortEpisodes,agent_factory,'new-source')
    resumed.advance(96)
    advanced_bytes=(run/'checkpoint.pt').read_bytes()
    perform_import(prefix)  # Repeated import never rolls back newer decisions.
    assert (run/'checkpoint.pt').read_bytes()==advanced_bytes
    resumed=ext.BASE_TRAINER(config(),run,ShortEpisodes,agent_factory,'new-source')
    resumed.advance(128)
    actual=torch.load(run/'checkpoint.pt',weights_only=False)
    for key in ext.RESUME_KEYS:assert_tree_equal(actual[key],expected[key])
    assert ext.learning_digest(actual)==ext.learning_digest(expected)
    assert len(actual['updates'])==2 and actual['optimized_decisions']==128


@pytest.mark.parametrize('field',('config','source_sha256','learning_state_sha256'))
def test_registered_parent_metadata_mismatch_is_rejected(prefix,field):
    _,run,protocol,_=prefix
    if field=='config':protocol['parents']['adaptive'][field]=dict(seed=999)
    else:protocol['parents']['adaptive'][field]='changed'
    with pytest.raises(ValueError,match='config/source/learning state'):
        perform_import(prefix)
    assert not run.exists()


def test_parent_artifact_hash_mismatch_is_rejected_without_publication(prefix):
    parent,run,_,_=prefix
    with (parent/'checkpoint.pt').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='Frozen prefix artifact changed'):
        perform_import(prefix)
    assert not run.exists()


def test_replay_failure_never_publishes_partial_import(prefix):
    parent,run,protocol,_=prefix
    class ChangedReplay(ShortEpisodes):
        def __init__(self,cfg,path_id):
            super().__init__(cfg,path_id);self.state[0]+=1
    before=(parent/'checkpoint.pt').read_bytes()
    with pytest.raises(AssertionError):
        ext._import_prefix(parent,run,protocol,'adaptive',ChangedReplay,agent_factory)
    assert not run.exists()
    assert (parent/'checkpoint.pt').read_bytes()==before
    assert list(run.parent.glob(f'.{run.name}.importing-*'))


@pytest.mark.parametrize('what',('runtime_source','runtime_learning','origin'))
def test_reimport_rejects_changed_imported_state_or_origin(prefix,what):
    _,run,_,_=prefix;perform_import(prefix)
    if what=='origin':
        with (run/'origin_step1024/checkpoint.pt').open('ab') as stream:stream.write(b'changed')
    else:
        value=torch.load(run/'checkpoint.pt',weights_only=False)
        if what=='runtime_source':value['source_sha256']='other'
        else:value['actor'][next(iter(value['actor']))].view(-1)[0]+=1
        torch.save(value,run/'checkpoint.pt')
    with pytest.raises(ValueError):perform_import(prefix)


def test_old_restore_source_guard_remains_strict_after_import(prefix):
    _,run,_,_=prefix;perform_import(prefix)
    with pytest.raises(ValueError,match='source mismatch'):
        ext.BASE_TRAINER(config(),run,ShortEpisodes,agent_factory,'old-source')


@pytest.mark.parametrize('completed',(False,True))
def test_incomplete_or_failed_grid_never_registers_extension(tmp_path,monkeypatch,completed):
    monkeypatch.setattr(ext,'OUT',tmp_path/'extension')
    monkeypatch.setattr(ext.grid,'OUT',tmp_path/'grid')
    previous=dict(input_sha256={},cached_D_mechanism={'same_actor_D_screen_passed':True})
    monkeypatch.setattr(ext.grid,'prepare',lambda:previous)
    monkeypatch.setattr(ext.audit,'verify_files',lambda *args:None)
    ext.audit.write(ext.grid.OUT/'delivery_manifest.json',dict(completed=completed,files={}))
    adaptive=dict(success_count=2,complete_total_time_s=None)
    fixed=dict(success_count=3,complete_total_time_s=15.)
    ext.audit.write(ext.grid.OUT/'summary.json',dict(selected_adaptive=adaptive,uniform_selected_fixed=fixed,
        original_off_adaptive_time_s=13.931,gate=ext.grid.eligibility(adaptive,fixed,13.931,True)))
    with pytest.raises(ValueError):ext.prepare()
    assert not (ext.OUT/'protocol.json').exists()


def test_registered_extension_rejects_changed_source(tmp_path,monkeypatch):
    monkeypatch.setattr(ext,'OUT',tmp_path/'extension')
    ext.audit.write(ext.OUT/'protocol.json',dict(input_sha256={},source={'source_sha256':'old'}))
    monkeypatch.setattr(ext,'source_state',lambda:dict(source_sha256='new'))
    with pytest.raises(ValueError,match='Frozen extension source changed'):ext.prepare()


@pytest.mark.parametrize('method,step',(('adaptive',1024),('adaptive',0),('adaptive',8192),('unknown',2048)))
def test_worker_refuses_old_new_or_unknown_budget(method,step):
    with pytest.raises(ValueError,match='Only cumulative2048/4096'):ext.worker(method,step)


def test_worker_requires_import_and_2048_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(ext,'OUT',tmp_path)
    with pytest.raises(ValueError,match='Import the registered1024'):ext.worker('adaptive',2048)
    ext.audit.write(tmp_path/'development/preview_time_adaptive_seed101/prefix_import.json',{})
    monkeypatch.setattr(ext,'prepare',lambda:{})
    with pytest.raises(ValueError,match='Complete2048'):ext.worker('adaptive',4096)


def test_configuration_keeps_exact_parent_and_has_no_new_seeds(monkeypatch):
    original=config();monkeypatch.setattr(ext,'prepare',lambda:dict(parents={'adaptive':{'config':original}}))
    copied=ext.configuration('preview_time','adaptive',101)
    assert copied==original and copied is not original
    copied['revision']['variant']='changed';assert original['revision']['variant']=='adaptive'
    with pytest.raises(ValueError):ext.configuration('preview_time','adaptive',102)


def nodes(times=None,successes=None):
    result=[]
    for i,step in enumerate(ext.STEPS):
        count=3 if successes is None else successes[i]
        result.append(dict(step=step,checkpoint=f'step{step}',checkpoint_sha256=f'sha{step}',cache_reused=step<=1024,
            rows=[dict(path_id=path,success=j<count,elapsed_time_s=(times or [3.]*6)[i]/3)
                for j,path in enumerate(ext.audit.PATH_IDS)]))
    return result


def test_six_node_selection_preserves_failures_and_earliest_ties():
    candidates=nodes([30.,.1,8.,8.,9.,10.],[3,2,3,3,3,3])
    result=ext.choose(list(reversed(candidates)))
    assert result['selected_step']==512 and result['complete_total_time_s']==8.
    assert result['all_nodes'][4]['rows'][2]['success'] is False
    assert ext.choose(nodes([30.,.1,8.,8.,9.,10.],[2]*6))['selected_step']==0
    assert ext.choose(nodes([30.,.1,8.,8.,9.,10.],[2]*6))['complete_total_time_s'] is None
    with pytest.raises(ValueError,match='All six'):ext.choose(candidates[:-1])
    candidates[-1]['rows'].pop()
    with pytest.raises(ValueError):ext.choose(candidates)


def test_selected_actor_D_effect_is_not_inherited_by_changed_checkpoint():
    protocol=dict(cached_D_mechanism=dict(checkpoint_sha256='step0sha',decision=dict(
        same_actor_D_screen_passed=True,learned_vs_best_complete_lock_time_gain_pct=1.44)))
    reused=ext.selected_D_gate({'checkpoint_sha256':'step0sha'},protocol)
    changed=ext.selected_D_gate({'checkpoint_sha256':'step4096sha'},protocol)
    assert reused['same_actor_D_gate_passed'] and reused['applicable_gain_pct']==1.44
    assert not changed['same_actor_D_gate_passed'] and changed['new_rollouts_required']==12
    assert changed['applicable_gain_pct'] is None


def test_additional_costs_use_parent_baseline_once():
    parent=dict(parent_decisions=1024,parent_training_wall_s=100.,parent_replay_wall_s=2.,
        parent_physics_steps=20000,parent_command_steps=1024,parent_optimizer_updates=16,
        parent_adam_steps_per_network=64)
    summary=dict(decisions=4096,training_wall_s=400.,environment_replay_wall_s=5.,
        physics_steps=80000,command_steps=4096,optimizer_updates=64,adam_steps_per_network=256,
        checkpoint_sha256='final')
    value=ext.extension_costs(summary,parent)
    assert value['new_training_decisions']==3072 and value['new_optimizer_updates']==48
    assert value['new_adam_steps_per_network']==192
    assert value['additional_training_wall_s']==300. and value['additional_replay_wall_s']==3.
    assert value['new_physics_steps']==60000 and value['new_command_steps']==3072


def test_install_restores_all_bindings_after_exception():
    names=('OUT','configuration','source_state','prepare')
    saved={name:getattr(ext.full,name) for name in names}
    old_core={name:getattr(ext.core,name) for name in (*names,'MacroEnvironment','make_agent')}
    with pytest.raises(RuntimeError,match='controlled'):
        with ext.installed():
            assert ext.full.configuration is ext.configuration
            with pytest.raises(RuntimeError,match='Do not nest'):
                with ext.installed():pass
            raise RuntimeError('controlled')
    assert {name:getattr(ext.full,name) for name in names}==saved
    assert {name:getattr(ext.core,name) for name in old_core}==old_core


def test_scheduler_uses_four_imports_then_only_new_budgets_with_two_process_cap(monkeypatch):
    calls=[];completed=[];active=0;peak=0;lock=threading.Lock()
    monkeypatch.setattr(ext,'prepare',lambda:None)
    monkeypatch.setattr(ext,'report',lambda:dict(complete=True))
    def child(command,method,step=None):
        nonlocal active,peak
        with lock:
            if command=='worker':
                assert sum(item[0]=='import' for item in completed)==4
                if step==4096:assert sum(item[2]==2048 for item in completed)==4
            active+=1;peak=max(peak,active);calls.append((command,method,step))
        time.sleep(.005)
        with lock:active-=1;completed.append((command,method,step))
    monkeypatch.setattr(ext,'_scheduled_child',child)
    assert ext.run()=={'complete':True}
    assert peak<=2 and len(calls)==12
    assert calls[:4]==[('import',method,None) for method in ext.METHODS]
    assert set(calls[4:])=={('worker',method,step) for method in ext.METHODS for step in ext.NEW_STEPS}
