"""Registered1024-to4096 continuation; preserve every frozen parent prefix.

Only source attribution and orchestration change. FullBatchTrainer restores
the original config/source first; an explicitly checked import then binds its
unchanged learning state to the new driver. No five-seed/pretraining interface.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

import preview_target_tracking_grid as grid

matched, audit, target = grid.matched, grid.audit, grid.target
full, core, ROOT, BASE = grid.full, grid.core, grid.ROOT, grid.BASE
OUT = BASE/'target_tracking_extension_v1'
METHODS = ('adaptive','fixed3','fixed1p2','fixed4')
STEPS = (0,256,512,1024,2048,4096)
NEW_STEPS = (2048,4096)
BASE_TRAINER = full.FullBatchTrainer
_active = False
# Include every replay/learning field; exclude only recorded source and clocks.
RESUME_KEYS = (
    'full_batch_version','config','actor','critic','actor_optimizer','critic_optimizer',
    'decisions','episode','optimized_decisions','buffer','actions','updates','episodes',
    'trace_files','ep_return','physics_before','commands_before','observation',
    'current_physics_steps','current_command_steps','completed_episode_physics_steps',
    'completed_episode_command_steps','episode_actions','torch_rng','numpy_rng','python_rng')


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if audit.read(path) != value:
            raise ValueError(f'Preserve registered artifact: {path}')
    else: audit.write(path,value)
    return value


def learning_digest(state):
    """Deterministic bitwise binding of numerical state, independent of torch ZIP."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            visit(value.detach().cpu().contiguous().numpy())
        elif isinstance(value,np.ndarray):
            digest.update(b'array');digest.update(str((value.shape,value.dtype)).encode());digest.update(value.tobytes())
        elif isinstance(value,dict):
            digest.update(b'dict')
            for key in sorted(value,key=lambda k:repr(k)): visit(key);visit(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(type(value).__name__.encode());digest.update(str(len(value)).encode())
            for item in value: visit(item)
        elif isinstance(value,np.generic): visit(value.item())
        else: digest.update((type(value).__name__+':'+repr(value)+';').encode())
    for key in RESUME_KEYS: visit(key);visit(state[key])
    return digest.hexdigest()


def source_state():
    parent = grid.source_state()
    files = dict(parent['source_files'])
    for name in ('PPO_project/tools/preview_target_tracking_extension.py',
                 'PPO_project/tests/test_preview_target_tracking_extension.py'):
        files[name] = audit.sha(ROOT/name)
    return dict(source_files=files,
        source_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),head=parent.get('head'))


def parent_root(method):
    if method not in METHODS: raise ValueError('Registered four methods only')
    return target.OUT if method=='adaptive' else matched.OUT if method=='fixed3' else grid.OUT


def _snapshot_files(path):
    manifest = audit.read(path/'snapshot_manifest.json')
    for name,expected in manifest['files'].items():
        target_path = (path/name).resolve()
        if not target_path.is_relative_to(path.resolve()) or audit.sha(target_path)!=expected:
            raise ValueError(f'Frozen prefix artifact changed: {name}')
    return manifest['files']


def _cached_node(path, method, step, ledger):
    provenance = audit.verify_parent(path,ledger)
    rows = audit.cached_rows(path/'validation',ledger)
    grid.validate_cached_binding(path,rows,step,method)
    summary = audit.read(path/'training_summary.json')
    if summary['method']!=method or summary['seed']!=101 or summary['decisions']!=step or summary['checkpoint_sha256']!=audit.sha(path/'checkpoint.pt'):
        raise ValueError('Cached prefix training identity changed')
    cfg = yaml.safe_load((path/'config.yaml').read_text(encoding='utf-8'))
    if cfg != matched.BASE_CONFIG('preview_time',method,101):
        raise ValueError('Cached prefix numerical config changed')
    return dict(step=step,checkpoint=path.relative_to(ROOT).as_posix(),
        checkpoint_sha256=audit.sha(path/'checkpoint.pt'),source_sha256=provenance['source_sha256'],
        rows=rows,cache_reused=True,training_summary=summary)


def preflight():
    previous = grid.prepare()
    completed = audit.read(grid.OUT/'delivery_manifest.json')
    if not completed['completed']:
        raise ValueError('Finish the grid before registering extension')
    ledger = dict(previous['input_sha256'])
    audit.verify_files(grid.OUT,completed['files'],ledger)
    report_path = grid.OUT/'summary.json'
    report = audit.read(report_path)
    expected_gate = grid.eligibility(report['selected_adaptive'],report['uniform_selected_fixed'],
        report['original_off_adaptive_time_s'],previous['cached_D_mechanism']['same_actor_D_screen_passed'])
    if report['gate']!=expected_gate or not expected_gate['eligible4096']:
        raise ValueError('Complete grid and D-effect eligibility gate must pass')
    source = source_state()
    if any(source['source_files'].get(name)!=digest for name,digest in previous['source']['source_files'].items()):
        raise ValueError('Only additive new driver/test sources are allowed')
    parents, cache = {}, {}
    for method in METHODS:
        cache[method] = []
        for step in STEPS[:4]:
            path = matched.snapshot_path(parent_root(method),method,step)
            cache[method].append(_cached_node(path,method,step,ledger))
        path = ROOT/cache[method][-1]['checkpoint']
        payload = torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
        cfg = yaml.safe_load((path/'config.yaml').read_text(encoding='utf-8'))
        provenance = audit.read(path/'provenance.json')
        if payload['config']!=cfg or payload['source_sha256']!=provenance['source_sha256']:
            raise ValueError('Parent payload does not match original config/source')
        if payload['full_batch_version']!=1 or payload['decisions']!=1024 or payload['optimized_decisions']+len(payload['buffer']['states'])!=1024:
            raise ValueError('Import complete standard1024 checkpoint only')
        if any(source['source_files'].get(name)!=digest for name,digest in provenance['source_files'].items()):
            raise ValueError('Parent numerical/audit sources changed')
        parents[method] = dict(snapshot=path.relative_to(ROOT).as_posix(),
            checkpoint_sha256=audit.sha(path/'checkpoint.pt'),source_sha256=provenance['source_sha256'],
            snapshot_manifest_sha256=audit.sha(path/'snapshot_manifest.json'),
            learning_state_sha256=learning_digest(payload),config=cfg,
            parent_decisions=1024,parent_optimized_decisions=payload['optimized_decisions'],
            parent_pending_decisions=len(payload['buffer']['states']),
            parent_optimizer_updates=len(payload['updates']),
            parent_adam_steps_per_network=len(payload['updates'])*cfg['ppo']['epochs'],
            parent_training_wall_s=payload['training_wall_s'],parent_replay_wall_s=payload['replay_wall_s'],
            parent_physics_steps=payload['physics_before']+payload['current_physics_steps'],
            parent_command_steps=payload['commands_before']+payload['current_command_steps'])
    # The authoritative grid gate must rank precisely the verified caches.
    fixed,_ = grid.select_uniform_fixed({m:cache[m] for m in METHODS if m!='adaptive'})
    adaptive = matched.choose(cache['adaptive'])
    for expected,saved in ((fixed,report['uniform_selected_fixed']),(adaptive,report['selected_adaptive'])):
        for key in ('checkpoint_sha256','selected_step','success_count','complete_total_time_s'):
            if expected[key]!=saved[key]: raise ValueError('Grid selection is not bound to verified prefix nodes')
    for path in (grid.OUT/'protocol.json',grid.OUT/'delivery_manifest.json',report_path):
        ledger[path.relative_to(ROOT).as_posix()] = audit.sha(path)
    d_path=matched.OUT/'intervention_adaptive_step0_v1/protocol.json'
    d_protocol=audit.read(d_path)
    if d_protocol['checkpoint_sha256']!=cache['adaptive'][0]['checkpoint_sha256']:
        raise ValueError('Cached D intervention must identify the verified step0 actor')
    ledger[d_path.relative_to(ROOT).as_posix()]=audit.sha(d_path)
    ledger.update(source['source_files'])
    return dict(version='target-tracking-extension-v1',source=source,input_sha256=ledger,
        parent_grid_gate=expected_gate,parents=parents,cached_nodes=cache,methods=list(METHODS),seed=101,
        cached_D_mechanism=dict(checkpoint_sha256=d_protocol['checkpoint_sha256'],
            source_protocol=d_path.relative_to(ROOT).as_posix(),decision=previous['cached_D_mechanism']),
        starting_decisions=1024,final_decisions=4096,new_decisions_per_method=3072,total_new_decisions=12288,
        validation_milestones=list(STEPS),new_validation_milestones=list(NEW_STEPS),
        new_validation_count=24,cached_validation_count=48,
        migration='Strict old config/source restore, bitwise learning-state verification, explicit source-only migration in new atomic directory',
        unchanged='Original target tracking,26D state,reward,PPO,actor/critic,twoAdam,RNG,buffer,ordered coverage,1ms KCM,20ms decisions,v2brake',
        resume_cost='Replay current unfinished episode only; never sample or optimize prefix again; report replay separately',
        selection='All four methods finish4096; all six nodes eligible; full successes first, time only for3/3, earliest node then shorter fixed distance',
        budget_accounting='New3072 per method; parent1024 counted once; shared historical sensor priors unchanged; snapshots are not independent runs',
        no_new_pretraining=True,no_new_smoke=True,no_new_independent_seeds=True,
        five_seed_stage='Not implemented; separately registered full from-scratch4096+4096 chains required later',
        test_access=False,adopted=False)


def prepare():
    path = OUT/'protocol.json'
    if path.exists():
        protocol = audit.read(path)
        audit.verify_inputs(protocol)
        if protocol['source']!=source_state(): raise ValueError('Frozen extension source changed')
        return protocol
    protocol = preflight()
    immutable_json(path,protocol)
    with zipfile.ZipFile(OUT/'registered_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source']['source_files']: archive.write(ROOT/name,name)
    return protocol


def configuration(candidate,method,seed):
    if candidate!='preview_time' or method not in METHODS or seed!=101:
        raise ValueError('Only existing four seed101 prefixes may continue')
    # Budget/source metadata belongs to new protocol, not the exact old config.
    return copy.deepcopy(prepare()['parents'][method]['config'])


@contextmanager
def installed():
    global _active
    if _active: raise RuntimeError('Do not nest extension installation')
    names = ('OUT','configuration','source_state','prepare')
    old_full = {name:getattr(full,name) for name in names}
    old_core = {name:getattr(core,name) for name in (*names,'MacroEnvironment','make_agent')}
    with audit.corrected_execution():
        _active=True
        try:
            full.OUT=core.OUT=OUT
            full.configuration=core.configuration=configuration
            full.source_state=core.source_state=source_state
            full.prepare=core.prepare=prepare
            core.MacroEnvironment,core.make_agent=full.local.LocalCornerEnvironment,full.local.make_agent
            yield
        finally:
            for name,value in old_full.items(): setattr(full,name,value)
            for name,value in old_core.items(): setattr(core,name,value)
            _active=False


def _import_prefix(parent, run, protocol, method, env_factory=None, agent_factory=None):
    """Atomic, zero-decision migration; injectable factories are for unit tests."""
    parent,run=Path(parent),Path(run)
    entry=protocol['parents'][method]
    files=_snapshot_files(parent)
    if audit.sha(parent/'checkpoint.pt')!=entry['checkpoint_sha256'] or audit.sha(parent/'snapshot_manifest.json')!=entry['snapshot_manifest_sha256']:
        raise ValueError('Registered prefix checkpoint/manifest changed')
    original=torch.load(parent/'checkpoint.pt',map_location='cpu',weights_only=False)
    if original['config']!=entry['config'] or original['source_sha256']!=entry['source_sha256'] or learning_digest(original)!=entry['learning_state_sha256']:
        raise ValueError('Registered prefix config/source/learning state changed')
    if run.exists():
        migration=audit.read(run/'prefix_import.json')
        if migration['parent_checkpoint_sha256']!=entry['checkpoint_sha256'] or migration['new_source_sha256']!=protocol['source']['source_sha256']:
            raise ValueError('Existing imported run belongs to another prefix/source')
        payload=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
        if payload['config']!=entry['config'] or payload['source_sha256']!=protocol['source']['source_sha256'] or payload['decisions']<entry['parent_decisions']:
            raise ValueError('Existing imported run config/source/accounting changed')
        if payload['decisions']==entry['parent_decisions'] and learning_digest(payload)!=entry['learning_state_sha256']:
            raise ValueError('Imported learning state differs before any new decision')
        if audit.sha(run/'origin_step1024/snapshot_manifest.json')!=entry['snapshot_manifest_sha256']:
            raise ValueError('Preserved origin manifest changed')
        _snapshot_files(run/'origin_step1024')
        return run
    began=time.perf_counter()
    run.parent.mkdir(parents=True,exist_ok=True)
    staging=run.parent/f'.{run.name}.importing-{time.time_ns()}'
    staging.mkdir()
    # Keep an untouched self-contained copy beside the new runtime files.
    origin=staging/'origin_step1024';origin.mkdir()
    for name in (*files,'snapshot_manifest.json'):
        shutil.copyfile(parent/name,origin/name)
    resume_names=['checkpoint.pt','config.yaml','updates.jsonl','episodes.jsonl']
    resume_names += [item['file'] for item in original['trace_files']]
    for name in resume_names: shutil.copyfile(parent/name,staging/name)
    trainer=BASE_TRAINER(entry['config'],staging,
        env_factory or full.local.LocalCornerEnvironment,
        agent_factory or full.local.make_agent,entry['source_sha256'])
    # The frozen restore has already validated exact observation after replay
    # and restored all RNGs. Changing this explicit new attribution is not a
    # relaxation of its old-source guard or a modification of any old artifact.
    trainer.source_hash=protocol['source']['source_sha256']
    saved=trainer.save()
    if learning_digest(saved)!=entry['learning_state_sha256']:
        raise ValueError('Source migration altered model/Adam/RNG/buffer/observation')
    core.old.save_trace(staging/'training_final_partial.npz',trainer.env.trace)
    audit.write(staging/'training_summary.json',trainer.summary(saved))
    audit.write(staging/'provenance.json',dict(**protocol['source'],
        config_sha256=audit.sha(staging/'config.yaml'),protocol_sha256=audit.sha(OUT/'protocol.json'),
        command='python '+' '.join(sys.argv),parent_checkpoint_sha256=entry['checkpoint_sha256'],
        source_migration='Numerical state unchanged; source-only registered driver extension'))
    with zipfile.ZipFile(staging/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['source']['source_files']: archive.write(ROOT/name,name)
    audit.write(staging/'prefix_import.json',dict(method=method,
        parent_snapshot=entry['snapshot'],parent_checkpoint_sha256=entry['checkpoint_sha256'],
        old_source_sha256=entry['source_sha256'],new_source_sha256=protocol['source']['source_sha256'],
        old_and_new_learning_state_sha256=entry['learning_state_sha256'],
        imported_checkpoint_sha256=audit.sha(staging/'checkpoint.pt'),
        imported_decisions=trainer.decisions,new_training_decisions=0,
        parent_files_unchanged=all(audit.sha(parent/name)==digest for name,digest in files.items()),
        import_wall_s=time.perf_counter()-began,
        new_environment_replay_wall_s=trainer.replay_wall_s-original['replay_wall_s'],
        replayed_current_episode_actions=len(original['actions']),
        equivalence_fields=list(RESUME_KEYS)))
    if not audit.read(staging/'prefix_import.json')['parent_files_unchanged']:
        raise ValueError('Parent changed while importing')
    if staging.resolve().parent!=run.resolve().parent:
        raise ValueError('Import publication must remain in its intended parent')
    staging.replace(run)
    return run


def import_method(method):
    if method not in METHODS: raise ValueError('Registered methods only')
    protocol=prepare();entry=protocol['parents'][method]
    with installed():
        return _import_prefix(ROOT/entry['snapshot'],matched.snapshot_path(OUT,method,0).parents[1],protocol,method)


def worker(method,step):
    if method not in METHODS or step not in NEW_STEPS:
        raise ValueError('Only cumulative2048/4096 for existing seed101 methods')
    run=OUT/f'development/preview_time_{method}_seed101'
    if not (run/'prefix_import.json').exists():
        raise ValueError('Import the registered1024 prefix before training')
    protocol=prepare()
    if step==4096 and not (run/'checkpoints/step2048/validation/summary.json').exists():
        raise ValueError('Complete2048 snapshot and all validation before4096')
    torch.set_num_threads(1)
    import_method(method)  # Idempotent lineage check; advanced checkpoints are retained.
    with installed():
        snapshot=full.train(method,101,step,'development')
        if not (snapshot/'validation/summary.json').exists(): core.evaluate(snapshot,snapshot/'validation')
        rows=audit.cached_rows(snapshot/'validation',{})
        grid.validate_cached_binding(snapshot,rows,step,method)
    prepare()
    summary=audit.read(snapshot/'training_summary.json')
    immutable_json(snapshot/'extension_costs.json',extension_costs(summary,protocol['parents'][method]))
    return snapshot


def extension_costs(summary,parent):
    return dict(cumulative_decisions=summary['decisions'],
        imported_decisions=parent['parent_decisions'],new_training_decisions=summary['decisions']-parent['parent_decisions'],
        cumulative_training_wall_s=summary['training_wall_s'],
        additional_training_wall_s=summary['training_wall_s']-parent['parent_training_wall_s'],
        additional_replay_wall_s=summary['environment_replay_wall_s']-parent['parent_replay_wall_s'],
        new_physics_steps=summary['physics_steps']-parent['parent_physics_steps'],
        new_command_steps=summary['command_steps']-parent['parent_command_steps'],
        cumulative_optimizer_updates=summary['optimizer_updates'],
        new_optimizer_updates=summary['optimizer_updates']-parent['parent_optimizer_updates'],
        new_adam_steps_per_network=summary['adam_steps_per_network']-parent['parent_adam_steps_per_network'],
        checkpoint_sha256=summary['checkpoint_sha256'])


def choose(nodes):
    if {n['step'] for n in nodes}!=set(STEPS) or len(nodes)!=len(STEPS):
        raise ValueError('All six registered nodes are required')
    for n in nodes: audit.require_paths(n['rows'])
    best=nodes[0]
    def count(n): return sum(bool(r['success']) for r in n['rows'])
    def total(n): return sum(r['elapsed_time_s'] for r in n['rows'])
    for n in nodes[1:]:
        if count(n)>count(best) or (count(n)==count(best)==3 and total(n)<total(best)-1e-9) or (
            count(n)==count(best) and (count(n)<3 or abs(total(n)-total(best))<=1e-9) and n['step']<best['step']):
            best=n
    return dict(selected_step=best['step'],checkpoint=best['checkpoint'],checkpoint_sha256=best['checkpoint_sha256'],
        success_count=count(best),complete_total_time_s=total(best) if count(best)==3 else None,
        rows=best['rows'],all_nodes=nodes,available_budget=4096,cache_reused=best['cache_reused'])


def selected_D_gate(adaptive,protocol):
    cached=protocol['cached_D_mechanism']
    matches=adaptive['checkpoint_sha256']==cached['checkpoint_sha256']
    passed=bool(matches and cached['decision']['same_actor_D_screen_passed'])
    return dict(selected_checkpoint_sha256=adaptive['checkpoint_sha256'],
        cached_checkpoint_sha256=cached['checkpoint_sha256'],cached_actor_matches=matches,
        same_actor_D_gate_passed=passed,status='cached_same_actor_passed' if passed else 'pending_selected_actor_12_closed_loops',
        applicable_gain_pct=cached['decision']['learned_vs_best_complete_lock_time_gain_pct'] if passed else None,
        new_rollouts_required=0 if passed else 12)


def report():
    protocol=prepare();groups={};costs={}
    for method in METHODS:
        run=OUT/f'development/preview_time_{method}_seed101'
        final=audit.read(run/'training_summary.json')
        if final['decisions']!=4096: raise ValueError('All four methods must finish4096 before selection')
        nodes=copy.deepcopy(protocol['cached_nodes'][method])
        for step in NEW_STEPS:
            path=matched.snapshot_path(OUT,method,step)
            ledger={};node=_cached_node(path,method,step,ledger)
            node['cache_reused']=False;nodes.append(node)
        groups[method]=choose(nodes)
        costs[method]=audit.read(run/'checkpoints/step4096/extension_costs.json')
    fixed_method=min((m for m in METHODS if m!='adaptive'),key=lambda m:(
        -groups[m]['success_count'],groups[m]['complete_total_time_s'] if groups[m]['success_count']==3 else 0.,
        groups[m]['selected_step'],grid.FIXED_DISTANCES[m]))
    adaptive,fixed=groups['adaptive'],groups[fixed_method]
    full_success=adaptive['success_count']==fixed['success_count']==3
    result=dict(groups=groups,uniform_fixed_method=fixed_method,
        adaptive_beats_uniform_fixed=bool(full_success and adaptive['complete_total_time_s']<fixed['complete_total_time_s']-1e-9),
        gain_vs_fixed_pct=(1-adaptive['complete_total_time_s']/fixed['complete_total_time_s'])*100 if full_success else None,
        selected_actor_D_gate=selected_D_gate(adaptive,protocol),
        costs=costs,new_training_decisions=12288,new_validation_count=24,cached_validation_count=48,
        no_new_pretraining=True,no_independent_repeatability_claim=True,test_access=False,adopted=False,
        next_gate='No automatic formal run; selected actor requires an applicable D-effect check and separately registered full independent training')
    immutable_json(OUT/'summary.json',result)
    return result


def _scheduled_child(command,method,step=None):
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),command,'--method',method]
    if step is not None:argv += ['--step',str(step)]
    label=f'scheduled_{method}_{command}'+(str(step) if step is not None else '')
    directory=OUT/'invocations';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{label}.json'
    if path.exists():raise FileExistsError('Preserve scheduled invocation; resume the exact worker separately')
    record=dict(argv=argv,command=subprocess.list2cmdline(argv),cwd=str(ROOT),parent_pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),status='starting')
    audit.write(path,record);started=time.perf_counter()
    with (directory/f'{label}.stdout.log').open('x',encoding='utf-8') as stdout,(
            directory/f'{label}.stderr.log').open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        record.update(pid=process.pid,status='running');audit.write(path,record)
        code=process.wait()
    record.update(exit_code=code,status='complete' if code==0 else 'failed',wall_s=time.perf_counter()-started,
        finished_utc=datetime.now(timezone.utc).isoformat());audit.write(path,record)
    print(f'{label}: exit={code}, wall={record["wall_s"]:.3f}s',flush=True)
    if code:raise RuntimeError(f'{label} failed; retain partial outputs and resume this exact worker')
    return record


def run():
    prepare()
    for method in METHODS:_scheduled_child('import',method)
    # A stage barrier keeps every method at2048 before any continuation4096.
    # Each child owns its module bindings/RNG; at most two children run at once.
    for step in NEW_STEPS:
        for start in range(0,len(METHODS),2):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(_scheduled_child,'worker',method,step) for method in METHODS[start:start+2]]
                results=[future.result() for future in futures]
    return report()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','import','worker','report','run'))
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--step',type=int,choices=NEW_STEPS)
    args=parser.parse_args()
    torch.set_num_threads(1)
    log=OUT/'invocations'/('coordinator_run.json' if args.command=='run' else f'{time.time_ns()}_{args.command}.json')
    if log.exists():raise FileExistsError('Preserve previous coordinator; resume exact import/worker/report separately')
    record=dict(argv=[sys.executable,*(['-B'] if sys.dont_write_bytecode else []),*sys.argv],
        pid=os.getpid(),parent_pid=os.getppid(),cwd=str(Path.cwd()),
        actual_interpreter_flag_B=bool(sys.dont_write_bytecode),
        started_utc=datetime.now(timezone.utc).isoformat(),status='running')
    audit.write(log,record);start=time.perf_counter()
    try:
        if args.command=='prepare':
            prepare();result=dict(protocol=str(OUT/'protocol.json'),training_started=False)
        elif args.command=='import': result=dict(run=str(import_method(args.method)),new_training_decisions=0)
        elif args.command=='worker': result=dict(snapshot=str(worker(args.method,args.step)))
        else:
            value=run() if args.command=='run' else report()
            result={k:value[k] for k in ('uniform_fixed_method','adaptive_beats_uniform_fixed','gain_vs_fixed_pct')}
        record.update(status='complete',exit_code=0,result=result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as error:
        record.update(status='failed_or_interrupted',exit_code=1,error=repr(error));raise
    finally:
        record.update(wall_s=time.perf_counter()-start,finished_utc=datetime.now(timezone.utc).isoformat());audit.write(log,record)


if __name__=='__main__':main()
