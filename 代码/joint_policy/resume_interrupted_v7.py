"""Continue a reboot-interrupted v7 run using its frozen training loop.

The registered source is executed unchanged except for the starting decision,
the conservative final decision, and the new journal file. No initialization,
completed evaluation, or past optimizer update is repeated.
"""
import argparse
import ast
import copy
import json
import os
import runpy
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from study_io import BASE, ROOT, register, receipt, sha, write_json
from resume_codec_v5 import load_resume, save_resume

EXPECTED_SOURCE = '0fd61ac7a260e2ab6e90232279bffbd4673a0340ab90161c9a3ae82a0fde4e0e'
STAGES = ['joint_independent_refinement_v7', 'joint_gsde_refinement_v7',
          'fixed3_independent_refinement_v7', 'fixed3_gsde_refinement_v7']


def frozen_continuation(source):
    """Reuse the actual registered loop; limit edits to restart bookkeeping."""
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.If))
    body = next(n for n in main.body if isinstance(n, ast.Try)).body
    functions = [copy.deepcopy(n) for n in body
                 if isinstance(n, ast.FunctionDef) and n.name in ('model_state', 'checkpoint')]
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.With)
                 and ast.unparse(n.items[0].context_expr.func) == 'gzip.open')
    continuation = copy.deepcopy(body[start:])
    loop = continuation[0].body[0]
    assert isinstance(loop, ast.For) and ast.unparse(loop.iter) == 'range(1, 360001)'
    loop.iter = ast.parse('range(resume_decision + 1, last_training_decision + 1)', mode='eval').body
    continuation[0].items[0].context_expr = ast.parse(
        "gzip.open(recovery_journal, 'wt', encoding='utf-8')", mode='eval').body
    final_checks = 0
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'evaluate_now' for t in node.targets):
            node.value = ast.parse('decision % 60000 == 0 or decision == last_training_decision', mode='eval').body
        if isinstance(node, ast.Compare) and ast.unparse(node) == 'decision == 360000':
            node.comparators[0] = ast.Name(id='last_training_decision', ctx=ast.Load())
            final_checks += 1
    assert final_checks == 1
    # The final pointer must describe the actual last saved decision.
    for node in ast.walk(continuation[-2]):
        if isinstance(node, ast.keyword) and node.arg == 'decision':
            node.value = ast.Name(id='last_training_decision', ctx=ast.Load())
    # Recovery has its own truthful progress note after the original receipt.
    assert isinstance(continuation[-1], ast.Expr) and ast.unparse(continuation[-1].value.func) == 'record_progress'
    continuation.pop()
    return compile(ast.fix_missing_locations(ast.Module(body=functions + continuation, type_ignores=[])), str(source), 'exec')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--interrupt-utc', required=True)
    args = parser.parse_args()
    original = BASE/args.stage
    source = BASE/'train_optimizer_refinement_v7.py'
    if sha(source) != EXPECTED_SOURCE:
        raise ValueError('Original v7 source changed')
    continuation_code = frozen_continuation(source)
    if any((original/name).exists() for name in ('receipt.json', 'failure.json')):
        raise RuntimeError('Completed/failed stage requires a separate decision, not automatic recovery')
    registration = json.loads((original/'registration.json').read_text(encoding='utf-8'))
    for path, expected in registration['source_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Registered source changed: '+path)
    storage = json.loads((original/'storage.json').read_text(encoding='utf-8'))
    volatile = Path(storage['rolling_state_directory'])
    pointer = json.loads((volatile/'resume_pointer.json').read_text(encoding='utf-8'))
    if pointer['phase'] != 'training' or pointer['training_complete']:
        raise RuntimeError('Only a completed training macro may be restored by this entry point')
    backup = volatile/'before_recovery_20260915_v3'
    backup.mkdir(exist_ok=False)
    for name in ('resume_pointer.json', 'resume_0.pt', 'resume_1.pt'):
        shutil.copy2(volatile/name, backup/name)
    recovery = register(args.stage+'_reboot_recovery_v3',
        [Path(__file__), source, original/'registration.json', original/'storage.json',
         original/'training_progress.json', original/'validations.json'],
        dict(original_total_decisions=360000, maximum_uncommitted_decisions=1,
             last_saved_training_decision=359999, original_runtime_wall_s=18000,
             completed_evaluation_repeats=0, initialization_repeats=0),
        'Resume preserved v7 after reboot; reserve one uncertain uncommitted macro inside original budget. Downtime is excluded; prior active runtime is conservatively charged through reboot.')
    snapshot = backup/f"resume_{pointer['slot']}.pt"
    state = load_resume(snapshot)
    assert state['decision'] == pointer['decision'] == state['costs']['training_actor_calls']
    assert state['phase'] == 'training' and state['observation_dim'] == 72
    latest = max(v['decisions'] for v in state['validations'])
    assert latest == 120000 and not any((original/f'validation_{n}').exists() for n in range(180000, 360000, 60000))
    assert len(state['episodes']) == state['episode_index']
    original_episode_directory = Path(storage['training_episode_directory'])
    uncommitted_episode = original_episode_directory/f"episode_{state['episode_index']}.npz"
    episode_directory = volatile/'episodes_recovered_20260915_v3'
    episode_directory.mkdir(exist_ok=False)
    interruption = datetime.fromisoformat(args.interrupt_utc)
    prior_wall = (interruption-datetime.fromisoformat(registration['created_utc'])).total_seconds()
    assert 0 < prior_wall < 18000
    write_json(recovery/'restoration.json', dict(
        snapshot=str(snapshot), snapshot_sha256=sha(snapshot), pointer=pointer,
        prior_runtime_wall_upper_s=prior_wall, remaining_runtime_wall_s=18000-prior_wall,
        original_journal_preserved=str(volatile/'decisions.jsonl.gz'),
        original_uncommitted_episode_preserved=str(uncommitted_episode) if uncommitted_episode.exists() else None,
        recovered_episode_directory=str(episode_directory),
        uncommitted_cost_upper_bound=dict(actor_calls=1, physical_ticks=2120, ppo_updates=1),
        recovery_pid=os.getpid(), recovered_utc=datetime.now(timezone.utc).isoformat(),
        original_settings_unchanged=True, goal_complete=False))
    namespace = runpy.run_path(str(source))
    torch.set_num_threads(1)
    mode, sampler = state['mode'], state['sampler']
    policy = (namespace['SmoothExplorationPolicy'](72, mode=mode, noise_interval=8)
              if sampler == 'gsde8' else namespace['JointPolicy'](72, mode=mode))
    policy.load_state_dict(state['policy'])
    optimizer = namespace['optimizers'](policy)
    for key, opt in optimizer.items():
        opt.load_state_dict(state['optimizer'][key])
    assert optimizer['actor'].param_groups[0]['lr'] == 3e-5
    assert optimizer['critic'].param_groups[0]['lr'] == 1e-3
    paths = json.loads((BASE/'learning_preparation_v4/paths.json').read_text(encoding='utf-8'))
    train_paths = [np.asarray(ref, float) for ref in paths['training']]
    valid_paths = [np.asarray(ref, float) for ref in paths['validation']]
    fixed_d = 3. if mode == 'fixed' else None
    env = namespace['environment_factory'](train_paths[state['path_index']], fixed_d=fixed_d)
    env.restore(state['environment'])
    assert env.observation().shape == (72,) and not env.done
    if state['batch']['reward'] and not state['batch']['terminal'][-1]:
        restored_value = float(policy.value(torch.tensor(env.observation())).detach())
        assert abs(restored_value-state['batch']['next_value'][-1]) < 1e-6
    rng = np.random.default_rng()
    rng.bit_generator.state = state['generator_rng']
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    started = time.perf_counter()
    def save_with_runtime(path, value):
        value['runtime_wall_upper_s'] = prior_wall + time.perf_counter()-started
        save_resume(path, value)
    namespace.update({key: state[key] for key in ('costs', 'episodes', 'updates', 'validations',
        'path_index', 'episode_index', 'episode_return', 'episode_rows', 'batch')})
    namespace.update(args=SimpleNamespace(stage=args.stage, mode=mode, sampler=sampler),
        stage=original, volatile=volatile, episode_directory=episode_directory,
        incoming=state, policy=policy, optimizer=optimizer, env=env, fixed_d=fixed_d,
        train_paths=train_paths, valid_paths=valid_paths, rng=rng, phase='training',
        started=started-prior_wall, deadline=started+18000-prior_wall,
        resume_decision=state['decision'], last_training_decision=359999,
        recovery_journal=volatile/'decisions_recovered_20260915_v3.jsonl.gz', save_resume=save_with_runtime)
    write_json(recovery/'started.json', dict(decision=state['decision'], pid=os.getpid(),
        actor_critic_optimizer_rng_environment_and_buffer_restored=True,
        critic_restore_check_calls=int(bool(state['batch']['reward'])), training_decisions=0))
    try:
        exec(continuation_code, namespace)
        receipt(recovery, dict(status='original_budget_continuation_completed',
            from_decision=state['decision'], through_decision=359999,
            original_receipt_sha256=sha(original/'receipt.json'), goal_complete=False))
    except Exception:
        write_json(recovery/'failure.json', dict(traceback=traceback.format_exc(),
            costs=namespace['costs'], automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
