"""Evaluate only the missing locked-preview controls of selected v7 actors."""
import importlib.util
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from joint_policy import JointPolicy
from joint_runtime_v5 import environment_factory
from smooth_policy_candidate import SmoothExplorationPolicy
from study_io import BASE, ROOT, register, receipt, write_json, record_progress, sha
from summarize_conditioned_training import diagnostics
from train_pilot import evaluate


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    names = ('joint_independent_refinement_v7', 'joint_gsde_refinement_v7')
    selections = [(BASE / name, read(BASE / name / 'selection.json')['selected']) for name in names]
    files = ['joint_policy.py', 'joint_runtime.py', 'joint_runtime_v2.py',
             'joint_runtime_v3.py', 'joint_runtime_v5.py', 'terminal_candidate_runtime_v1.py',
             'geometry_spec.py', 'train_pilot.py', 'validate_actions.py', 'smooth_policy_candidate.py']
    sources = [Path(__file__), BASE / 'v7_lock_design_v1.md', BASE / 'study_io.py',
               BASE / 'learning_preparation_v4/paths.json', BASE / 'summarize_conditioned_training.py',
               BASE / 'v7_complete_curve_audit_v1/results.json'] + [BASE / name for name in files]
    models, old_groups = [], []
    for folder, selected in selections:
        assert selected['successes'] == 12
        model = folder / ('final.pt' if selected['tag'] == 'final_validation' else f"checkpoint_{selected['decisions']}.pt")
        analysis = BASE / f"v7_checkpoint_{selected['decisions']}_analysis_v1/results.json"
        old = next(g for g in read(analysis)['groups'] if g['stage'] == folder.name)
        assert old['tag'] == selected['tag'] and old['successes'] == 12
        bindings = read(folder / 'registration.json')['source_sha256']
        for name in files:
            p = BASE / name
            assert sha(p) == bindings[p.relative_to(ROOT).as_posix()], name
        sources += [model, analysis, folder / 'selection.json', folder / 'receipt.json',
                    folder / selected['tag'] / 'summary.json']
        models.append(model)
        old_groups.append(old)
    preflight_file = ROOT / '论文项目/revision_20260910_continued/root_cause_iteration_v1/experiment_preflight_v1.py'
    spec = importlib.util.spec_from_file_location('stage_guard', preflight_file)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    stage_name = 'v7_selected_lock3_v1'
    check = guard.check_new_stage(BASE / stage_name, ['registration.json', 'receipt.json', 'results.json', 'failure.json'],
        {p.relative_to(ROOT).as_posix(): sha(p) for p in models})
    stage = register(stage_name, sources + [preflight_file],
        dict(model_loads=2, model_constructions=2, evaluation_runs=24, wall_s=600,
             actor_calls_upper=50000, physics_ticks_upper=900000, training_decisions=0, optimization_steps=0),
        'R02: newly selected v7 actors require missing same-actor locked-D attribution; reuse all existing original evaluations.')
    write_json(stage / 'preflight.json', check)
    costs = dict(evaluation_actor_calls=0, evaluation_physics_ticks=0)
    started = time.perf_counter()
    groups = []
    torch.set_num_threads(1)
    paths = [np.asarray(p, float) for p in read(BASE / 'learning_preparation_v4/paths.json')['validation']]
    assert len(paths) == 12

    def bounded_environment(reference, **kwargs):
        if costs['evaluation_actor_calls'] >= 50000 or costs['evaluation_physics_ticks'] >= 900000:
            raise RuntimeError('Registered physical evaluation budget reached')
        return environment_factory(reference, **kwargs)

    try:
        for model, old, (folder, selected) in zip(models, old_groups, selections):
            incoming = torch.load(model, map_location='cpu', weights_only=False)
            assert incoming['mode'] == 'joint' and incoming['observation_dim'] == 72
            cls = SmoothExplorationPolicy if incoming['sampler'] == 'gsde8' else JointPolicy
            policy = cls(72, mode='joint')
            policy.load_state_dict(incoming['policy'])
            policy.eval()
            tag = 'locked_' + incoming['sampler']
            result = evaluate(policy, paths, stage, tag, 3., costs, started + 600., bounded_environment)
            for key, value in incoming['policy'].items():
                assert torch.equal(value, policy.state_dict()[key]), key
            cases = []
            originals = {c['result']['path']: c for c in old['cases']}
            for row in result:
                trace = stage / tag / f"{row['path']}.npz"
                with np.load(trace) as data:
                    assert np.all(data['actions'][:, 2] == 3.)
                    diag = diagnostics(data['reference'], data['rows'], data['actions'])
                original = originals[row['path']]
                cases.append(dict(path=row['path'], original=original, locked_result=row,
                    locked_diagnostics=diag, locked_trace_sha256=sha(trace),
                    locked_minus_adaptive_s=(row['completion_time_s'] - original['result']['completion_time_s']) if row['success'] else None))
            successes = sum(r['success'] for r in result)
            total = sum(r['completion_time_s'] for r in result) if successes == 12 else None
            groups.append(dict(stage=folder.name, selected=selected, original_full_time_s=old['full_time_s'],
                locked_successes=successes, locked_full_time_s=total,
                locked_minus_adaptive_s=total-old['full_time_s'] if total is not None else None, cases=cases))
            write_json(stage / 'progress.json', dict(groups=groups, costs=costs))
        assert costs['evaluation_actor_calls'] <= 50000 and costs['evaluation_physics_ticks'] <= 900000
        output = dict(groups=groups, costs=costs, sealed_test_opened=False, training_decisions=0,
            optimization_steps=0, independent_seed_claim=False, goal_complete=False,
            scope='Closed-loop intervention at D=3 mm on two selected joint actors; not a retrained fixed-preview baseline or a continuous-D optimum.')
        write_json(stage / 'results.json', output)
        lines = ['# 所选v7同actor锁D完整开发核查', '',
                 '| 来源 | 自适应成功 | 锁3成功 | 自适应/s | 锁3/s | 锁3减自适应/s |',
                 '|---|---:|---:|---:|---:|---:|']
        for group in groups:
            fmt = lambda x: '不可全量排名' if x is None else f'{x:.3f}'
            lines.append(f"| {group['stage']} | 12/12 | {group['locked_successes']}/12 | {fmt(group['original_full_time_s'])} | {fmt(group['locked_full_time_s'])} | {fmt(group['locked_minus_adaptive_s'])} |")
        lines += ['', output['scope'], '', '原模型所有张量不变，原自适应评价只复用；全部逐路径结果、四角诊断与失败保留。']
        (stage / '同actor锁D核查.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        receipt(stage, dict(status='completed_selected_v7_lock3_evaluation', costs=costs, evaluation_runs=24,
            model_loads=2, model_constructions=2, wall_s=time.perf_counter()-started, training_decisions=0,
            optimization_steps=0, goal_complete=False))
        record_progress(stage_name, [stage / 'receipt.json', stage / 'results.json'],
            '两份所选v7联合actor的缺失锁D3评价已完成，原自适应轨迹直接复用。按实际结果解释D作用，不能代替独立训练的强固定对照；与λ1完整负结果及真实梯度一起接入报告和英文工作稿。最终方案与20–25页投稿全文仍未完成。')
        print('\n'.join(lines))
    except Exception:
        write_json(stage / 'failure.json', dict(error=traceback.format_exc(), costs=costs,
            completed_groups=len(groups), wall_s=time.perf_counter()-started, automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
