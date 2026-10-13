"""Close the single preview-noise comparison from saved evidence only."""
import gzip
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json, record_progress, sha
from summarize_conditioned_training import diagnostics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def coverage(path):
    rows = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(row)
            if row['decision'] == 60000:
                break
    assert [r['decision'] for r in rows] == list(range(1, 60001))
    held = np.asarray([r['held'] for r in rows])
    preview, feed = held[:, 2], held[:, 1]
    counts = {}
    for threshold in (4.6, 6., 8.):
        mask = preview >= threshold
        current = longest = 0
        for i, yes in enumerate(mask):
            if i and rows[i]['episode'] != rows[i-1]['episode']:
                current = 0
            current = current + 1 if yes else 0
            longest = max(longest, current)
        counts[str(threshold)] = dict(count=int(mask.sum()),
            fraction=float(mask.mean()), longest_within_episode=longest)
    ended = [r for r in rows if r['terminal']]
    reasons = {}
    for row in ended:
        reasons[row['reason']] = reasons.get(row['reason'], 0) + 1
    return dict(decisions=len(rows), first=1, last=60000,
        preview_quantiles_mm=np.quantile(preview, [0., .01, .1, .5, .9, .99, 1.]).tolist(),
        feed_quantiles_mm_s=np.quantile(feed, [.01, .1, .5, .9, .99]).tolist(),
        coverage=counts, ended_episodes=len(ended), termination_reasons=reasons,
        scope='Global changing-policy actions; not conditional corner coverage or independent reliability trials.')


def main():
    folder = BASE/'wide_preview_independent_v1'
    original = BASE/'joint_independent_refinement_v7'
    reference = BASE/'v7_checkpoint_60000_analysis_v1'
    required = [folder/name for name in ('receipt.json', 'selection.json',
        'validations.json', 'training_progress.json', 'initial_reuse.json',
        'registration.json', 'storage.json', 'final_validation/summary.json')]
    if not all(p.exists() for p in required):
        raise RuntimeError('The registered experiment is not complete; do not rerun or read an active journal')
    storage = read(folder/'storage.json')
    external = Path(storage['rolling_state_directory'])
    pointer = read(external/'resume_pointer.json')
    assert pointer['training_complete'] and pointer['decision'] == 60000
    sources = [Path(__file__), BASE/'preview_exploration_design_v1.md',
        BASE/'summarize_conditioned_training.py', BASE/'study_io.py',
        reference/'receipt.json', reference/'results.json',
        original/'storage.json', original/'receipt.json', *required]
    sources += [folder/'final_validation'/f'{i}.npz' for i in range(12)]
    paths = [Path(read(original/'storage.json')['decision_journal']), Path(storage['decision_journal'])]
    bindings = [dict(role=role, path=str(path), sha256=sha(path))
        for role, path in zip(('narrow_initial_60k', 'wide_initial_60k'), paths)]
    reg, rec = read(folder/'registration.json')['budget'], read(folder/'receipt.json')
    assert reg['seed'] == 713 and reg['initial_preview_std'] == .4 and reg['gae_lambda'] == .98
    assert reg['training_decisions'] == rec['costs']['training_actor_calls'] == 60000
    assert reg['sampler'] == 'independent' and reg['fixed_d'] is None
    assert rec['wall_s'] <= reg['wall_s']
    vals = read(folder/'validations.json')['validations']
    assert [v['decisions'] for v in vals] == [0, 60000]
    curves = []
    for v in vals:
        assert sorted(r['path'] for r in v['results']) == list(range(12))
        n = sum(r['success'] for r in v['results'])
        curves.append(dict(decisions=v['decisions'], tag=v['tag'], successes=n,
            complete_time_sum_s=sum(r['completion_time_s'] for r in v['results']) if n == 12 else None))
    best = max(curves, key=lambda r: (r['successes'],
        -r['complete_time_sum_s'] if r['complete_time_sum_s'] is not None else -r['decisions']))
    assert best == rec['selected'] == read(folder/'selection.json')['selected']
    assert read(folder/'final_validation/summary.json')['results'] == vals[-1]['results']
    stage = register('preview_exploration_60000_analysis_v1', sources,
        dict(new_saved_trace_analyses=12, new_journal_windows=2,
             analyzed_journal_rows=120000, model_calls=0, physics_ticks=0, training_decisions=0),
        'Registered endpoint: compare only D initial standard deviation, retaining initial selection and strong fixed controls.')
    write_json(stage/'external_sources.json', bindings)
    cases = []
    for row in vals[-1]['results']:
        path = folder/'final_validation'/f"{row['path']}.npz"
        with np.load(path) as data:
            diag = diagnostics(data['reference'], data['rows'], data['actions'])
            speed = np.linalg.norm(data['rows'][:, 2:4], axis=1)
            for corner in diag['corners']:
                vertex = data['reference'][corner['vertex_index']]
                local = np.linalg.norm(data['rows'][:, :2]-vertex, axis=1) < 10.
                corner['median_speed_within_10mm'] = float(np.median(speed[local])) if local.any() else None
        cases.append(dict(result=row, diagnostics=diag, trace_sha256=sha(path)))
    old = read(reference/'results.json')['groups']
    comparisons = []
    for name in ('joint_independent_refinement_v7', 'fixed3_independent_refinement_v7'):
        base = next(g for g in old if g['stage'] == name)
        n = curves[-1]['successes']
        total = curves[-1]['complete_time_sum_s']
        paired = [dict(path=c['result']['path'], reference_s=c['result']['completion_time_s'],
            wide_s=next(r['completion_time_s'] for r in vals[-1]['results'] if r['path'] == c['result']['path']))
            for c in base['cases'] if c['result']['success'] and
                next(r['success'] for r in vals[-1]['results'] if r['path'] == c['result']['path'])]
        comparisons.append(dict(reference_stage=name, reference_successes=base['successes'],
            wide_successes=n, reference_full_time_s=base['full_time_s'], wide_full_time_s=total,
            wide_minus_reference_s=total-base['full_time_s'] if n == base['successes'] == 12 else None,
            endpoint_support=n == base['successes'] == 12 and total < base['full_time_s'], paired=paired))
    logs = [dict(role=b['role'], **coverage(path)) for path, b in zip(paths, bindings)]
    for path, binding in zip(paths, bindings):
        assert sha(path) == binding['sha256']
    progress = read(folder/'training_progress.json')
    updates = progress['updates']
    assert len(updates) == rec['costs']['ppo_updates']
    for key in ('actor_minibatches', 'critic_minibatches'):
        assert sum(u['statistics'][key] for u in updates) == rec['costs'][key]
    result = dict(curves=curves, selected=best, comparisons=comparisons, cases=cases,
        exploration=logs, training_costs=rec['costs'], wall_s=rec['wall_s'],
        actor_early_stops=sum(u['statistics']['actor_early_stop'] for u in updates),
        completed_training_episodes=len(progress['episodes']),
        training_successes=sum(e['success'] for e in progress['episodes']),
        pre_registered_gate_passed=comparisons[0]['endpoint_support'],
        automatic_extension=False, shape_visual_review_pending=True,
        independent_full_seed_replications=False, sealed_test_opened=False, goal_complete=False)
    write_json(stage/'results.json', result)
    lines = ['# 前瞻初始探索范围：60k单因素结果', '',
        '| 比较 | 原设置成功 | 宽探索成功 | 原时间/s | 宽探索时间/s | 宽探索减原/s |',
        '|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: '不可全量排名' if x is None else f'{x:.3f}'
    for c in comparisons:
        lines.append(f"| {c['reference_stage']} | {c['reference_successes']}/12 | {c['wide_successes']}/12 | {fmt(c['reference_full_time_s'])} | {fmt(c['wide_full_time_s'])} | {fmt(c['wide_minus_reference_s'])} |")
    lines += ['', f"所选节点：{best['decisions']}；完整时间 {fmt(best['complete_time_sum_s'])} s。原0/60k选择机会保留。",
        '', '两份日志均只分析初始1–60000决策，区别于旧所选节点之前的60k覆盖。新训练仅改变D初始标准差；固定有效动作不变，复用原对照。覆盖和失败频率是变化策略下的描述统计，不能替代任务效率或独立可靠性。',
        '', '只读12条新保存轨迹与两份日志窗口，没有新模型、物理或训练。没有自动扩预算或组合；图形需另行目视验收。']
    (stage/'探索范围比较.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_preview_exploration_endpoint_audit',
        new_saved_trace_analyses=12, analyzed_journal_rows=120000, model_calls=0,
        physics_ticks=0, training_decisions=0, goal_complete=False))
    record_progress(stage.name, [stage/'receipt.json', stage/'results.json', stage/'external_sources.json'],
        'D初始std .40单因素已完成60k并核查预算、0/60k选择、完整运动与匹配窄探索/固定对照；新覆盖分析只读两组初始60k日志。按真实结论更新报告和英文稿、检查四角图及全PDF；无自动扩预算，封存测试未打开，总目标未完成。')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
