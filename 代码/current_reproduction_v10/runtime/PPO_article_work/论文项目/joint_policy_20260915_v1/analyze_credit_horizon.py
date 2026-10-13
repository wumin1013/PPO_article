"""Compare a new lambda-one checkpoint with the already analyzed base."""
import argparse
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json, record_progress, sha
from summarize_conditioned_training import diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decisions', type=int, choices=(30000, 60000), required=True)
    args = parser.parse_args()
    old = BASE/f'minimal_factors_{args.decisions}_analysis_v1'
    old_result = json.loads((old/'results.json').read_text(encoding='utf-8'))
    tag = 'final_validation' if args.decisions == 60000 else 'validation_30000'
    folders = [BASE/f'credit_horizon_{mode}_pilot_v1' for mode in ('joint','fixed')]
    sources = [Path(__file__), BASE/'credit_horizon_design_v1.md',
        BASE/'summarize_conditioned_training.py', old/'receipt.json', old/'results.json']
    for folder in folders:
        required = [folder/'registration.json', folder/tag/'summary.json']
        if args.decisions == 60000:
            required += [folder/'receipt.json', folder/'selection.json']
        if not all(p.exists() for p in required):
            raise RuntimeError('Wait for the common registered checkpoint and final receipts')
        sources += required + [folder/tag/f'{i}.npz' for i in range(12)]
    stage = register(f'credit_horizon_{args.decisions}_analysis_v1', sources,
        dict(saved_trace_files=24, model_calls=0, physics_ticks=0, training_decisions=0),
        'Same-budget lambda comparison against reused matched base; retain failures and stronger pretrained references.')
    groups, comparisons = [], []
    for mode, folder in zip(('joint','fixed'), folders):
        raw = json.loads((folder/tag/'summary.json').read_text(encoding='utf-8'))['results']
        assert sorted(r['path'] for r in raw) == list(range(12))
        cases = []
        for r in raw:
            p = folder/tag/f"{r['path']}.npz"
            with np.load(p) as d:
                info = diagnostics(d['reference'], d['rows'], d['actions'])
            cases.append(dict(result=r, diagnostics=info, trace_sha256=sha(p)))
        n = sum(r['success'] for r in raw)
        total = sum(r['completion_time_s'] for r in raw) if n == 12 else None
        base = next(g for g in old_result['groups'] if g['arm'] == 'base' and g['mode'] == mode)
        both = n == base['successes'] == 12
        old_cases = {c['result']['path']:c['result'] for c in base['cases']}
        common = [dict(path=r['path'],base_s=old_cases[r['path']]['completion_time_s'],
            changed_s=r['completion_time_s']) for r in raw if r['success'] and old_cases[r['path']]['success']]
        groups.append(dict(mode=mode, stage=folder.name, decisions=args.decisions,
            successes=n, full_time_s=total, cases=cases))
        comparisons.append(dict(mode=mode,base_successes=base['successes'],changed_successes=n,
            base_full_time_s=base['full_time_s'],changed_full_time_s=total,
            changed_minus_base_s=total-base['full_time_s'] if both else None,
            directional_support=both and total < base['full_time_s'], common_success=common))
    result = dict(groups=groups, comparisons=comparisons,
        joint_and_fixed_directional_support=all(c['directional_support'] for c in comparisons),
        primary_budget_endpoint=args.decisions == 60000, independent_seeds_per_configuration=1,
        shape_acceptance_pending=True, automatic_extension=False, sealed_test_opened=False, goal_complete=False)
    write_json(stage/'results.json',result)
    lines = [f'# 信用范围配对比较：{args.decisions}决策', '',
        '| 模式 | base λ=.98成功 | λ=1成功 | base十二条/s | λ=1十二条/s | 差值/s |',
        '|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: '未全部成功' if x is None else f'{x:.3f}'
    for c in comparisons:
        lines.append(f"| {c['mode']} | {c['base_successes']}/12 | {c['changed_successes']}/12 | {fmt(c['base_full_time_s'])} | {fmt(c['changed_full_time_s'])} | {fmt(c['changed_minus_base_s'])} |")
    lines += ['', '主比较为60k；所有节点和较好检查点保留。这里只读新保存评价24条，base复用已完成诊断，无新模型、物理或训练。',
        '', '同一seed917开发比较不证明独立可靠优势。形状、局部速度及全部结局均保留，不以信用指标代替性能；无自动扩预算或状态/奖励组合。']
    (stage/'信用范围比较.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_saved_credit_horizon_comparison',decisions=args.decisions,
        new_saved_trace_analyses=24,model_calls=0,physics_ticks=0,training_decisions=0))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        f'λ=1配对先导共同{args.decisions}节点已分析，只读24条新保存轨迹并复用base。以60k主比较、完整率与形状评价有界后继；保留六组简化负结果和强v7，未打开封存测试，最终方法、独立种子、长路径及20–25页全文仍未完成。')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
