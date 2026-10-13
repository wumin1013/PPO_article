"""Analyze each new four-arm checkpoint once, entirely from saved evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json, sha
from summarize_conditioned_training import diagnostics

GROUPS = [('Joint independent', 'joint_independent_refinement_v7'),
          ('Joint gSDE8', 'joint_gsde_refinement_v7'),
          ('Fixed3 independent', 'fixed3_independent_refinement_v7'),
          ('Fixed3 gSDE8', 'fixed3_gsde_refinement_v7')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decisions', type=int, choices=[*range(60000, 360001, 60000),359999], required=True)
    args = parser.parse_args()
    tag = 'final_validation' if args.decisions in (359999,360000) else f'validation_{args.decisions}'
    sources = [Path(__file__), BASE/'summarize_conditioned_training.py', BASE/'learning_design_v7.md',
               BASE/'s2_line_development_v2/results.json',
               BASE/'exploration_learning_analysis_60000_v1/results.json']
    for _, name in GROUPS:
        folder = BASE/name/tag
        sources += [folder/'summary.json', *[folder/f'{i}.npz' for i in range(12)]]
        if tag == 'final_validation':
            source = BASE/name/'validations.json'
            validations = json.loads(source.read_text(encoding='utf-8'))['validations']
            final = [entry for entry in validations if entry['tag']==tag]
            if len(final)!=1 or final[0]['decisions']!=args.decisions:
                raise ValueError(f'Actual final decision metadata differs for {name}')
            sources += [source, BASE/name/'receipt.json', BASE/name/'selection.json']
    if not all(p.exists() for p in sources):
        raise RuntimeError('The same checkpoint must be complete for all four arms before analysis')
    stage = register(f'v7_checkpoint_{args.decisions}_analysis_v1', sources,
        dict(saved_trace_files=48, models=0, physics_ticks=0, training_decisions=0),
        'Compare four completed paired development evaluations at a common v7 checkpoint; keep stronger earlier results.')
    records = []
    for label, name in GROUPS:
        folder = BASE/name/tag
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        cases = []
        for r in summary['results']:
            file = folder/f"{r['path']}.npz"
            with np.load(file) as data:
                detail = diagnostics(data['reference'], data['rows'], data['actions'])
            cases.append(dict(result=r, diagnostics=detail, trace_sha256=sha(file)))
        complete = all(r['result']['success'] for r in cases)
        lines_complete = all(r['result']['success'] for r in cases[:8])
        records.append(dict(label=label, stage=name, tag=tag, decisions=args.decisions,
            successes=summary['successes'],
            full_time_s=sum(r['result']['completion_time_s'] for r in cases) if complete else None,
            line_time_s=sum(r['result']['completion_time_s'] for r in cases[:8]) if lines_complete else None,
            cases=cases))
    s2 = json.loads((BASE/'s2_line_development_v2/results.json').read_text(encoding='utf-8'))
    reference = sum(r['completion_time_s'] for r in s2['results'])
    write_json(stage/'results.json', dict(groups=records, s2_eight_path_time_s=reference,
        previous_best_joint_twelve_s=20.512, previous_best_fixed_twelve_s=20.598,
        common_initial_twelve_s=20.598, independent_test=False, goal_complete=False))
    fmt = lambda x: '不可全量排名' if x is None else f'{x:.3f}'
    lines = [f'# v7共同{args.decisions}决策评价', '',
        '| 方法 | 完整成功 | 十二条时间/s | 同八条时间/s | 150°最近尖顶/mm | 150°全程D中位数/mm |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        corner = r['cases'][3]['diagnostics']
        lines.append(f"| {r['label']} | {r['successes']}/12 | {fmt(r['full_time_s'])} | {fmt(r['line_time_s'])} | {corner['corners'][0]['minimum_vertex_distance_mm']:.3f} | {corner['preview_quantiles_mm'][1]:.3f} |")
    lines += ['', f'常规S2同八条完整时间{reference:.3f} s；共同初始化十二条20.598 s，旧较强联合20.512 s和固定20.598 s继续保留。所有新评价均来自保存轨迹，未重跑。', '',
        '当前为共同初始化、同一种子下的开发检查点，不是独立完整训练种子。失败前缀的角点距离仅描述前缀，不用于完整时间排名。中间节点不替代全部检查点的最终选择，不能以固定方法退化制造联合优势。其余已登记训练按原预算继续；30条封存测试尚未打开。']
    (stage/'检查点比较.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_v7_checkpoint_analysis', saved_trace_files=48,
        model_calls=0, physics_ticks=0, training_decisions=0, decisions=args.decisions))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
