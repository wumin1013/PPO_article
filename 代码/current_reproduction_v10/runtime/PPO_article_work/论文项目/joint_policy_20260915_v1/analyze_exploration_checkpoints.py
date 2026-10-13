"""Analyze a completed checkpoint from every arm of the exploration study."""
import argparse
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json
from summarize_conditioned_training import diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decisions', type=int, choices=[30000, 60000], required=True)
    args = parser.parse_args()
    phase = 'validation_30000' if args.decisions == 30000 else 'final_validation'
    groups = [('Joint independent', 'joint_independent_exploration_learning_v1'),
              ('Joint gSDE8', 'joint_gsde_exploration_learning_v1'),
              ('Fixed3 independent', 'fixed3_independent_exploration_learning_v1'),
              ('Fixed3 gSDE8', 'fixed3_gsde_exploration_learning_v1')]
    sources = [Path(__file__), BASE/'summarize_conditioned_training.py', BASE/'s2_line_development_v2/results.json']
    for _, name in groups:
        folder = BASE/name/phase
        sources += [folder/'summary.json', *[folder/f'{i}.npz' for i in range(12)]]
        if args.decisions == 60000:
            sources += [BASE/name/'receipt.json', BASE/name/'selection.json']
    if not all(p.exists() for p in sources):
        raise RuntimeError('Wait for the completed checkpoint from all four arms')
    stage = register(f'exploration_learning_analysis_{args.decisions}_v1', sources,
                     dict(saved_traces=48, model_calls=0, physics_ticks=0),
                     'Compare all four arms at the same completed checkpoint without rerunning evaluations.')
    rows = []
    for label, name in groups:
        folder = BASE/name/phase
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        cases = []
        for i, result in enumerate(summary['results']):
            with np.load(folder/f'{i}.npz') as data:
                detail = diagnostics(data['reference'], data['rows'], data['actions'])
            cases.append(dict(result=result, diagnostics=detail))
        complete = all(c['result']['success'] for c in cases)
        lines_ok = all(c['result']['success'] for c in cases[:8])
        rows.append(dict(label=label, stage=name, decisions=args.decisions, phase=phase,
            successes=summary['successes'],
            complete_time_sum_s=sum(c['result']['completion_time_s'] for c in cases) if complete else None,
            line_time_sum_s=sum(c['result']['completion_time_s'] for c in cases[:8]) if lines_ok else None,
            cases=cases))
    s2 = json.loads((BASE/'s2_line_development_v2/results.json').read_text(encoding='utf-8'))
    write_json(stage/'results.json', dict(groups=rows,
        reused_s2_line_sum_s=sum(r['completion_time_s'] for r in s2['results']),
        independent_test=False, goal_complete=False))
    fmt = lambda value: '不可全量排名' if value is None else f'{value:.3f}'
    text = [f'# 探索学习{args.decisions}步检查点', '',
        '| 方法 | 完整成功 | 十二条总时间/s | 同八条直线路径/s | 150°最近尖顶/mm | 150°全程D中位数/mm |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        corner = row['cases'][3]['diagnostics']
        text.append(f"| {row['label']} | {row['successes']}/12 | {fmt(row['complete_time_sum_s'])} | {fmt(row['line_time_sum_s'])} | {corner['corners'][0]['minimum_vertex_distance_mm']:.3f} | {corner['preview_quantiles_mm'][1]:.3f} |")
    text += ['',
        '所有组均从同一v6固定3 final初始化，初始十二条总时间21.040 s。较早检查点继续参与选择，不以final退化削弱对照。当前比较为开发数据；即使时间差较小，也不能据单一种子宣称统计稳定优势。角点距离如来自失败路径，只描述前缀。', '',
        '平滑探索用于训练动作采样，评价均使用确定性均值。改善或退化是整组学习结果，不能直接推定探索平滑性是唯一原因。原S2同八条总时间9.493 s，直接复用。']
    (stage/'检查点比较.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_checkpoint_analysis', saved_traces=48,
        model_calls=0, physics_ticks=0, training_decisions=0))
    print('\n'.join(text))


if __name__ == '__main__':
    main()
