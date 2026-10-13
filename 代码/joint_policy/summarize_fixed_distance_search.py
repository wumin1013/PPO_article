"""Select a single development distance using all completed registered arms."""
import csv
import json
from pathlib import Path
from study_io import BASE, register, receipt, write_json


def main():
    groups = [(0.8, 'fixed0p8_distance_search_v1'), (1.5, 'fixed1p5_distance_search_v1'),
              (3., 'fixed3_refinement_training_v6'), (4.6, 'fixed4p6_refinement_training_v6'),
              (6., 'fixed6_distance_search_v1'), (8., 'fixed8_distance_search_v1'),
              (12., 'fixed12_distance_search_v1')]
    sources = [Path(__file__), BASE/'fixed_distance_search_v1.md']
    for _, name in groups:
        sources += [BASE/name/p for p in ['registration.json', 'receipt.json', 'selection.json', 'validations.json']]
    if not all(p.exists() for p in sources):
        raise RuntimeError('The complete seven-distance search is not finished')
    stage = register('fixed_distance_search_results_v1', sources,
                     dict(summary_files=7, model_calls=0, physics_ticks=0),
                     'Select one global fixed distance from all seven equally budgeted completed candidates and every checkpoint, retaining failures and stronger earlier models.')
    rows, histories = [], []
    for distance, name in groups:
        selection = json.loads((BASE/name/'selection.json').read_text(encoding='utf-8'))
        best = selection['selected']
        row = dict(distance_mm=distance, stage=name, **best)
        rows.append(row)
        histories += [dict(distance_mm=distance, stage=name, **point) for point in selection['candidates']]
    complete = [r for r in rows if r['successes'] == 12]
    if not complete:
        raise ValueError('No single fixed distance completed all development paths')
    selected = min(complete, key=lambda r:r['complete_time_sum_s'])
    write_json(stage/'results.json', dict(candidates=rows, checkpoints=histories,
        selected_global_distance=selected, development_selection=True,
        individual_new_decision_budget=120000, total_candidate_new_decisions=840000,
        inherited_pretraining_cost_not_counted_again=True,
        continuous_distance_optimality=False, sufficient_convergence_established=False,
        sealed_test_opened=False, goal_complete=False))
    with (stage/'distance_learning_curves.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    text = ['# 固定前瞻范围搜索', '',
        '七个距离均完成120,000个新决策；3和4.6直接复用已完成的v6，不重复训练。每组从共同v5固定3初始化并获得相同预算，按全部检查点选择。', '',
        '| 固定D/mm | 所选新增决策 | 完整成功 | 十二条总时间/s |', '|---|---:|---:|---:|']
    for r in rows:
        time = '不可全量排名' if r['complete_time_sum_s'] is None else f"{r['complete_time_sum_s']:.3f}"
        text.append(f"| {r['distance_mm']:g} | {r['decisions']} | {r['successes']}/12 | {time} |")
    text += ['', f"此有界开发搜索选择单一D={selected['distance_mm']:g} mm。选择仅针对已登记的七个距离和当前训练阶段，不能称连续距离上的最优固定方法，也不能把训练预算结束等同于充分收敛。", '',
        '后续给予所选固定方法与联合策略同等优化机会。所有较早较强检查点、失败距离和未通过路径保留；失败经过时间不参与完整时间排名。']
    (stage/'固定前瞻选择.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_all_seven_fixed_distance_selection',
        candidates=7, selected_distance_mm=selected['distance_mm'],
        model_calls=0, physics_ticks=0, new_training_decisions=0))
    print('\n'.join(text))


if __name__ == '__main__':
    main()
