"""Read only the completed 120k development commands, without rollouts."""
import json
import numpy as np
from study_io import BASE, register, receipt, write_json
from summarize_conditioned_training import diagnostics


def main():
    names = ['joint_refinement_training_v6', 'fixed3_refinement_training_v6',
             'fixed4p6_refinement_training_v6']
    phases = ['final_validation']
    sources = [__file__, BASE/'summarize_conditioned_training.py',
               BASE/'s2_line_development_v2/results.json']
    for name in names:
        for phase in phases:
            folder = BASE/name/phase
            sources.extend([folder/'summary.json', BASE/name/'receipt.json', BASE/name/'selection.json'])
            sources.extend(folder/f'{i}.npz' for i in range(12))
    stage = register('refinement_final_analysis_v6', sources,
        dict(saved_traces=36, models=0, physics_ticks=0),
        'Read all completed 120k results and corner geometry; keep failures and stronger fixed results.')
    groups = []
    for name in names:
        for phase in phases:
            folder = BASE/name/phase
            summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            cases = []
            for i, result in enumerate(summary['results']):
                with np.load(folder/f'{i}.npz') as data:
                    ref, rows, actions = data['reference'], data['rows'], data['actions']
                    detail = diagnostics(ref, rows, actions)
                    if not result['success']:
                        segments = np.diff(ref, axis=0)
                        f = np.clip(np.sum((rows[-1,:2]-ref[:-1])*segments, axis=1)/
                                    np.sum(segments*segments, axis=1), 0., 1.)
                        nearest = ref[:-1] + f[:,None]*segments
                        detail['failure_nearest_reference_distance_mm'] = float(
                            np.linalg.norm(nearest-rows[-1,:2],axis=1).min())
                        detail['last_executed_speed_mm_s'] = float(np.linalg.norm(rows[-1,2:4]))
                        detail['last_held_action'] = actions[-1].tolist()
                cases.append(dict(result=result, diagnostics=detail))
            all_ok = all(r['result']['success'] for r in cases)
            lines_ok = all(r['result']['success'] for r in cases[:8])
            groups.append(dict(stage=name, phase=phase, successes=summary['successes'],
                complete_total_s=sum(r['result']['completion_time_s'] for r in cases) if all_ok else None,
                line_total_s=sum(r['result']['completion_time_s'] for r in cases[:8]) if lines_ok else None,
                cases=cases))
    s2 = json.loads((BASE/'s2_line_development_v2/results.json').read_text(encoding='utf-8'))
    s2_total = sum(r['completion_time_s'] for r in s2['results'])
    write_json(stage/'results.json', dict(groups=groups, s2_eight_line_total_s=s2_total,
        final_selection=True, independent_test=False))
    write_json(stage/'selected_checkpoints.json', {name: json.loads((BASE/name/'selection.json').read_text(encoding='utf-8')) for name in names})
    def fmt(x):
        return '不可全量排名' if x is None else f'{x:.3f}'
    lines = ['# 12万步结束时的效率与切角', '',
        '| 方法 | 决策数 | 完整成功 | 十二条总时间/s | 同八条直线路径/s |',
        '|---|---:|---:|---:|---:|']
    for group in groups:
        lines.append(f"| {group['stage']} | {group['phase'].replace('final_validation','120000')} | {group['successes']}/12 | "
                     f"{fmt(group['complete_total_s'])} | {fmt(group['line_total_s'])} |")
    lines += ['', f'同八条常规S2的已完成总时间为{s2_total:.3f} s。该结果直接复用，不重新规划或评价。', '',
        '| 方法及检查点 | 转角/° | 最近尖顶/mm | 局部最低速度/(mm/s) | 全程D中位数/mm |',
        '|---|---:|---:|---:|---:|']
    for group in groups:
        for angle, case in zip([50,95,125,150], group['cases'][:4]):
            detail = case['diagnostics']
            corner = detail['corners'][0]
            lines.append(f"| {group['stage']} {group['phase'].replace('final_validation','120000')} | {angle} | "
                f"{corner['minimum_vertex_distance_mm']:.3f} | {fmt(corner['speed_min_within_10mm'])} | "
                f"{detail['preview_quantiles_mm'][1]:.3f} |")
    lines += ['', '三组原120k预算均已结束；最终选模沿用各组selection.json中完整成功优先的结果，较好的早期检查点继续保留。单角距离对这些完整单角命令计算；连续路径失败时的距离只能描述执行前缀，不能称为完整过弯性能。', '',
        '这次分析只读取已保存的36份final轨迹，没有模型调用、新命令或训练。失败路径的最后动作、速度和到参考的距离见results.json。']
    (stage/'最终检查点分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage, dict(status='completed_saved_120k_analysis', saved_traces=36,
        models=0, physics_ticks=0, goal_complete=False))
    print(json.dumps([dict(stage=g['stage'],phase=g['phase'],successes=g['successes'],
        total=g['complete_total_s'],lines=g['line_total_s']) for g in groups]))


if __name__ == '__main__':
    main()
