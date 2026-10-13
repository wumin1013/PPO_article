"""Describe turning dynamics from saved selected-v6 commands, without rollouts."""
from pathlib import Path
import csv
import json
import numpy as np
from study_io import BASE, register, receipt, write_json


def main():
    groups = [('Joint', 'joint_refinement_training_v6/final_validation'),
              ('Fixed3', 'fixed3_refinement_training_v6/final_validation'),
              ('Fixed4.6', 'fixed4p6_refinement_training_v6/validation_90000')]
    sources = [Path(__file__), BASE/'refinement_final_analysis_v6/selected_checkpoints.json']
    for _, name in groups:
        sources += [BASE/name/'summary.json', *[BASE/name/f'{i}.npz' for i in range(4)]]
    stage = register('selected_turning_dynamics_v6', sources,
                     dict(saved_traces=12, model_calls=0, physics_ticks=0),
                     'Measure signed direction changes and discrete kinematic curvature near all four isolated corners; distinguish broad cuts from slow or reversing transitions.')
    results = []
    for label, name in groups:
        summary = json.loads((BASE/name/'summary.json').read_text(encoding='utf-8'))
        for i, angle in enumerate([50, 95, 125, 150]):
            assert summary['results'][i]['success']
            with np.load(BASE/name/f'{i}.npz') as data:
                ref, rows, actions = data['reference'], data['rows'], data['actions']
            xy, velocity = rows[:, :2], rows[:, 2:4]
            speed = np.linalg.norm(velocity, axis=1)
            distance = np.linalg.norm(xy-ref[1], axis=1)
            nearby = distance <= 10.
            assert not np.any(rows[nearby, 12] > .5), 'Corner overlaps automatic terminal tail'
            cross = velocity[:-1, 0]*velocity[1:, 1]-velocity[:-1, 1]*velocity[1:, 0]
            dot = np.sum(velocity[:-1]*velocity[1:], axis=1)
            dtheta = np.arctan2(cross, dot)
            valid = nearby[:-1] & nearby[1:] & (speed[:-1] >= 5.) & (speed[1:] >= 5.)
            assert valid.any()
            incoming, outgoing = np.diff(ref, axis=0)
            sign = np.sign(incoming[0]*outgoing[1]-incoming[1]*outgoing[0])
            turn = sign*dtheta[valid]
            mean_speed = (speed[:-1]+speed[1:])/2
            curvature = dtheta[valid]/(.001*mean_speed[valid])
            selected_indices = np.flatnonzero(valid)
            peak_local = int(np.argmax(np.abs(curvature)))
            peak_index = int(selected_indices[peak_local]+1)
            idx = np.flatnonzero(nearby)
            held_index = idx//20
            assert held_index.max() < len(actions)
            held = actions[held_index]
            acceleration = np.linalg.norm(rows[nearby, 4:6], axis=1)
            jerk = np.linalg.norm(rows[nearby, 6:8], axis=1)
            result = dict(method=label, turn_deg=angle,
                complete_time_s=summary['results'][i]['completion_time_s'],
                window_time_s=.001*int(nearby.sum()),
                median_speed_mm_s=float(np.median(speed[nearby])),
                minimum_speed_mm_s=float(speed[nearby].min()),
                peak_discrete_curvature_per_mm=float(np.max(np.abs(curvature))),
                equivalent_minimum_radius_mm=float(1/np.max(np.abs(curvature))),
                peak_curvature_vertex_distance_mm=float(distance[peak_index]),
                signed_direction_change_deg=float(np.rad2deg(turn.sum())),
                total_absolute_direction_change_deg=float(np.rad2deg(np.abs(turn).sum())),
                reverse_direction_change_deg=float(np.rad2deg(np.maximum(-turn, 0.).sum())),
                acceleration_near_limit_fraction=float(np.mean(acceleration >= .99*2000)),
                jerk_near_limit_fraction=float(np.mean(jerk >= .99*62500)),
                feed_median_mm_s=float(np.median(held[:, 1])),
                preview_median_mm=float(np.median(held[:, 2])),
                steering_median_deg=float(np.rad2deg(np.median(held[:, 0]))))
            results.append(result)
    write_json(stage/'results.json', dict(results=results,
        definition='Window: executed samples within 10 mm of vertex. Direction differences use consecutive nonterminal velocities with both speeds >= 5 mm/s. Curvature proxy = signed direction difference / mean speed / 1 ms; not exact curvature of the piecewise-linear command interpolant.',
        reverse_turn_definition='Accumulated angular change opposite to the reference turn within the window; no smoothing or zero-crossing threshold.',
        causal_diagnosis=False, physical_rollouts=0))
    with (stage/'turning_dynamics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ['# 所选策略的过弯动态', '',
        '分析十二份已有完整单角命令，窗口固定为距尖顶10 mm以内。累计反向转角只统计相对参考转向的反向变化；它描述入弯、出弯修正，不自动表示几何失败。未平滑、未重放、未调用模型。', '',
        '| 方法 | 转角/° | 近角时间/s | 近角最低速度/(mm/s) | 峰值等效曲率/(1/mm) | 累计反向转角/° |',
        '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['method']} | {r['turn_deg']} | {r['window_time_s']:.3f} | {r['minimum_speed_mm_s']:.2f} | {r['peak_discrete_curvature_per_mm']:.3f} | {r['reverse_direction_change_deg']:.2f} |")
    lines += ['',
        '曲率列采用连续两个速度向量的方向差，除以平均速度和1 ms；只计速度均不低于5 mm/s的样本。它是离散运动的曲率估计，不是分段直线位置命令的严格微分曲率。倒数只用于比较局部转弯尺度，不能作为完整轨迹最小曲率半径证书。', '',
        '同一个10 mm空间窗口的轨迹长度可能不同，局部时间需要与完整时间共同解释。对固定D的速度学习、终端制动和其他角点均没有施加新条件。这里只定位当前行为，不证明改变某个奖励或探索方式必然有效。']
    (stage/'过弯动态诊断.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_turning_dynamics', saved_traces=12,
        model_calls=0, physics_ticks=0, training_decisions=0))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
