"""Separate requested and executed speed on the two completed long traces."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from study_io import BASE, register, receipt, write_json, record_progress, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    names = ('v7_joint_long_mixed_v1', 'v7_fixed_long_mixed_v1')
    prior = BASE/'v7_long_time_regions_v1'
    sources = [Path(__file__), BASE/'joint_runtime.py', BASE/'analyze_v7_long_regions.py',
        prior/'results.json', prior/'receipt.json']
    sources += [BASE/name/file for name in names for file in ('trace.npz', 'evaluation.json', 'receipt.json')]
    stage = register('v7_cruise_request_diagnosis_v1', sources,
        dict(saved_trace_files=2, model_calls=0, physics_ticks=0, training_decisions=0),
        'Diagnose whether the recorded long-path speed deficit primarily appears in requested feed or in subsequent execution. Reuse the existing spatial region definition; do not alter actions or rerun motion.')
    records = []
    for name in names:
        folder = BASE/name
        assert sha(folder/'trace.npz') == read(folder/'receipt.json')['trace_sha256']
        with np.load(folder/'trace.npz') as data:
            ref, rows = data['reference'], data['rows']
        tangent = np.diff(ref, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        turns = np.rad2deg(np.arccos(np.clip(np.sum(tangent[:-1]*tangent[1:], axis=1), -1., 1.)))
        vertices = ref[np.flatnonzero(turns >= 30.)+1]
        xy = rows[:, :2]
        region = np.minimum(np.linalg.norm(xy-ref[0], axis=1), np.linalg.norm(xy-ref[-1], axis=1)) > 15.
        region &= cKDTree(vertices).query(xy)[0] > 15.
        previous = next(r for r in read(prior/'results.json')['results'] if r['stage'] == name)
        expected = next(p['time_s'] for p in previous['parts'] if p['region'] == 'remaining_path')
        assert abs(region.sum()*.001-expected) < 1e-8
        assert not rows[region, 12].any(), 'Terminal motion must not enter this comparison'
        speed = np.linalg.norm(rows[region, 2:4], axis=1)
        requested = np.linalg.norm(rows[region, 8:10], axis=1)
        gap = requested-speed
        acceleration = np.linalg.norm(rows[region, 4:6], axis=1)
        jerk = np.linalg.norm(rows[region, 6:8], axis=1)
        records.append(dict(stage=name, ticks=int(region.sum()), region_time_s=expected,
            requested_speed_quantiles_mm_s=np.quantile(requested, [.1, .5, .9]).tolist(),
            executed_speed_quantiles_mm_s=np.quantile(speed, [.1, .5, .9]).tolist(),
            signed_requested_minus_executed_quantiles_mm_s=np.quantile(gap, [.01, .1, .5, .9, .99]).tolist(),
            absolute_speed_gap_median_mm_s=float(np.median(np.abs(gap))),
            fraction_absolute_gap_at_most_half_mm_s=float(np.mean(np.abs(gap) <= .5)),
            fraction_requested_below_95_mm_s=float(np.mean(requested < 95.)),
            fraction_executed_at_least_99_mm_s=float(np.mean(speed >= 99.)),
            fraction_acceleration_at_least_99pct=float(np.mean(acceleration >= 1980.)),
            fraction_jerk_at_least_99pct=float(np.mean(jerk >= 61875.)),
            viability_guard_fraction=float(np.mean(rows[region, 11] > 0.)),
            full_success=read(folder/'evaluation.json')['success']))
    result = dict(records=records, model_calls=0, physics_ticks=0, training_decisions=0,
        thresholds=dict(near_limit_fraction=.99, requested_feed_threshold_mm_s=95., absolute_gap_threshold_mm_s=.5),
        scope='Descriptive saved-command comparison in the original remaining-path region, including arcs and connections. Close scalar speeds do not imply equal velocity directions. Low recorded constraint engagement does not establish the feasibility of a counterfactual faster action, or that the projector had no effect on learning.',
        goal_complete=False)
    write_json(stage/'results.json', result)
    lines = ['# 长路径巡航：请求与执行的速度差', '',
        '| 方法 | 请求中位数/(mm/s) | 执行中位数/(mm/s) | 绝对速度差≤0.5比例 | 请求低于95比例 |',
        '|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['stage']} | {r['requested_speed_quantiles_mm_s'][1]:.3f} | {r['executed_speed_quantiles_mm_s'][1]:.3f} | {r['fraction_absolute_gap_at_most_half_mm_s']:.4%} | {r['fraction_requested_below_95_mm_s']:.4%} |")
    lines += ['', '复用原其余路径空间定义，包含圆弧与连接。新增结果只回答已执行轨迹的请求/执行关系；未干预进给、未重跑物理，不能证明所有点均允许100 mm/s，也不能推断KCM对学习没有影响。阈值仅用于描述，不加入控制器。']
    (stage/'巡航请求与执行.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_cruise_request_analysis', saved_trace_files=2,
        model_calls=0, physics_ticks=0, training_decisions=0, goal_complete=False))
    record_progress(stage.name, [stage/'receipt.json', stage/'results.json'],
        '已只读两份所选v7长mixed保存命令，核对其余区域的请求/执行速度差及约束接触，未重跑模型或物理。宽探索60k已完成，终点20.932秒仍选回20.598秒初值；结合新巡航诊断更新中英文报告和全PDF验收，最终研究目标未完成。')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
