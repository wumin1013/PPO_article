"""Analyze only two new long traces and reuse the completed S2 partition."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from study_io import BASE, register, receipt, write_json, record_progress, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    old_dir = BASE / 'long_time_regions_v6'
    old = read(old_dir / 'results.json')
    assert old['window_radius_mm'] == 15. and old['turn_threshold_deg'] == 30.
    s2 = next(r for r in old['results'] if r['method'] == 'S2')
    names = [('Joint', 'v7_joint_long_mixed_v1'), ('Fixed3', 'v7_fixed_long_mixed_v1')]
    sources = [Path(__file__), BASE / 'v7_long_development_plan_v1.md',
               BASE / 'analyze_long_time_regions_v6.py', old_dir / 'receipt.json', old_dir / 'results.json']
    for _, name in names:
        sources += [BASE / name / f for f in ('receipt.json', 'evaluation.json', 'trace.npz', 'registration.json')]
    stage = register('v7_long_time_regions_v1', sources,
        dict(saved_trace_files=2, reused_s2_summaries=1, model_calls=0, physics_ticks=0, training_decisions=0),
        'Reuse the exact v6 spatial partition on new v7 selected-policy long traces; do not reload or reanalyze old S2 or v6 commands.')
    records = []
    reference_bytes = None
    cost = dict(actor_calls=0, physics_ticks=0, model_loads=0, training_decisions=0)
    for label, name in names:
        folder = BASE / name
        ev, rec = read(folder / 'evaluation.json'), read(folder / 'receipt.json')
        assert ev['success'] and ev['geometry']['gates_passed'] == ev['geometry']['gate_count'] == 1373
        assert rec['trace_sha256'] == sha(folder / 'trace.npz')
        with np.load(folder / 'trace.npz') as data:
            ref, rows, actions = data['reference'], data['rows'], data['actions']
        if reference_bytes is None:
            reference_bytes = ref.tobytes()
        assert ref.tobytes() == reference_bytes
        d = np.diff(ref, axis=0)
        tangent = d / np.linalg.norm(d, axis=1)[:, None]
        turns = np.rad2deg(np.arccos(np.clip(np.sum(tangent[:-1]*tangent[1:], axis=1), -1., 1.)))
        vertices = ref[np.flatnonzero(turns >= 30.)+1]
        assert len(vertices) == old['reference_sharp_vertex_count']
        xy, speed = rows[:, :2], np.linalg.norm(rows[:, 2:4], axis=1)
        start_end = np.minimum(np.linalg.norm(xy-ref[0], axis=1), np.linalg.norm(xy-ref[-1], axis=1)) <= 15.
        corner = (cKDTree(vertices).query(xy)[0] <= 15.) & ~start_end
        masks = [('start_end_15mm', start_end), ('sharp_corner_15mm', corner), ('remaining_path', ~start_end & ~corner)]
        parts = [dict(region=region, time_s=.001*int(mask.sum()),
            median_speed_mm_s=float(np.median(speed[mask])),
            speed_below_50_fraction=float(np.mean(speed[mask] < 50.))) for region, mask in masks]
        assert abs(sum(p['time_s'] for p in parts)-ev['completion_time_s']) < 1e-8
        previous = next(r for r in old['results'] if r['method'] == label)
        s2parts = {p['region']: p for p in s2['parts']}
        records.append(dict(method=label, stage=name, complete_time_s=ev['completion_time_s'],
            previous_v6_time_s=previous['complete_time_s'], change_from_v6_s=ev['completion_time_s']-previous['complete_time_s'],
            excess_over_s2_s=ev['completion_time_s']-s2['complete_time_s'], parts=parts,
            region_excess_over_s2_s={p['region']:p['time_s']-s2parts[p['region']]['time_s'] for p in parts},
            speed_quantiles_mm_s=np.quantile(speed,[.1,.5,.9]).tolist(),
            feed_intent_quantiles_mm_s=np.quantile(actions[:,1],[.1,.5,.9]).tolist(),
            preview_quantiles_mm=np.quantile(actions[:,2],[.1,.5,.9]).tolist(),
            maximum_speed_mm_s=float(speed.max()), audit=ev['audit'], geometry=ev['geometry']))
        for key in cost:
            cost[key] += ev['costs'].get(key,0)
    result = dict(results=records, reused_s2=s2, new_evaluation_costs=cost,
        original_s2_recomputed=False, definition=old['definition'], causal_attribution=False,
        selected_by_short_development_before_long_evaluation=True, final_test=False, goal_complete=False)
    write_json(stage / 'results.json', result)
    lines = ['# 所选v7长mixed完整结果与分区', '',
        '| 方法 | 完整时间/s | 急角15 mm邻域/s | 其余路径/s | 比S2慢/s |',
        '|---|---:|---:|---:|---:|']
    for r in records + [dict(s2, excess_over_s2_s=0.)]:
        p = {x['region']:x for x in r['parts']}
        lines.append(f"| {r['method']} | {r['complete_time_s']:.3f} | {p['sharp_corner_15mm']['time_s']:.3f} | {p['remaining_path']['time_s']:.3f} | {r['excess_over_s2_s']:.3f} |")
    lines += ['', old['definition'], '',
        '双方完整通过1373道门与V/A/J、停稳审核；各取相同探索候选池内短开发成绩最强者，采样方式不同，不是同采样方式单因素比较。',
        '只读两份新trace，S2及旧v6分区直接复用。其余路径含圆弧和连接，不能当作纯直线；分区差不是因果归因。全部结果仍为公开开发证据。']
    (stage / '长路径分区与后续.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage,dict(status='completed_new_v7_long_partition',saved_trace_files=2,
        model_calls=0,physics_ticks=0,training_decisions=0,goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        'v7已选最强联合/固定的两次公开长mixed评价均完成并分区，376.113/347.717秒，S2复用323.367秒，均1373门完整；新分析仅读两份新trace，不重跑。转向/进给与探索覆盖仍需机制证据；最新长结果尚需接入已验收17页中文v9、20页英文v12。最终研究目标、Fig.4–9、独立种子/合理选距/三类最终路径/消融/封存测试未完成。')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
