"""Partition saved complete long-path time using common spatial regions."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from study_io import BASE, ROOT, register, receipt, write_json


def main():
    cache = ROOT/'论文项目/revision_20260913_preview_efficiency_v2/recursive_kcm_roundoff_continuation_v1/figures_v1/two_step_cache_adapter_v1/cache_v1/mixed_lengths_large/TS_source_data.npz'
    groups = [('Joint', 'joint_long_mixed_development_v6'), ('Fixed3', 'fixed3_long_mixed_development_v6')]
    sources = [Path(__file__), cache, BASE/'s2_frozen_command_new_geometry_v1/receipt.json']
    for _, name in groups:
        sources += [BASE/name/p for p in ['receipt.json', 'evaluation.json', 'trace.npz']]
    stage = register('long_time_regions_v6', sources,
                     dict(saved_trace_files=3, model_calls=0, physics_ticks=0),
                     'Separate sharp-corner-neighbourhood time from the rest of the complete mixed path, using common fixed spatial windows and saved S2 commands.')
    with np.load(cache) as data:
        reference = data['reference_positions_mm']
        s2xy, s2speed, s2time = data['positions_mm'][1:], data['speed_mm_s'][1:], float(data['timestamp_s'][-1])
    d = np.diff(reference, axis=0)
    tangent = d/np.linalg.norm(d, axis=1)[:, None]
    turns = np.rad2deg(np.arccos(np.clip(np.sum(tangent[:-1]*tangent[1:], axis=1), -1., 1.)))
    vertices = reference[np.flatnonzero(turns >= 30.)+1]
    tree = cKDTree(vertices)
    records, table = [], []
    def analyze(label, xy, speed, complete_time, actions=None):
        start_end = np.minimum(np.linalg.norm(xy-reference[0], axis=1), np.linalg.norm(xy-reference[-1], axis=1)) <= 15.
        corner = (tree.query(xy)[0] <= 15.) & ~start_end
        other = ~start_end & ~corner
        masks = [('start_end_15mm', start_end), ('sharp_corner_15mm', corner), ('remaining_path', other)]
        parts = []
        assert abs(len(xy)*.001-complete_time) < 1e-8
        for region, mask in masks:
            part = dict(method=label, region=region, time_s=.001*int(mask.sum()),
                        median_speed_mm_s=float(np.median(speed[mask])) if mask.any() else None,
                        speed_below_50_fraction=float(np.mean(speed[mask]<50.)) if mask.any() else None)
            parts.append(part)
            table.append(part)
        result = dict(method=label, complete_time_s=complete_time,
            mean_speed_mm_s=float(speed.mean()), maximum_speed_mm_s=float(speed.max()),
            speed_quantiles_mm_s=np.quantile(speed, [.1, .5, .9]).tolist(), parts=parts)
        if actions is not None:
            result['feed_intent_quantiles_mm_s'] = np.quantile(actions[:, 1], [.1, .5, .9]).tolist()
            result['preview_quantiles_mm'] = np.quantile(actions[:, 2], [.1, .5, .9]).tolist()
        records.append(result)
    for label, name in groups:
        folder = BASE/name
        evaluation = json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        assert evaluation['success']
        with np.load(folder/'trace.npz') as data:
            assert np.allclose(data['reference'], reference, rtol=0., atol=1e-10)
            rows, actions = data['rows'], data['actions']
        analyze(label, rows[:, :2], np.linalg.norm(rows[:, 2:4], axis=1), evaluation['completion_time_s'], actions)
    analyze('S2', s2xy, s2speed, s2time)
    s2parts = {r['region']: r for r in records[-1]['parts']}
    for r in records[:-1]:
        r['time_excess_over_s2_s'] = r['complete_time_s']-s2time
        r['region_excess_over_s2_s'] = {p['region']: p['time_s']-s2parts[p['region']]['time_s'] for p in r['parts']}
    write_json(stage/'results.json', dict(reference_sharp_vertex_count=len(vertices), results=records,
        window_radius_mm=15., turn_threshold_deg=30.,
        definition='Assign each 1 ms interval to its command endpoint: start/end radius 15 mm first; union of >=30 degree corner discs second; remaining path third. Remaining path includes curves and connections and is not synonymous with straight cruise.',
        causal_attribution=False, independent_test=False))
    with (stage/'time_regions.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ['# 长混合路径的时间分布', '',
        '三组完整时间均包含制动停稳。空间分区完全相同：起终点15 mm邻域优先，其次为不小于30°的尖角15 mm邻域并集，其余归入剩余路径。每个1 ms区间按命令末点分区，三个分区的时间相加等于完整时间。剩余路径包含圆弧和连接段，不能简称纯直线。', '',
        '| 方法 | 分区 | 时间/s | 速度中位数/(mm/s) |', '|---|---|---:|---:|']
    for row in table:
        lines.append(f"| {row['method']} | {row['region']} | {row['time_s']:.3f} | {row['median_speed_mm_s']:.2f} |")
    lines += ['',
        '这些空间分区用于描述已完成轨迹的时间差，不是更好使用允差带、学习机制或某个代码部件的因果归因。没有重新运行S2或两个策略，也没有修改参考路径与审核条件。']
    (stage/'长路径时间分布.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_time_partition', saved_trace_files=3, model_calls=0, physics_ticks=0))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
