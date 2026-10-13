"""New finite branch-overlap inventory and offset-domain comparison."""
import math
import sys
import traceback
import numpy as np
from shapely.geometry import LineString
from study_io import BASE, ROOT, register, write_json, receipt, record_progress
from geometry_spec import Corridor


def nonadjacent_overlaps(corridor):
    rows = []
    count = len(corridor.starts)
    segments = [LineString([a,b]) for a,b in zip(corridor.starts,corridor.ends)]
    for i in range(count):
        low = corridor.box_min[i] - corridor.epsilon
        high = corridor.box_max[i] + corridor.epsilon
        possible = np.flatnonzero(np.all(corridor.box_max >= low, axis=1)
                                 & np.all(corridor.box_min <= high, axis=1))
        for j in possible:
            if j <= i+1 or (corridor.closed and i == 0 and j == count-1):
                continue
            distance = segments[i].distance(segments[j])
            if distance <= 2*corridor.epsilon+1e-9:
                rows.append(dict(segments=[i,int(j)], distance_mm=distance,
                    intervening_arc_mm=float(corridor.arcs[j]-corridor.arcs[i+1])))
    return rows


if __name__ == '__main__':
    references = sorted((ROOT/'论文项目/revision_20260913_preview_efficiency_v2/geometry_design_v1/references').glob('*.csv'))
    stage = register('geometry_semantics_v1', [__file__, BASE/'study_io.py', BASE/'geometry_spec.py',
        ROOT/'PPO_project/src/utils/geometry.py', *references],
        dict(reference_inventories=3, synthetic_domain_comparisons=1, physics_ticks=0, training_updates=0),
        'Inspect nonadjacent corridor overlaps and analytically distinguish Euclidean and miter domains.')
    try:
        rows = []
        for source in references:
            ref = np.loadtxt(source, delimiter=',', skiprows=1)
            corridor = Corridor(ref, closed=bool(np.linalg.norm(ref[0]-ref[-1]) < 1e-8))
            overlaps = nonadjacent_overlaps(corridor)
            rows.append(dict(path=source.stem, nonadjacent_overlaps=overlaps,
                             count=len(overlaps)))
        sys.path.insert(0, str(ROOT/'PPO_project'))
        from src.utils.geometry import generate_offset_paths
        angle = math.radians(130)
        ref = np.array([[-20.,0.],[0.,0.],[20*math.cos(angle),20*math.sin(angle)]])
        corridor = Corridor(ref)
        pl, pr = generate_offset_paths(ref, 1.25, closed=False)
        comparison = dict(turn_deg=130., miter_vertex_radius_mm=1.25/math.cos(angle/2),
            left_inner_vertex=np.asarray(pl[1]).tolist(), right_outer_vertex=np.asarray(pr[1]).tolist(),
            inner_distance_to_reference_mm=corridor.point_distance(pl[1]),
            outer_distance_to_reference_mm=corridor.point_distance(pr[1]),
            domains_equal=False,
            selected_domain='Finite reference polyline Euclidean 1.25 mm neighborhood; all command segments audited.',
            display='New figures must depict the same Euclidean domain; historical Pl/Pr figures retained as historical.')
        assert abs(comparison['inner_distance_to_reference_mm']-1.25)<1e-9
        assert comparison['outer_distance_to_reference_mm']>1.25
        write_json(stage/'results.json', dict(reference_inventory=rows, domain_comparison=comparison))
        receipt(stage, dict(status='completed_static_inventory', result='results.json',
            reference_overlap_counts={row['path']:row['count'] for row in rows},
            environment_steps=0, training_updates=0, performance_certificate=False))
        record_progress('joint_geometry_stage_v1', [BASE/'geometry_validation_v1/receipt.json', stage/'receipt.json'],
            '联合策略新几何25项合成检查通过；完成三条参考的非相邻条带盘点和Pl/Pr差异核查，详见joint_policy_20260915_v1。下一步核对盘点结果并接通20ms转向/速度/D与1ms原R2执行，有限权限检查后再成本标定和训练。新联合模型尚无性能结果；保留最新训练授权、原稿英文风格、旧数据和失败，全PDF验收前目标保持active。')
        print({'overlap_counts':{row['path']:row['count'] for row in rows}, 'domain_comparison':comparison}, flush=True)
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc()))
        raise
