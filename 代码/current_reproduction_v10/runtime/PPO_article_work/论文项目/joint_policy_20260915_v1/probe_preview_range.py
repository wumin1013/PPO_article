"""Finite controller-reachability probes, not trained baseline comparisons."""
import math
import time
import traceback
import numpy as np
from study_io import BASE, ROOT, register, write_json, receipt, record_progress
from joint_runtime_v2 import JointRuntimeV2
from validate_actions import difference_audit
from analyze_pilot import vertex_distance


if __name__=='__main__':
    stage = register('preview_range_probe_v2',[__file__,BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',
        BASE/'geometry_spec.py',BASE/'study_io.py',BASE/'validate_actions.py',BASE/'analyze_pilot.py'],
        dict(angles_deg=[90,130,155],speeds_mm_s=[35,60,85],fixed_distances_mm=[3,4.6,8,12],
             runs=36,max_ticks_per_run=5000,wall_s=600,training_updates=0),
        'Test whether the old 4.6 mm D range prevents useful earlier guidance; all failures retained; no statistical or learned superiority claims.')
    started = time.perf_counter()
    results = []
    try:
        for angle in [90.,130.,155.]:
            theta = math.radians(angle)
            ref = np.array([[-25.,0.],[0.,0.],[30*math.cos(theta),30*math.sin(theta)]])
            for speed in [35.,60.,85.]:
                for distance in [3.,4.6,8.,12.]:
                    env = JointRuntimeV2(ref,fixed_d=distance,max_time=5.)
                    action = np.array([0.,np.arctanh(speed/50.-1.),0.])
                    chunks = []
                    while not env.done:
                        if time.perf_counter()-started>600: raise TimeoutError('Probe wall budget')
                        _,_,_,info = env.step(action)
                        chunks.append(info['rows'])
                    rows = np.vstack(chunks)
                    audit = difference_audit(ref[0],rows)
                    success = env.success and audit['kinematic_pass'] and audit['rest_pass']
                    record = dict(angle_deg=angle,speed=speed,D=distance,success=bool(success),reason=env.reason,
                        elapsed_s=env.ticks*.001,completion_time_s=env.ticks*.001 if success else None,
                        minimum_vertex_distance_mm=vertex_distance(np.vstack([ref[0],rows[:,:2]])),
                        audit=audit,ticks=env.ticks)
                    results.append(record)
                    np.savez_compressed(stage/(f'angle{int(angle)}_speed{int(speed)}_D{distance:g}.npz'),reference=ref,rows=rows)
                    write_json(stage/'results.json',dict(runs=results,complete=len(results)==36))
                    print(record,flush=True)
        receipt(stage,dict(status='finite_preview_range_probe_completed',runs=36,
            successes=sum(row['success'] for row in results),physics_ticks=sum(row['ticks'] for row in results),
            training_updates=0,model_calls=0,wall_s=time.perf_counter()-started,
            trained_comparator=False,performance_superiority_established=False))
        record_progress('joint_range_probe_v2',[stage/'receipt.json',BASE/'pilot_analysis_v1/receipt.json'],
            '三组各10000步先导均4/4成功；固定3mm总4.904s快于联合5.216s，仅D5.548s，尚未达到自适应优于固定的目标。完成共同扩展D范围的有限权限/可行性探测并保留全部失败，按preview_range_probe_v2结果确定下一有界课程。修正远端观测归一化，保持V/A/J、允差和完整停稳；不以弱固定制造优势。目标active，论文沿用原稿英文风格，全PDF/20–25页/逐图报告未完成。')
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),completed_runs=len(results)))
        raise
