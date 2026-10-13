"""Conventional S2 on the eight new line-only development paths, once each."""
import importlib.util
import json
import sys
import time
import traceback
import numpy as np
from study_io import BASE,ROOT,register,write_json,receipt,sha,record_progress
from geometry_spec import audit_commands
from validate_actions import difference_audit

SOURCE=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


if __name__=='__main__':
    prepared=BASE/'mixed_curriculum_preparation_v2'
    stage=register('s2_line_development_v2',[__file__,BASE/'geometry_spec.py',BASE/'validate_actions.py',
        SOURCE/'geometry.py',SOURCE/'scheduler.py',SOURCE/'command.py',prepared/'paths.json',prepared/'receipt.json'],
        dict(new_paths=list(range(8)),geometry_builds=8,schedules=8,complete_command_generations=8,
             command_ticks_per_case=25000,wall_s=900,old_experiments_rerun=0),
        'Reuse the conventional Sencer geometry and compatible S-curve scheduler for eight new line-only paths; arc/circle validation is outside this adapter.')
    started=time.perf_counter()
    results=[]
    try:
        geometry=module('joint_new_s2_geometry',SOURCE/'geometry.py')
        scheduler=module('joint_new_s2_scheduler',SOURCE/'scheduler.py')
        command=module('joint_new_s2_command',SOURCE/'command.py')
        paths=json.loads((prepared/'paths.json').read_text(encoding='utf-8'))['validation']
        # Same fixed sampling margin as the adopted S2; no post-result tuning.
        geometry.CORNER_ERROR_MM=1.2447
        geometry.SOURCE_REGISTRATION_SHA=sha(prepared/'receipt.json')
        for index in range(8):
            if time.perf_counter()-started>900: raise TimeoutError('Finite S2 wall budget')
            ref=np.array(paths[index],float)
            arcs=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))]
            raw=[geometry.Line(ref[i],ref[i+1],arcs[i],arcs[i+1]) for i in range(len(ref)-1)]
            item=dict(id=f'new_line_validation_{index}',closed=False,
                reference_file=str(prepared/'paths.json')+f'#validation/{index}',reference_sha256=sha(prepared/'paths.json'))
            # Only the source provider is replaced. Geometry/scheduling equations are reused.
            geometry._source_geometry=lambda path_id:(item,ref,raw,0.)
            curve=geometry.build_geometry(item['id'])
            schedule=scheduler.schedule(curve.curve_lengths_mm,curve.curve_speed_caps_mm_s,
                curve.straight_lengths_mm,F=100.,A=1998.,J=62437.5)
            assembled=command.assemble(schedule,curve)
            positions=assembled['pos']
            if len(positions)>25000: raise RuntimeError('S2 command budget exceeded')
            geometric,events=audit_commands(ref,positions,endpoint_tolerance=1.25)
            rows=np.zeros((len(positions)-1,13))
            rows[:,:2]=positions[1:]
            rows[:,2:4]=np.diff(positions,axis=0)/.001
            rows[:,4:6]=np.diff(np.vstack([np.zeros(2),rows[:,2:4]]),axis=0)/.001
            rows[:,6:8]=np.diff(np.vstack([np.zeros(2),rows[:,4:6]]),axis=0)/.001
            audit=difference_audit(ref[0],rows)
            success=bool(geometric['geometry_complete'] and audit['kinematic_pass'] and audit['rest_pass'])
            result=dict(path=index,success=success,elapsed_s=float(assembled['t'][-1]),
                completion_time_s=float(assembled['t'][-1]) if success else None,
                geometry=geometric,audit=audit,command_ticks=len(rows),
                corner_error_parameter_mm=1.2447,curves=len(curve.curve_lengths_mm))
            results.append(result)
            np.savez_compressed(stage/f'path_{index}.npz',reference=ref,rows=rows,t=assembled['t'],sigma=assembled['sigma'])
            write_json(stage/f'path_{index}.json',dict(result,planning_metadata=assembled['metadata'],events=events))
            write_json(stage/'results.json',dict(results=results,complete=len(results)==8))
            print(result,flush=True)
        receipt(stage,dict(status='new_line_only_s2_evaluations_completed',cases=8,
            successes=sum(r['success'] for r in results),command_ticks=sum(r['command_ticks'] for r in results),
            old_experiments_rerun=0,wall_s=time.perf_counter()-started,
            comparative_superiority_not_yet_established=True))
        record_progress('joint_s2_line_stage_v2',[stage/'receipt.json'],
            '新混合课程三臂仍按各自登记推进；常规S2在8条新直线拐角开发路径完成独立生成与新任务审核，结果见s2_line_development_v2。该适配器未评价圆弧/纯圆，不复跑旧三条S2。自适应优于固定和S2目标尚未达到，保留所有失败、恢复和费用，英文论文及全PDF验收待真实完整证据。')
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),completed_cases=len(results),wall_s=time.perf_counter()-started))
        raise
