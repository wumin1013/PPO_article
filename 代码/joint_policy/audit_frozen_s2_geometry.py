"""Audit frozen S2 position commands under the new common geometric task."""
import json
import time
import traceback
import numpy as np
from study_io import BASE,ROOT,register,receipt,write_json,sha,record_progress
from geometry_spec import Corridor,GateTracker

CASES=['mixed_lengths_large','dense_corners_large','closed_line_arc_large']
SOURCE=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1/sampling_reserve_v1/cases'


if __name__=='__main__':
    sources=[__file__,BASE/'geometry_spec.py',BASE/'task_contract.md',BASE/'study_io.py',
             BASE/'geometry_semantics_v1/receipt.json']
    records=[]
    for case in CASES:
        folder=SOURCE/case/'full'
        result=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
        original=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
        command_receipt=json.loads((folder/'commands_receipt.json').read_text(encoding='utf-8'))
        if not result['full_success'] or not original['full_success']:
            raise ValueError('Frozen successful S2 evidence required')
        commands=folder/'commands.npz'
        relative=commands.relative_to(ROOT).as_posix()
        expected=command_receipt['files'][relative]
        if original['outputs'][relative]!=expected:
            raise ValueError('Command/audit source binding differs')
        if sha(folder/'evaluation.json')!=original['outputs'][(folder/'evaluation.json').relative_to(ROOT).as_posix()]:
            raise ValueError('Original evaluation changed')
        reference=ROOT/result['reference']['file']
        sources += [folder/'evaluation.json',folder/'receipt.json',folder/'commands_receipt.json',reference]
        records.append(dict(case=case,folder=folder,commands=commands,reference=reference,
            reference_sha256=result['reference']['sha256'],command_sha256=expected,result=result))
    stage=register('s2_frozen_command_new_geometry_v1',sources,
        dict(cases=CASES,command_files_loaded=3,command_segments_per_case_max=500000,
             wall_s=5400,models=0,new_command_generations=0,training_decisions=0),
        'New Euclidean corridor/directed-section audit of already saved S2 commands; reuse the byte-bound original V/A/J and rest audit. No re-planning or re-training.')
    started=time.perf_counter()
    results=[]
    loaded=0
    checked=0
    try:
        for item in records:
            case=item['case']
            if sha(item['commands'])!=item['command_sha256'] or sha(item['reference'])!=item['reference_sha256']:
                raise ValueError('Frozen trajectory/reference bytes changed')
            ref=np.loadtxt(item['reference'],delimiter=',',skiprows=1)
            with np.load(item['commands']) as data:
                positions=data['pos']
                times=data['t']
            loaded+=1
            if len(positions)-1>500000 or not np.allclose(np.diff(times),.001,rtol=0,atol=1e-12):
                raise ValueError('Command budget or public timestep differs')
            corridor=Corridor(ref,closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-8))
            tracker=GateTracker(corridor,positions[0])
            for index in range(1,len(positions)):
                tracker.update(positions[index])
                checked+=1
                if index%5000==0:
                    write_json(stage/'progress.json',dict(case=case,command_index=index,
                        command_count=len(positions)-1,total_segments_checked=checked,
                        wall_s=time.perf_counter()-started,geometry=tracker.snapshot(1.25)))
                    # Stores the exact audit frontier without copying motion commands.
                    state={key:value for key,value in vars(tracker).items() if key!='corridor'}
                    state['position']=state['position'].tolist()
                    write_json(stage/'audit_frontier.json',dict(case=case,command_index=index,
                        command_sha256=item['command_sha256'],state=state))
                    if time.perf_counter()-started>5400:
                        raise TimeoutError('Registered audit wall limit')
            geometry=tracker.snapshot(1.25)
            success=bool(geometry['geometry_complete'] and item['result']['full_success'])
            result=dict(case=case,success=success,geometry=geometry,
                complete_time_s=float(times[-1]) if success else None,
                saved_elapsed_s=float(times[-1]),command_sha256=item['command_sha256'],
                reference_sha256=item['reference_sha256'],command_segments_checked=len(positions)-1,
                original_dynamic_and_rest_audit_reused=True,
                original_evaluation=str((item['folder']/'evaluation.json').relative_to(ROOT)))
            if abs(float(times[-1])-item['result']['full_time_s'])>1e-9:
                raise ValueError('Original complete time differs from saved time vector')
            results.append(result)
            write_json(stage/(case+'.json'),dict(result,events=tracker.events))
            write_json(stage/'results.json',dict(results=results,complete=len(results)==3))
            print(result,flush=True)
        receipt(stage,dict(status='frozen_s2_new_geometry_audit_completed',cases=3,
            successes=sum(r['success'] for r in results),existing_commands_loaded=loaded,
            command_segments_checked=checked,models=0,new_command_generations=0,
            training_decisions=0,wall_s=time.perf_counter()-started,goal_complete=False))
        record_progress('s2_frozen_command_new_geometry_v1',[stage/'receipt.json'],
            '原三条S2命令完成新欧氏允差与有向门序审核，结果见s2_frozen_command_new_geometry_v1；只复用原命令和其字节绑定的V/A/J/停稳收据，没有重新规划或生成运动。v4三组训练仍按实际进程/收据推进，后续需新联合策略长路径评价、强固定D、封存测试、多种子和最终论文全PDF验收；目标active。')
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),
            completed_cases=len(results),loaded_command_files=loaded,
            command_segments_checked=checked,wall_s=time.perf_counter()-started,automatic_retry=False))
        raise
