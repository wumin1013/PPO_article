"""Audit the preserved original NNC commands under the current geometry."""
import io
import hashlib
import json
import time
import traceback
import numpy as np
from study_io import BASE,ROOT,register,receipt,write_json,sha
from geometry_spec import Corridor,GateTracker


def main():
    root=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/recursive_kcm_roundoff_continuation_v1/figures_v1/case_comparison_v1/figures_v1'
    cases=['mixed_lengths_large','dense_corners_large','closed_line_arc_large']
    bindings=[]
    sources=[__file__,BASE/'geometry_spec.py',BASE/'task_contract.md']
    for case in cases:
        file=root/case/'source_binding.json'
        binding=json.loads(file.read_text(encoding='utf-8'))
        assert binding['NNC']['observation_dimension']==14
        assert binding['NNC']['original_audit']['active_limits']==dict(MAX_VEL=100.,MAX_ACC=2000.,MAX_JERK=62500.)
        assert sha(ROOT/binding['reference']['file'])==binding['reference']['sha256']
        sources += [file,ROOT/binding['reference']['file']]
        bindings.append(binding)
    stage=register('nnc_saved_command_new_geometry_v1',sources,
        dict(derived_archives=3,command_segments_max=3600000,wall_s=1800,models=0,new_commands=0,training_decisions=0),
        'Reuse original 14D NNC commands and dynamic audits; independently apply the current Euclidean corridor and ordered sections for preserved figure roles.')
    results=[];checked=0;loaded=0;current=None
    started=time.perf_counter()
    try:
        for binding in bindings:
            current=binding['case'];nnc=binding['NNC']
            archive=(ROOT/nnc['cache_file']).read_bytes()
            assert hashlib.sha256(archive).hexdigest()==nnc['cache_sha256']
            with np.load(io.BytesIO(archive)) as data:
                positions=data['NNC__position'];times=data['NNC__timestamp']
            del archive
            loaded+=1
            assert len(positions)==nnc['original_audit']['sample_count']
            assert np.allclose(np.diff(times),.001,rtol=0,atol=1e-12)
            ref=np.loadtxt(ROOT/binding['reference']['file'],delimiter=',',skiprows=1)
            corridor=Corridor(ref,closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-8))
            tracker=GateTracker(corridor,positions[0])
            for index in range(1,len(positions)):
                tracker.update(positions[index]);checked+=1
                if index%10000==0:
                    state={k:v for k,v in vars(tracker).items() if k!='corridor'}
                    state['position']=state['position'].tolist()
                    write_json(stage/'audit_frontier.json',dict(case=current,command_index=index,state=state,cache_sha256=nnc['cache_sha256']))
                    write_json(stage/'progress.json',dict(case=current,command_index=index,total_checked=checked,wall_s=time.perf_counter()-started))
                    if checked>3600000 or time.perf_counter()-started>1800:
                        raise TimeoutError('Registered saved-command audit budget exhausted')
            geometry=tracker.snapshot(1.25)
            original=nnc['original_audit']
            dynamics=all(c['satisfied'] for c in original['constraints'].values())
            rest=bool(original['full_rest_satisfied'])
            success=bool(geometry['geometry_complete'] and dynamics and rest)
            result=dict(case=current,observation_dimension=14,reference=binding['reference'],
                cache_file=nnc['cache_file'],cache_sha256=nnc['cache_sha256'],
                original_command_sha256=nnc['original_command_sha256'],geometry=geometry,
                original_dynamic_and_rest_audit_reused=True,dynamic_pass=dynamics,rest_pass=rest,
                full_success=success,recorded_elapsed_s=float(times[-1]),
                complete_time_s=float(times[-1]) if success else None,
                endpoint_marker='o' if np.linalg.norm(positions[-1]-ref[-1])<=1.25 else 'x',
                local_bounds_mm=binding['local_bounds_mm'],segments_checked=len(positions)-1,
                compared_as_original_baseline_not_matched_joint_ablation=True)
            write_json(stage/(current+'.json'),result);results.append(result)
            print(result,flush=True)
        write_json(stage/'results.json',dict(results=results,complete=True))
        receipt(stage,dict(status='original_nnc_saved_commands_new_geometry_checked',cases=3,
            derived_archives_loaded=loaded,command_segments_checked=checked,models=0,
            new_commands=0,training_decisions=0,wall_s=time.perf_counter()-started,goal_complete=False))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),case=current,
            archives_loaded=loaded,segments_checked=checked,automatic_retry=False))
        raise


if __name__=='__main__':main()
