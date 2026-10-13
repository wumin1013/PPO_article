"""Counterfactual target queries at identical saved physical gate states."""
import math
import numpy as np
from euclidean_preview_geometry import circle_exit_target
from joint_runtime_v5 import environment_factory
from joint_runtime import wrap
from study_io import BASE,register,receipt,write_json


def main():
    parent=BASE/'constrained_parameter_step_probe_20260916/validation_1'
    stage=register('guidance_target_gate_jump_audit_20260916',[__file__,BASE/'euclidean_preview_geometry.py',BASE/'joint_runtime.py']+[parent/f'{i}.npz' for i in (0,1,2,3)],
        dict(physics_ticks=0,training_decisions=0,gate_events=4),
        'Hold actual physical state and held D fixed, query old/new ordered gate frontiers. Measure original arc-progress target jump and ordered circle-exit alternative. Pure geometry diagnosis, no learned-policy claim or changed trajectory. Distinguish target discontinuity from earlier actor-observation counterfactual.')
    results=[]
    for path in range(4):
        with np.load(parent/f'{path}.npz') as z:ref,rows,actions=z['reference'],z['rows'],z['actions']
        env=environment_factory(ref)
        for tick,row in enumerate(rows):
            if row[12]>.5:break
            if tick%20==0:env.held=actions[tick//20].copy()
            env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
            env.heading=math.atan2(row[9],row[8]);env.ticks=tick+1
            old_frontier=env.tracker.frontier;old_progress=env.progress()[0]
            old_target=env.sample_reference(old_progress+env.held[2])[0]
            old_circle=circle_exit_target(env,old_progress,env.held[2])[0]
            env.tracker.update(env.position)
            if env.tracker.frontier==old_frontier:continue
            new_progress=env.progress()[0];new_target=env.sample_reference(new_progress+env.held[2])[0]
            new_circle=circle_exit_target(env,new_progress,env.held[2])[0]
            def angle(target):return math.atan2(*(target-env.position)[::-1])
            result=dict(path=path,tick=tick+1,held_preview_mm=float(env.held[2]),position=env.position.tolist(),
                speed_mm_s=float(np.linalg.norm(env.velocity)),projected_progress_jump_mm=new_progress-old_progress,
                arc_target_jump_mm=float(np.linalg.norm(new_target-old_target)),arc_bearing_jump_deg=math.degrees(wrap(angle(new_target)-angle(old_target))),
                circle_target_jump_mm=float(np.linalg.norm(new_circle-old_circle)),circle_bearing_jump_deg=math.degrees(wrap(angle(new_circle)-angle(old_circle))),
                old_arc_target=old_target.tolist(),new_arc_target=new_target.tolist(),old_circle_target=old_circle.tolist(),new_circle_target=new_circle.tolist())
            results.append(result);print(result,flush=True)
    assert len(results)==4
    write_json(stage/'results.json',dict(events=results,physics_ticks=0,
        limitation='Same-state same-D geometry queries establish only target discontinuity. Whether it causes the curvature spike or whether another guide improves closed-loop performance requires physical intervention. Circle preview has a different distance meaning and would require matched retraining/fixed-D comparison if adopted.'))
    receipt(stage,dict(status='completed_same_state_guidance_jump_audit',gate_events=len(results),physics_ticks=0,training_decisions=0))


if __name__=='__main__':main()
