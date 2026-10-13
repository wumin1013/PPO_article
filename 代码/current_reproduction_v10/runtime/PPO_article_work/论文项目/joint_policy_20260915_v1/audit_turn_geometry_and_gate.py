"""Saved-data audit of curvature concentration and guide jumps at ordered gates."""
import math
import numpy as np
from geometric_turn_metrics import geometric_turn_metrics
from joint_runtime_v5 import environment_factory
from joint_runtime import wrap
from study_io import BASE,register,receipt,write_json,sha


def gate_events(ref,rows,actions):
    env=environment_factory(ref);events=[]
    for i,row in enumerate(rows):
        if row[12]>.5:break
        if i%20==0:env.held=actions[i//20].copy()
        previous=env.tracker.frontier
        env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
        env.heading=math.atan2(row[9],row[8]);env.tracker.update(env.position)
        env.ticks=i+1
        if env.tracker.frontier!=previous:
            old_index=min(previous,len(env.corridor.lengths)-1)
            old_s=env.corridor.arcs[old_index]+np.clip((env.position-env.corridor.starts[old_index])@env.corridor.tangents[old_index],0.,env.corridor.lengths[old_index])
            new_s,new_index=env.progress()
            old_point,_=env.sample_reference(old_s+env.held[2]);new_point,_=env.sample_reference(new_s+env.held[2])
            old_delta=old_point-env.position;new_delta=new_point-env.position
            bearing_old=math.atan2(old_delta[1],old_delta[0]);bearing_new=math.atan2(new_delta[1],new_delta[0])
            tangent=env.corridor.tangents[new_index];heading=math.atan2(row[3],row[2])
            next_action=actions[min(i//20+1,len(actions)-1)]
            events.append(dict(tick=i+1,frontier_before=previous,frontier_after=env.tracker.frontier,
                progress_jump_at_same_position_mm=float(new_s-old_s),
                guide_bearing_jump_at_same_position_deg=math.degrees(wrap(bearing_new-bearing_old)),
                physical_heading_lag_to_outgoing_deg=math.degrees(wrap(heading-math.atan2(tangent[1],tangent[0]))),
                speed_at_gate_mm_s=float(np.linalg.norm(row[2:4])),held_before=env.held.tolist(),
                held_next_decision=next_action.tolist(),decision_delay_ms=20-(i%20+1)))
    return events


def main():
    folders=[('PPO_source','ar_conditional_joint_20260916/final_validation'),
             ('correlated_g4','correlated_parameter_search_20260916/validation_4'),
             ('balanced_g5','balanced_shape_time_learning_20260916/validation_5')]
    files=[BASE/name/f'{i}.npz' for _,name in folders for i in (1,2,3)]
    stage=register('turn_geometry_gate_audit_20260916',[__file__,BASE/'geometric_turn_metrics.py']+files,
        dict(saved_traces=9,physics_ticks=0,model_calls=0,training_updates=0),
        'Inspect true spatial turn concentration independently of speed, and same-position guide-coordinate jumps when an ordered gate changes. No causal physical intervention or new rollout.')
    records=[]
    for label,name in folders:
        for i in (1,2,3):
            with np.load(BASE/name/f'{i}.npz') as d:
                ref,rows,actions=d['reference'],d['rows'],d['actions']
                fine=geometric_turn_metrics(ref,rows,.01)
                coarser=geometric_turn_metrics(ref,rows,.02)
                events=gate_events(ref,rows,actions)
            record=dict(method=label,path=i,geometry=fine,coarser_energy=coarser['bending_energy_mm_inv'],events=events,
                trace_sha256=sha(BASE/name/f'{i}.npz'))
            records.append(record)
            print(label,i,'E',round(fine['bending_energy_mm_inv'],2),'normE',round(fine['normalized_bending_mm_inv'],2),
                '80%turn_mm',round(fine['middle80_turn_arc_length_mm'],2),'jumpdeg',round(events[0]['guide_bearing_jump_at_same_position_deg'],2),
                'next_feed',round(events[0]['held_next_decision'][1]-events[0]['held_before'][1],2),flush=True)
    write_json(stage/'results.json',dict(records=records,physics_ticks=0,model_calls=0,
        boundary='Spatial energy is a diagnostic approximated at 0.01/0.02 mm. Guide jumps at fixed physical position are algebraic properties of the current guide, not a measured causal contribution to overshoot.',goal_complete=False))
    receipt(stage,dict(status='completed_saved_geometry_and_guide_audit',saved_traces=9,physics_ticks=0,model_calls=0,training_updates=0))


if __name__=='__main__':main()
