"""Check all saved motion points, not only the ordered gate event."""
import math
import numpy as np
from euclidean_preview_geometry import circle_exit_target
from joint_runtime_v5 import environment_factory
from study_io import BASE,register,receipt,write_json


def main():
    parent=BASE/'constrained_parameter_step_probe_20260916/validation_1'
    stage=register('circle_query_continuity_audit_20260916',[__file__,BASE/'euclidean_preview_geometry.py']+[parent/f'{i}.npz' for i in range(4)],
        dict(physics_ticks=0,training_decisions=0,fixed_preview_values=[3.,8.],wall_s=60),
        'Falsify the assumption that removing the target jump at a gate ensures globally continuous guidance. Along saved physical motion hold D fixed, compare consecutive1ms arc and ordered-circle target bearings. Counterfactual geometry queries are not new trajectories or policy performance.')
    records=[]
    for path in range(4):
        with np.load(parent/f'{path}.npz') as z:ref,rows=z['reference'],z['rows']
        for distance in (3.,8.):
            env=environment_factory(ref);angles=[[],[]];valid=[];positions=[]
            for row in rows:
                if row[12]>.5:break
                env.position=row[:2].copy();env.tracker.update(env.position);progress=env.progress()[0]
                targets=[env.sample_reference(progress+distance)[0],circle_exit_target(env,progress,distance)[0]]
                for i,target in enumerate(targets):angles[i].append(math.atan2(*(target-env.position)[::-1]))
                valid.append(np.linalg.norm(row[2:4])>5 and min(np.linalg.norm(t-env.position) for t in targets)>.5);positions.append(env.position.copy())
            valid=np.array(valid);mask=valid[1:]&valid[:-1]
            result=dict(path=path,fixed_D_mm=distance)
            for name,angle in zip(('arc','circle'),angles):
                delta=np.rad2deg(np.abs(np.diff(np.unwrap(angle))));delta[~mask]=0.;peak=int(delta.argmax())
                result[name]=dict(maximum_1ms_bearing_change_deg=float(delta[peak]),tick=peak+2,position=positions[peak+1].tolist())
            records.append(result);print(result,flush=True)
    write_json(stage/'results.json',dict(records=records,physics_ticks=0,
        caveat='A first-forward-circle intersection can switch reference segments before or at the gate; same-state gate continuity is not a global continuity proof. No physical-guide adoption based on this geometry audit.'))
    receipt(stage,dict(status='completed_full_saved_motion_circle_query_audit',physics_ticks=0,training_decisions=0,conditions=len(records)))


if __name__=='__main__':main()
