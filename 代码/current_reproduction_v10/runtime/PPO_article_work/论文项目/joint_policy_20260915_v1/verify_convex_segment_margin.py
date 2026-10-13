"""Check an exact convex upper-bound tightening without new physical runs."""
import time
import numpy as np
from audit_function_constraint_interference import command_segment_margin
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_waveform_learned_update_20260917'
    stage=register('convex_segment_margin_bound_verification_20260917',
        [__file__,BASE/'audit_function_constraint_interference.py',source/'results.json',
         source/'receipt.json',BASE/'joint_runtime.py',BASE/'geometry_spec.py'],
        dict(analytical_cases=3,saved_complete_paths=128,new_physics_ticks=0,
             new_actor_decisions=0,wall_s=30,tolerance_mm=1e-6),
        'The Lipschitz-only diagnostic subdivides coincident straight intervals '
        'unnecessarily. Tighten its interval upper bound using convex Euclidean '
        'distance to each closed reference segment: min_j max(d(a,Sj),d(b,Sj)) '
        'also bounds distance to the reference union everywhere on[a,b]. Take '
        'the minimum of this bound and the existing Lipschitz bound. Preserve '
        'the declared1e-6 enclosure, physical runtime,1.25 tolerance and complete '
        'acceptance. Verify analytical straight/parallel/kink cases and interval '
        'compatibility with all128 previously saved parent/candidate certificates. '
        'No policy run is repeated; this changes calculation cost, not limits.')
    start=time.perf_counter();tests=[]
    ref=np.array([[0.,0.],[100.,0.]])
    for y,expected in ((0.,0.),(.75,.75)):
        rows=np.column_stack([np.linspace(.1,100.,1000),np.full(1000,y)])
        r=command_segment_margin(ref,rows)
        assert r['iterations']==0 and r['max_distance_lower_mm']<=expected+1e-12<=r['max_distance_upper_mm']+1e-12
        assert abs(r['max_distance_lower_mm']-expected)<1e-12
        tests.append(dict(case='straight' if y==0 else 'parallel_offset',expected=expected,result=r))
    ref=np.array([[-1.,0.],[0.,0.],[0.,1.]])
    rows=np.array([[-.1,0.],[0.,.1],[0.,1.]])
    r=command_segment_margin(ref,rows)
    assert abs(r['max_distance_lower_mm']-.05)<1e-12
    tests.append(dict(case='corner_cut_with_interior_maximum',expected=.05,result=r))
    data=read(source/'results.json');index=data['metric_names'].index('segment_max_distance_mm')
    checks=[]
    for arm in ('full_parent','full_candidate'):
        records=data[arm]
        for row,trace in zip(records['metrics'],records['traces']):
            with np.load(BASE/trace) as z:r=command_segment_margin(z['reference'],z['rows'])
            old_upper=row['values'][index];new_upper=r['max_distance_upper_mm']
            assert abs(old_upper-new_upper)<=1e-6+1e-10,(arm,row['path'],old_upper,new_upper)
            checks.append(dict(arm=arm,path=row['path'],old_upper_mm=old_upper,
                new_upper_mm=new_upper,iterations=r['iterations']))
    elapsed=time.perf_counter()-start
    assert len(checks)==128 and elapsed<30
    write_json(stage/'results.json',dict(analytical=tests,records=checks,
        maximum_saved_bound_difference_mm=max(abs(r['old_upper_mm']-r['new_upper_mm']) for r in checks),
        physical_acceptance_changed=False,goal_complete=False))
    receipt(stage,dict(status='verified_convex_segment_margin_tightening',saved_paths=128,
        new_physics_ticks=0,new_actor_decisions=0,wall_s=elapsed))
    print('CONVEX_MARGIN_CHECK',len(checks),'paths','seconds',elapsed,
        'max_bound_difference',max(abs(r['old_upper_mm']-r['new_upper_mm']) for r in checks),flush=True)


if __name__=='__main__':main()
