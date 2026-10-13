"""Independent property checks and complete successful/failed run records."""
import json
from pathlib import Path
import numpy as np
from baselines.motion_overlap import plan


def finite_segment_distance(x, p):
    result=np.full(len(x),np.inf)
    for a,b in zip(p[:-1],p[1:]):
        d=b-a; dd=np.dot(d,d)
        t=np.clip((x-a)@d/dd,0,1) if dd>0 else np.zeros(len(x))
        result=np.minimum(result,np.linalg.norm(x-a-t[:,None]*d,axis=1))
    return result


def main():
    out=Path(__file__).resolve().parents[2]/"literature"/"baseline_validation_v3"
    out.mkdir(parents=True,exist_ok=True)
    cases={
        "straight": np.array([[0,0],[30,0]]),
        "corner90": np.array([[0,0],[10,0],[10,10],[20,10]]),
        "uturn": np.array([[0,0],[10,0],[0,0],[0,10]]),
        "duplicate_and_collinear":np.array([[0,0],[0,0],[5,0],[10,0],[10,10]]),
        "short_dense":np.c_[np.arange(31)*.15,np.arange(31)%2*.2],
        "unequal_acute":np.array([[0,0],[100,0],[99.98,.01],[101,4],[101.1,4.01]]),
        "mixed_sampled_arc":np.r_[np.array([[0.,0.],[10.,0.]]),np.c_[10+5*np.sin(np.linspace(0,np.pi/2,21)[1:]),5-5*np.cos(np.linspace(0,np.pi/2,21)[1:])],np.array([[15.,15.]])],
    }
    records=[]
    for name,p in cases.items():
        for tol,lim,dt in [(1.25,dict(v=100,a=2000,j=62500),.001),(.25,dict(v=80,a=1000,j=30000),.0005),(.05,dict(v=100,a=2000,j=62500),.001)]:
            record=dict(case=name,tolerance_mm=tol,limits=lim,dt_s=dt)
            try:
                r=plan(p,tol,lim,dt)
                x=r['positions']; m=r['metrics'];
                # Check every endpoint and three interior positions of every
                # finite command chord. The analytic bound covers the rest.
                check=np.concatenate([x]+[x[:-1]*(1-a)+x[1:]*a for a in [.25,.5,.75]])
                measured=float(np.max(finite_segment_distance(check,p)))
                assert measured<=tol+1e-8,("geometry",measured,tol)
                assert m['sampled_command_chord_deviation_bound_mm']<=tol+1e-8
                for key,limit in [('vector_speed_max',lim['v']),('vector_acceleration_max',lim['a']),('vector_jerk_max',lim['j'])]:
                    assert m[key]<=limit*(1+1e-8),(key,m[key],limit)
                assert np.allclose(x[0],p[0]) and np.allclose(x[-1],p[-1])
                assert np.all(r['velocities'][0]==0) and np.all(r['velocities'][-1]==0)
                assert np.all(r['accelerations'][0]==0) and np.all(r['accelerations'][-1]==0)
                assert np.all(np.diff(r['corner_witness_times'])>0)
                assert m['ordered_vertex_witness_bound_mm']<=tol
                assert np.allclose(x[-int(np.ceil(.02/dt))-1:],p[-1])
                # Independent finite differences from positions, with initial
                # rest, verify uniform-grid vector kinematics too.
                xp=np.r_[np.repeat(x[:1],3,axis=0),x]
                vm=np.diff(xp,axis=0)/dt; am=np.diff(vm,axis=0)/dt; jm=np.diff(am,axis=0)/dt
                fd=[float(np.max(np.linalg.norm(a,axis=1))) for a in [vm,am,jm]]
                assert np.all(np.asarray(fd)<=np.asarray(list(lim.values()))*(1+1e-5)),fd
                stationary=(np.linalg.norm(vm[-len(x):],axis=1)<1e-7)&(np.linalg.norm(am[-len(x):],axis=1)<1e-7)
                trailing=int(np.argmax(~stationary[::-1])) if np.any(~stationary) else len(x)
                measured_rest=(trailing-1)*dt
                assert measured_rest>=.020-1e-12,measured_rest
                record.update(status='passed',observed_chord_deviation_mm=measured,finite_difference_maxima=fd,position_derived_stationary_duration_s=measured_rest,metrics=m)
            except Exception as e:
                record.update(status='failed',reason=repr(e))
            records.append(record)
    # Exact rotation invariance of Euclidean vector constraints.
    p=cases['corner90']; theta=.713
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    a=plan(p); b=plan(p@rotation.T)
    records.append(dict(case='rotation_invariance',status='passed' if a['metrics']['moving_time_s']==b['metrics']['moving_time_s'] and np.max(np.abs(a['positions']@rotation.T-b['positions']))<1e-8 else 'failed'))
    # Higher overlap must outperform the exact same no-overlap construction.
    a=plan(cases['corner90'],overlap_fraction=0); b=plan(cases['corner90'])
    records.append(dict(case='overlap_vs_same_scurve_stop_go',status='passed' if b['metrics']['completion_time_s']<a['metrics']['completion_time_s'] else 'failed',no_overlap_s=a['metrics']['completion_time_s'],overlap_s=b['metrics']['completion_time_s']))
    for name,p,lim in [('zero_path',np.zeros((2,2)),dict(v=100,a=2000,j=62500)),('invalid_limit',cases['straight'],dict(v=0,a=2000,j=62500)),('nonfinite_path',np.array([[0,0],[np.nan,0]]),dict(v=100,a=2000,j=62500))]:
        try:
            plan(p,limits=lim)
            records.append(dict(case=name,status='failed',reason='invalid input accepted'))
        except ValueError as e:
            records.append(dict(case=name,status='passed',expected_rejection=str(e)))
    (out/'property_checks.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(checks=len(records),passed=sum(r['status']=='passed' for r in records),failed=[r for r in records if r['status']=='failed']),indent=2))
    if any(r['status']=='failed' for r in records): raise SystemExit(1)


if __name__=='__main__': main()
