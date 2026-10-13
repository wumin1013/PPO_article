"""Add one vector-audited terminal family after the original cubic qualifies.

Call install(vendor_dir=...) after runtime_physical_time_v5.install. This changes
only terminal candidate selection at the existing decision boundary. It neither
changes policy actions nor requests an earlier terminal check. The original
cubic, its endpoint trigger and its qualified rest duration remain the fallback.
"""
import copy
import hashlib
import inspect
import math
from pathlib import Path
import sys
import time
import numpy as np

VERSION='v9_scalar_scurve_normal_cubic'
SCALAR_FRACTION=.98
MAX_PREFIX_S=.030
ENDPOINT_TRIGGER_MM=1.


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def differences(velocity,initial_acceleration,dt):
    acceleration=np.vstack((initial_acceleration,np.diff(velocity,axis=0)/dt))
    jerk=np.diff(acceleration,axis=0)/dt
    peaks=np.array([np.linalg.norm(velocity[1:],axis=1).max(),
        np.linalg.norm(acceleration[1:],axis=1).max(),np.linalg.norm(jerk,axis=1).max()])
    return acceleration,jerk,peaks


def qualified_rest_ticks(velocity,initial_acceleration,dt):
    acceleration,_,_=differences(velocity,initial_acceleration,dt)
    resting=(np.linalg.norm(velocity[1:],axis=1)<=.001)&(np.linalg.norm(acceleration[1:],axis=1)<=.01)
    nonrest=np.flatnonzero(~resting)
    return len(resting)-(int(nonrest[-1])+1 if len(nonrest) else 0)


def candidate_columns(position,velocity,acceleration,dt,prefix_ticks,rest_ticks,*,terminal,ruckig):
    limits=np.array([terminal.LIMITS['MAX_VEL'],terminal.LIMITS['MAX_ACC'],terminal.LIMITS['MAX_JERK']])
    speed=float(np.linalg.norm(velocity))
    if speed<=1e-8:raise ValueError('No scalar velocity frame at rest; retain the original cubic')
    tangent=velocity/speed;normal=np.array([-tangent[1],tangent[0]])
    frame=np.column_stack((tangent,normal))
    local_v=velocity@frame;local_a=acceleration@frame
    prefix=np.asarray([local_v+k*dt*local_a for k in range(prefix_ticks+1)])
    inp=ruckig.InputParameter(1);inp.control_interface=ruckig.ControlInterface.Velocity
    inp.current_position=[0.];inp.current_velocity=[float(prefix[-1,0])]
    inp.current_acceleration=[float(local_a[0])]
    inp.target_velocity=[0.];inp.target_acceleration=[0.]
    inp.max_velocity=[float(limits[0])]
    inp.max_acceleration=[float(SCALAR_FRACTION*limits[1])]
    inp.max_jerk=[float(SCALAR_FRACTION*limits[2])]
    trajectory=ruckig.Trajectory(1)
    status=ruckig.Ruckig(1).calculate(inp,trajectory)
    if status not in (ruckig.Result.Working,ruckig.Result.Finished):
        raise ValueError('Scalar Ruckig status: '+str(status))
    count=max(2,math.ceil(trajectory.duration/dt))
    tangent_speed=np.asarray([trajectory.at_time(min(k*dt,trajectory.duration))[1][0]
                              for k in range(count+1)])
    normal_speed=terminal.discrete_planar_stop_profile(
        [prefix[-1,1],0.],[local_a[1],0.],dt,count)[0][:,0]
    local=np.vstack((prefix,np.column_stack((tangent_speed,normal_speed))[1:]))
    vectors=local@frame.T;vectors[0]=velocity;vectors[-1]=0.
    pad=rest_ticks-qualified_rest_ticks(vectors,acceleration,dt)
    if pad<0:raise ValueError('Retain the original rest duration; candidate is already longer')
    vectors=np.vstack((vectors,np.zeros((pad,2))))
    acc,jerk,peaks=differences(vectors,acceleration,dt)
    points=np.vstack((position,position+np.cumsum(vectors[1:]*dt,axis=0)))
    size=len(vectors);j=np.vstack((np.full(2,np.nan),jerk))
    columns=dict(time_s=np.arange(size)*dt,x_mm=points[:,0],y_mm=points[:,1],
        velocity_x_mm_s=vectors[:,0],velocity_y_mm_s=vectors[:,1],
        acceleration_x_mm_s2=acc[:,0],acceleration_y_mm_s2=acc[:,1],
        jerk_x_mm_s3=j[:,0],jerk_y_mm_s3=j[:,1],
        v_mm_s=np.linalg.norm(vectors,axis=1),a_norm_mm_s2=np.linalg.norm(acc,axis=1),
        j_norm_mm_s3=np.linalg.norm(j,axis=1))
    metadata=dict(prefix_ticks=prefix_ticks,prefix_s=prefix_ticks*dt,
        scalar_continuous_duration_s=trajectory.duration,scalar_sampled_ticks=count,
        total_tail_s=(size-1)*dt,rest_ticks=rest_ticks,
        vector_peaks=peaks.tolist(),vector_limit_ratios=(peaks/limits).tolist(),
        vector_pass=bool(np.all(peaks<=limits*(1+1e-10))),
        nonreversing=bool(np.all(vectors@tangent>=-1e-10)))
    assert qualified_rest_ticks(vectors,acceleration,dt)==rest_ticks
    return columns,metadata


def geometry_pass(env,columns):
    points=np.column_stack((columns['x_mm'],columns['y_mm']))
    tracker=copy.copy(env.tracker)
    for key,value in vars(env.tracker).items():
        if key!='corridor':setattr(tracker,key,copy.deepcopy(value))
    for point in points[1:]:
        type(tracker).update(tracker,point)
        if not tracker.inside:return False
    return bool(tracker.snapshot(endpoint_tolerance=ENDPOINT_TRIGGER_MM)['geometry_complete'])


def consider(env,*,original_consider,joint,ruckig):
    original_consider(env)
    if env.brake is None:return
    started=time.perf_counter()
    original=env.brake
    old_velocity=np.column_stack((original['velocity_x_mm_s'],original['velocity_y_mm_s']))
    rest_ticks=qualified_rest_ticks(old_velocity,env.acceleration,joint.DT)
    old_time=float(original['time_s'][-1])
    diagnostics=dict(version=VERSION,original_tail_s=old_time,original_rest_ticks=rest_ticks,
        selected='original_cubic',selected_tail_s=old_time,candidates=[],geometry_checks=0)
    feasible=[]
    for prefix in range(round(MAX_PREFIX_S/joint.DT)+1):
        try:
            columns,item=candidate_columns(env.position,env.velocity,env.acceleration,joint.DT,
                prefix,rest_ticks,terminal=joint.terminal,ruckig=ruckig)
            error=float(np.linalg.norm(np.array([columns['x_mm'][-1],columns['y_mm'][-1]])-env.corridor.reference[-1]))
            item.update(endpoint_error_mm=error,geometry_checked=False)
            if (item['vector_pass'] and item['nonreversing'] and error<=ENDPOINT_TRIGGER_MM
                    and item['total_tail_s']<old_time-1e-12):
                feasible.append((columns,item))
            diagnostics['candidates'].append(item)
        except Exception as exc:
            diagnostics['candidates'].append(dict(prefix_ticks=prefix,error=type(exc).__name__+': '+str(exc)))
    # This is a fixed candidate-family search, not a path-specific parameter.
    # Once the shortest passing tail is found, longer candidates cannot improve it.
    for columns,item in sorted(feasible,key=lambda pair:(pair[1]['total_tail_s'],pair[1]['prefix_ticks'])):
        diagnostics['geometry_checks']+=1;item['geometry_checked']=True
        try:
            item['geometry_pass']=geometry_pass(env,columns)
        except Exception as exc:
            item.update(geometry_pass=False,geometry_error=type(exc).__name__+': '+str(exc))
            continue
        if item['geometry_pass']:
            env.brake=columns;env.brake_index=0
            diagnostics.update(selected=VERSION,selected_prefix_ticks=item['prefix_ticks'],
                selected_tail_s=item['total_tail_s'])
            break
    diagnostics['planning_wall_s']=time.perf_counter()-started
    env.v9_terminal_diagnostics=diagnostics


def install(*,vendor_dir):
    """Return dependency bindings; run after the unchanged v5 runtime install."""
    import joint_runtime as joint
    current=joint.JointRuntime._consider_terminal
    if getattr(current,'_terminal_candidate_v9',False):return dict(current._terminal_binding)
    vendor=Path(vendor_dir).resolve()
    sys.path.insert(0,str(vendor))
    import ruckig
    dependency=Path(ruckig.__file__).resolve()
    if not dependency.is_relative_to(vendor):raise ValueError('Ruckig must resolve inside the explicit frozen vendor')
    if getattr(ruckig,'__version__',None)!='0.12.2':raise ValueError('This candidate binds Ruckig0.12.2')
    binding=dict(version=VERSION,source_sha256=_sha(__file__),ruckig_version=ruckig.__version__,
        ruckig_module=str(dependency),ruckig_sha256=_sha(dependency),scalar_fraction=SCALAR_FRACTION,
        max_prefix_s=MAX_PREFIX_S,endpoint_trigger_mm=ENDPOINT_TRIGGER_MM,
        base_consider_source=str(inspect.getsourcefile(current)),
        base_consider_source_sha256=_sha(inspect.getsourcefile(current)),
        original_cubic_fallback=True,qualified_rest_ticks_unchanged=True,
        terminal_check_frequency_unchanged=True,whole_interval_gate_audit=True,
        limits_unchanged=True,policy_actions_unchanged=True)
    def wrapped(env):return consider(env,original_consider=current,joint=joint,ruckig=ruckig)
    wrapped._terminal_candidate_v9=True;wrapped._terminal_binding=binding
    joint.JointRuntime._consider_terminal=wrapped
    return dict(binding)
