"""Official COPP TOPP3-SOCP on the project's unchanged S2 fixed geometry.

Only this adapter is new: the upstream solver and S2 geometry source are not
edited. This is not Li2023/TOTP-SPLP reproduction or a global-optimum claim.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"literature"/"copp_runtime"))
import numpy as np
import copp_py as copp
from baselines.legacy_s2 import prepare_geometry
from metrics import metrics
from runtime_configuration import configure_runtime

SETTINGS=dict(max_station_spacing_mm=.5,min_curve_intervals=16,min_line_intervals=2,
              polygon_sides=32,iterations=3,corner_scale=1.,velocity_reserve=.999,
              acceleration_reserve=.99,jerk_reserve=.99,terminal_hold_s=.025,
              spatial_scale_mm=100.,allow_almost_solved=True,max_complete_time_s=1200.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def flatten(curve):
    if hasattr(curve,"pieces"):
        return [p for child in curve.pieces for p in flatten(child)]
    return [curve]


def derivatives(piece,local):
    local=np.asarray(local,float)
    if piece.kind=="line":
        t=np.broadcast_to(piece.tangent,(len(local),2)).copy()
        return t,np.zeros_like(t),np.zeros_like(t)
    if piece.kind=="arc":
        t=piece.tangent_at(local)
        k=np.full(len(local),1/piece.radius)
        ks=np.zeros(len(local))
    elif piece.kind=="bezier5":
        t,k,ks=piece.geometry_u(piece.parameter(local))
    else:raise ValueError("Unsupported S2 primitive "+piece.kind)
    n=np.column_stack([-t[:,1],t[:,0]])
    return t,k[:,None]*n,ks[:,None]*n-k[:,None]**2*t


def grid_geometry(curve,source_geometry,settings):
    pieces=flatten(curve)
    stations=[];first=[];second=[];third=[];boundaries=[]
    reflection=np.ones(2)
    if source_geometry is not None and any(p.get("sweep_rad",0.)<0 for p in source_geometry["primitives"]):
        reflection=np.array([1.,-1.])
    offset=0.;sub_ulp_pieces=[]
    for index,piece in enumerate(pieces):
        if piece.length < 1e-12 and piece.length <= 4*np.spacing(max(1.,offset)):
            # Keep the original piece in curve.position and source traversal.
            # A positive roundoff remnant cannot support a distinct station in
            # binary64 cumulative arc length; do not manufacture zero ds rows.
            sub_ulp_pieces.append(dict(piece_index=index,length_mm=piece.length,
                source_start=piece.source_start,source_end=piece.source_end))
            offset+=piece.length
            continue
        count=max(settings["min_line_intervals"] if piece.kind=="line" else settings["min_curve_intervals"],
                  math.ceil(piece.length/settings["max_station_spacing_mm"]))
        local=np.linspace(0.,piece.length,count+1)
        d1,d2,d3=derivatives(piece,local)
        d1*=reflection;d2*=reflection;d3*=reflection
        if index:
            # Retain the previous piece's endpoint as the grid node and add
            # the next side's jerk rows there. C2 joins may have a jerk jump.
            boundaries.append(dict(index=len(stations)-1,first=d1[0],second=d2[0],third=d3[0],
                velocity_jump=float(np.linalg.norm(first[-1]-d1[0])),
                curvature_jump=float(np.linalg.norm(second[-1]-d2[0]))))
            local,d1,d2,d3=local[1:],d1[1:],d2[1:],d3[1:]
        stations.extend((offset+local).tolist())
        first.extend(d1);second.extend(d2);third.extend(d3)
        offset+=piece.length
    sigma=np.array(stations)
    if np.any(np.diff(sigma)<=1e-12):raise ValueError("Unresolved near-duplicate station")
    if any(x["velocity_jump"]>1e-7 or x["curvature_jump"]>1e-7 for x in boundaries):
        raise ValueError("S2 geometric join is not C2 under arc-length derivatives")
    scale=settings["spatial_scale_mm"]
    curve.metadata["sub_ulp_pieces_without_internal_stations"]=sub_ulp_pieces
    return sigma,np.array(first),np.array(second)*scale,np.array(third)*scale**2,boundaries


def setup_problem(reference,tolerance,source_geometry,settings,limits):
    clocks={};clock=time.perf_counter()
    curve,_,_,_,source,extra=prepare_geometry(reference,tolerance=tolerance,limits=limits,
        corner_scale=settings["corner_scale"],source_geometry=source_geometry)
    clocks["geometry_wall_s"]=time.perf_counter()-clock
    clock=time.perf_counter()
    sigma,d1,d2,d3,boundaries=grid_geometry(curve,source_geometry,settings)
    scale=settings["spatial_scale_mm"]
    s=sigma/scale
    q=(curve.position(sigma)-reference[0])/scale
    con=copp.Constraints(2,capacity=len(s));con.append_s(s)
    vmax=limits[0]/scale*settings["velocity_reserve"]
    amax=limits[1]/scale*settings["acceleration_reserve"]
    jmax=limits[2]/scale*settings["jerk_reserve"]
    m=settings["polygon_sides"]
    angles=2*np.pi*np.arange(m)/m
    normals=np.column_stack([np.cos(angles),np.sin(angles)])
    factor=np.cos(np.pi/m)
    con.add_constraint_1st((vmax**2/np.sum(d1*d1,axis=1))[:,None],0)
    con.add_constraint_2nd(d2@normals.T,d1@normals.T,np.full((len(s),m),amax*factor),0)
    con.add_constraint_3rd(d3@normals.T,3*d2@normals.T,d1@normals.T,
        np.zeros((len(s),m)),np.full((len(s),m),jmax*factor),0)
    for row in boundaries:
        con.add_constraint_3rd((row["third"]*scale**2@normals.T)[None,:],
            (3*row["second"]*scale@normals.T)[None,:],(row["first"]@normals.T)[None,:],
            np.zeros((1,m)),np.full((1,m),jmax*factor),row["index"])
    clocks["grid_constraints_wall_s"]=time.perf_counter()-clock
    clock=time.perf_counter()
    seed=copp.solver.topp2_ra.solve(copp.solver.topp2_ra.Problem(con,
        idx_s_interval=(0,len(s)-1),a_boundary=(0.,0.)))
    con.amax_substitute(seed,0)
    clocks["seed_wall_s"]=time.perf_counter()-clock
    metadata=dict(method="COPP 0.2.3 TOPP3-SOCP with unchanged S2 fixed geometry",
        upstream_commit="a5fd651b9a1e9d91a0f03ff785057cd74712d9e2",settings=settings,
        station_count=len(s),piece_count=len(flatten(curve)),length_mm=curve.length,
        source_sha256={x:sha(source/x) for x in ["geometry.py"]},
        geometry_adapter_sha256=sha(ROOT/"src/baselines/legacy_s2.py"),
        adapter_sha256=sha(__file__),geometry=curve.metadata,small_angle_extrapolated_degrees=extra,
        constraint_adaptation="Exact norm speed cap; 32-direction polygon contained in A/J disks, both one-sided geometric jerk rows at C2 joins.",
        effective_vector_limits=[vmax*scale,amax*scale,jmax*scale],
        max_join_tangent_mismatch=max((x["velocity_jump"] for x in boundaries),default=0.),
        max_join_curvature_mismatch=max((x["curvature_jump"] for x in boundaries),default=0.),clocks=clocks)
    return curve,s,con,seed,metadata


def sample(curve,s,profile,dt,settings):
    clock=time.perf_counter()
    total,t_s=copp.interpolation.s_to_t_topp3(s,profile,0.)
    if not 0<total<settings["max_complete_time_s"]:raise ValueError("Finite complete-command budget exceeded")
    times=np.arange(math.ceil(total/dt)+math.ceil(settings["terminal_hold_s"]/dt)+1)*dt
    active=times<total
    progress=np.full(len(times),s[-1])
    progress[active]=copp.interpolation.t_to_s_topp3_samples(s,profile,t_s,times[active])
    if not np.isfinite(progress).all() or np.any(np.diff(progress)<-1e-10):
        raise ValueError("Nonfinite or reversing full-path time mapping")
    positions=curve.position(progress*settings["spatial_scale_mm"])
    return positions,times,progress,dict(continuous_motion_duration_s=total,
        complete_command_time_s=float(times[-1]),sampling_wall_s=time.perf_counter()-clock,
        stationary_intervals=list(profile.num_stationary),boundary_a=[float(profile.a[0]),float(profile.a[-1])],
        boundary_b=[float(profile.b[0]),float(profile.b[-1])])


def run_case(path_id,reference,tolerance,source_geometry,output,settings=SETTINGS,limits=(100.,2000.,62500.)):
    output.mkdir(parents=True,exist_ok=True)
    record=dict(path_id=path_id,tolerance_mm=tolerance,limits=list(limits),iterations=[])
    started=time.perf_counter()
    try:
        curve,s,con,seed,meta=setup_problem(reference,tolerance,source_geometry,settings,limits)
        record["planner"]=meta
        previous_b=None
        for iteration in range(1,settings["iterations"]+1):
            clock=time.perf_counter()
            kwargs={} if previous_b is None else dict(b_linearization=previous_b)
            problem=copp.solver.topp3_socp.Problem(con,seed,idx_s_start=0,
                a_boundary=(0.,0.),b_boundary=(0.,0.),num_stationary_max=1,**kwargs)
            profile=copp.solver.topp3_socp.solve(problem,copp.solver.topp3_socp.Options(
                allow_almost_solved=settings["allow_almost_solved"]))
            result=dict(iteration=iteration,solver_wall_s=time.perf_counter()-clock,
                native_node_violations=list(con.exceed_topp3(profile)),runs=[])
            np.savez_compressed(output/f"{path_id}_iteration{iteration}_profile.npz",s=s,a=profile.a,b=profile.b,
                reference=reference,num_stationary=profile.num_stationary)
            record["iterations"].append(result)
            write(output/(path_id+".json"),record)
            for dt in (.001,.0005):
                clock=time.perf_counter()
                positions,times,progress,execution=sample(curve,s,profile,dt,settings)
                configure_runtime("v3",tolerance=tolerance)
                metric=metrics(reference,positions,dt=dt,tolerance=tolerance,limits=limits)
                filename=f"{path_id}_iteration{iteration}_dt{round(dt*1e6)}us.npz"
                np.savez_compressed(output/filename,positions=positions,time=times,progress=progress,
                    reference=reference,dt_s=dt)
                result["runs"].append(dict(dt_s=dt,metrics=metric,execution=execution,
                    sample_audit_save_wall_s=time.perf_counter()-clock,trace=filename,trace_sha256=sha(output/filename)))
                write(output/(path_id+".json"),record)
                print(json.dumps(dict(path=path_id,iteration=iteration,dt=dt,success=metric["success"],
                    T=metric["complete_time_s"],peaks=metric["vector_peaks"],reason=metric["reason"])),flush=True)
            seed=profile.a;previous_b=profile.b
        record["status"]="completed"
    except Exception as exc:
        record.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        print(json.dumps(dict(path=path_id,status="failed",error=str(exc))),flush=True)
    record["total_wall_s"]=time.perf_counter()-started
    write(output/(path_id+".json"),record)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,
        default=ROOT/"results/copp_development_v4");parser.add_argument("--case",choices=["dense","validation_trapezoid","validation_zigzag","train_sine","all"],default="all")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for name in ("dense","validation_trapezoid","validation_zigzag","train_sine"):
        if args.case not in (name,"all"):continue
        source=(ROOT/"baseline_records/supplementary_reproduction/cases/dense/reference.csv") if name=="dense" else ROOT/"data"/(name+".csv")
        geometry=json.loads(source.with_name("geometry.json").read_text("utf-8")) if name=="dense" else None
        chord=max((x.get("chord_error_bound_mm",0.) for x in geometry["primitives"]),default=0.) if geometry else 0.
        cases.append((name,source,geometry,1.25-chord))
    registration=dict(method="COPP/S2 geometry development adapter",settings=SETTINGS,limits=[100.,2000.,62500.],
        command_periods_s=[.001,.0005],nominal_tolerance_mm=1.25,final_test_paths_executed=0,
        final_iteration_rule="Third solve is fixed as candidate output. All intermediate runs and failures retained; no outcome-based choice.",
        scope="Development only; not a claim to reproduce Li2023 or IJMTM TOTP-SPLP.",
        source_sha256=sha(__file__),cases=[dict(id=n,file=str(p),sha256=sha(p),tolerance_mm=tol) for n,p,g,tol in cases])
    reg=args.output/("registration_"+args.case+".json")
    if reg.exists():raise FileExistsError("Do not overwrite existing evidence: "+str(reg))
    write(reg,registration)
    records=[run_case(n,np.loadtxt(p,delimiter=",",skiprows=1),tol,g,args.output) for n,p,g,tol in cases]
    write(args.output/("summary_"+args.case+".json"),records)


if __name__=="__main__":main()
