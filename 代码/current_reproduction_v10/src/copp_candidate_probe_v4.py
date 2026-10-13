"""Audit probe for the official COPP 0.2.3 implementation (not project results).

Runs the official tutorial's exact sinusoidal path using analytic derivatives
instead of JAX. Separately checks a 2-D circle with conservative polygonal
approximations to Euclidean derivative bounds through the public raw API.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "literature" / "copp_runtime"))
import numpy as np
import copp_py as copp


def solve_case(name, s, q, dq, ddq, dddq, polygon_sides=None):
    start = time.perf_counter()
    robot = copp.Robot(q.shape[1], capacity=len(s))
    robot.append_s(s)
    robot.set_q(q, dq, ddq, 0, dddq=dddq)
    if polygon_sides is None:
        upper = np.ones(q.shape[1])
        robot.add_velocity_limits(upper, -upper, start_idx_s=0)
        robot.add_acceleration_limits(upper, -upper, start_idx_s=0)
        robot.add_jerk_limits(upper, -upper, start_idx_s=0)
    else:
        # Intersection n_i.x <= cos(pi/M) is a regular M-gon contained
        # in the unit disk. This is an explicit adaptation, not a native norm
        # constraint. Every constrained direction is represented, not just x/y.
        angle = 2*np.pi*np.arange(polygon_sides)/polygon_sides
        normal = np.column_stack([np.cos(angle), np.sin(angle)])
        limit = np.cos(np.pi/polygon_sides)
        con = robot.constraints
        con.add_constraint_1st((1/np.sum(dq*dq,axis=1))[:,None],0)
        con.add_constraint_2nd(ddq@normal.T,dq@normal.T,np.full((len(s),polygon_sides),limit),0)
        con.add_constraint_3rd(dddq@normal.T,3*ddq@normal.T,dq@normal.T,
                               np.zeros((len(s),polygon_sides)),
                               np.full((len(s),polygon_sides),limit),0)
    seed = copp.solver.topp2_ra.solve(copp.solver.topp2_ra.Problem(
        robot.constraints, idx_s_interval=(0,len(s)-1),a_boundary=(0.,0.)))
    robot.constraints.amax_substitute(seed,0)
    seed_time = time.perf_counter()-start
    records = []
    arrays = dict(s=s,q=q,dq=dq,ddq=ddq,dddq=dddq,seed=seed)
    for iteration in range(2):
        clock = time.perf_counter()
        problem = copp.solver.topp3_socp.Problem(robot.constraints, seed, idx_s_start=0,
            a_boundary=(0.,0.), b_boundary=(0.,0.),num_stationary_max=1)
        profile = copp.solver.topp3_socp.solve(problem,copp.solver.topp3_socp.Options(allow_almost_solved=True))
        solve_time = time.perf_counter()-clock
        final_time, t_s = copp.interpolation.s_to_t_topp3(s,profile,0.)
        station = copp.interpolation.t_to_s_topp3_uniform(s,profile,t_s,.001,t0=0.,include_final=True)
        a,b = profile.a,profile.b
        v = dq*np.sqrt(np.maximum(a,0.))[:,None]
        acc = ddq*a[:,None]+dq*b[:,None]
        item = dict(iteration=iteration+1,duration_s=final_time,solver_wall_s=solve_time,
                    node_violations=list(robot.constraints.exceed_topp3(profile)),
                    stationary_intervals=list(profile.num_stationary),
                    boundary_a=[a[0],a[-1]],boundary_b=[b[0],b[-1]],
                    node_velocity_norm_max=float(np.linalg.norm(v,axis=1).max()),
                    node_acceleration_norm_max=float(np.linalg.norm(acc,axis=1).max()),
                    sample_count=len(station),station_monotone=bool(np.min(np.diff(station))>=-1e-12),
                    station_endpoints=[float(station[0]),float(station[-1])])
        records.append(item)
        arrays.update({f"a_{iteration+1}":a,f"b_{iteration+1}":b,
                       f"t_s_{iteration+1}":t_s,f"s_t_{iteration+1}":station})
        seed = a
    output = ROOT/"audit"/(name+"_v4.npz")
    np.savez_compressed(output,**arrays)
    return dict(name=name,station_count=len(s),constraint_type="axis_box" if polygon_sides is None else "inscribed_polygon",
                polygon_sides=polygon_sides,seed_and_setup_wall_s=seed_time,iterations=records,
                arrays=str(output),arrays_sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    s=np.linspace(0.,1.,1001)
    freq=np.array([2*np.pi,3*np.pi,5*np.pi])
    phase=np.array([0.,.3,.7])
    theta=s[:,None]*freq+phase
    cases=[]
    specifications=[("copp_official_sine_analytic",s,np.sin(theta),np.cos(theta)*freq,
                     -np.sin(theta)*freq**2,-np.cos(theta)*freq**3,None)]
    arc_s=np.linspace(0.,np.pi,201)
    q=np.column_stack([np.cos(arc_s),np.sin(arc_s)])
    dq=np.column_stack([-np.sin(arc_s),np.cos(arc_s)])
    specifications.append(("copp_circle_polygon32",arc_s,q,dq,-q,-dq,32))
    for spec in specifications:
        try:
            result=solve_case(*spec)
            result["status"]="solver_completed"
        except Exception as exc:
            result=dict(name=spec[0],status="failed",error_type=type(exc).__name__,error=str(exc))
        cases.append(result)
    report=dict(scope="Candidate native-solver probes only; no project paths, corridor checks, command-interval certification, or full-paper reproduction.",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                official_source_commit="a5fd651b9a1e9d91a0f03ff785057cd74712d9e2",version="0.2.3",
                runtime_path=str(Path(copp.__file__).resolve()),cases=cases)
    output=ROOT/"audit"/"copp_candidate_probe_v4.json"
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
