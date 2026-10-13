"""Versioned full-path time dilation of saved COPP profiles; geometry unchanged.

The exact piecewise-quadratic speed peak is checked, including stationary
end-block maxima. A fixed additional 1% time reserve was set from development
failures before re-evaluation. This is a numerical adaptation, not an upstream
algorithm guarantee; full command checks still decide success.
"""
from pathlib import Path
import json
import time
import numpy as np
from copp_path_adapter_v4 import ROOT,SETTINGS,copp,prepare_geometry,sample,sha,write
from metrics import metrics
from runtime_configuration import configure_runtime


def speed_peak(s,a,b,num_stationary):
    peak=float(max(a))
    n=len(s)-1
    for i in range(num_stationary[0],n-num_stationary[1]):
        h=s[i+1]-s[i]
        c=(b[i+1]-b[i])/h
        right=a[i]>0 and (a[i+1]==0 or i+1==n-num_stationary[1])
        anchor_a=a[i+1] if right else a[i]
        anchor_b=-b[i+1] if right else b[i]
        points=[0.,h]
        if c<0 and 0 < -anchor_b/c < h:points.append(-anchor_b/c)
        peak=max(peak,max(anchor_a+2*anchor_b*x+c*x*x for x in points))
    return float(np.sqrt(peak))


def main():
    original=ROOT/"results/copp_development_v4"
    output=ROOT/"results/copp_development_v4b"
    output.mkdir(parents=True,exist_ok=True)
    candidates=list(original.glob("*_profile.npz"))
    registration=dict(version="v4b-time-projection",previous="copp_development_v4",settings=SETTINGS,
        additional_fixed_time_reserve=1.01,
        exact_speed_dilation_rule="lambda=1.01*max(1,peak_interpolated_speed/(.999*V)); a,b <- a,b / lambda^2",
        source_sha256=sha(__file__),adapter_sha256=sha(ROOT/"src/copp_path_adapter_v4.py"),
        inputs={p.name:sha(p) for p in candidates},final_iteration_rule="third iteration remains fixed; all iterations retained",
        reason="Development revealed node-to-interval velocity overshoot and slight dense jerk overshoot; original failures preserved.",
        scope="Three completed development paths; original train_sine pre-COPP geometry failure remains in original evidence.")
    if (output/"registration.json").exists():raise FileExistsError("Preserve existing projection results")
    write(output/"registration.json",registration)
    results=[]
    for name in ("dense","validation_trapezoid","validation_zigzag"):
        native=json.loads((original/(name+".json")).read_text("utf-8"))
        source=ROOT/"baseline_records/supplementary_reproduction/cases/dense" if name=="dense" else ROOT/"data"
        ref=np.loadtxt(source/("reference.csv" if name=="dense" else name+".csv"),delimiter=",",skiprows=1)
        geometry=json.loads((source/"geometry.json").read_text("utf-8")) if name=="dense" else None
        clock=time.perf_counter()
        curve,*_=prepare_geometry(ref,tolerance=native["tolerance_mm"],source_geometry=geometry)
        geometry_wall=time.perf_counter()-clock
        for iteration in (1,2,3):
            data=np.load(original/f"{name}_iteration{iteration}_profile.npz")
            s,a,b=data["s"],data["a"],data["b"]
            stationary=tuple(int(x) for x in data["num_stationary"])
            clock=time.perf_counter()
            peak=speed_peak(s,a,b,stationary)*SETTINGS["spatial_scale_mm"]
            dilation=1.01*max(1.,peak/(100*SETTINGS["velocity_reserve"]))
            profile=copp.Profile3rd(a/dilation**2,b/dilation**2,stationary)
            projection_wall=time.perf_counter()-clock
            np.savez_compressed(output/f"{name}_iteration{iteration}_profile.npz",s=s,a=profile.a,b=profile.b,
                num_stationary=stationary,time_dilation=dilation,reference=ref)
            for dt in (.001,.0005):
                pos,t,progress,execution=sample(curve,s,profile,dt,SETTINGS)
                configure_runtime("v3",tolerance=native["tolerance_mm"])
                clock=time.perf_counter()
                result=metrics(ref,pos,dt=dt,tolerance=native["tolerance_mm"])
                audit_wall=time.perf_counter()-clock
                key=f"{name}_iteration{iteration}_dt{round(dt*1e6)}us"
                np.savez_compressed(output/(key+".npz"),positions=pos,time=t,progress=progress,reference=ref,dt_s=dt)
                row=dict(path_id=name,iteration=iteration,dt_s=dt,lambda_time=dilation,
                    original_interpolated_peak_speed_mm_s=peak,projection_wall_s=projection_wall,
                    geometry_rebuild_wall_s=geometry_wall,audit_wall_s=audit_wall,
                    original_planning_clocks=native["planner"]["clocks"],
                    original_solver_wall_s=native["iterations"][iteration-1]["solver_wall_s"],
                    metrics=result,execution=execution,trace=key+".npz",sha256=sha(output/(key+".npz")))
                write(output/(key+".json"),row);results.append(row)
                print(json.dumps(dict(path=name,iteration=iteration,dt=dt,success=result["success"],
                    T=result["complete_time_s"],peaks=result["vector_peaks"],dilation=dilation)),flush=True)
    write(output/"summary.json",results)


if __name__=="__main__":main()
