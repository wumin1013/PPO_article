"""Audit complete COPP circle commands with the project's shared evaluator."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"literature"/"copp_runtime"))
import numpy as np
import copp_py as copp
from metrics import metrics
from runtime_configuration import configure_runtime


def main():
    source=ROOT/"audit"/"copp_circle_polygon32_v4.npz"
    data=np.load(source)
    s=data["s"]
    reference=data["q"]
    # This prescribed unit-radius probe uses 1 mm/s, 1 mm/s², 1 mm/s³.
    # It is not a held-out project test case or a comparison result.
    nominal_tolerance=.005
    sagitta=float(1-np.cos(np.max(np.diff(s))/2))
    tolerance=nominal_tolerance-sagitta
    configure_runtime("v3",tolerance=tolerance)
    rows=[]
    for iteration in (1,2):
        a,b=data[f"a_{iteration}"],data[f"b_{iteration}"]
        profile=copp.Profile3rd(a,b,(1,1))
        total,t_s=copp.interpolation.s_to_t_topp3(s,profile,0.)
        h0,h1=s[1]-s[0],s[-1]-s[-2]
        # Exact stationary-block mappings from official interpolation.rs:
        # start s=s0+C0*t³; finish s=sf+C1*(t-T)³.
        c0=a[1]**1.5/(27*h0*h0)
        c1=a[-2]**1.5/(27*h1*h1)
        local_start=np.linspace(0,t_s[1],101)
        local_end=np.linspace(t_s[-2],total,101)
        mapped_start=copp.interpolation.t_to_s_topp3_samples(s,profile,t_s,local_start)
        mapped_end=copp.interpolation.t_to_s_topp3_samples(s,profile,t_s,local_end)
        terminal_formula=dict(coefficient_start=float(c0),coefficient_end=float(c1),
            start_duration=float(t_s[1]),end_duration=float(total-t_s[-2]),
            scalar_jerk_start=float(6*c0),scalar_jerk_end=float(6*c1),
            start_formula_max_error=float(np.max(np.abs(mapped_start-(s[0]+c0*local_start**3)))),
            end_formula_max_error=float(np.max(np.abs(mapped_end-(s[-1]+c1*(local_end-total)**3)))),
            analytic_boundary_speed=[0.,0.],analytic_boundary_acceleration=[0.,0.],
            start_join_speed_error=float(abs(3*c0*t_s[1]**2-np.sqrt(a[1]))),
            start_join_acceleration_error=float(abs(6*c0*t_s[1]-b[1])),
            end_join_speed_error=float(abs(3*c1*(t_s[-2]-total)**2-np.sqrt(a[-2]))),
            end_join_acceleration_error=float(abs(6*c1*(t_s[-2]-total)-b[-2])))
        for dt in (.001,.0005):
            # Fixed command periods throughout. Evaluate only actual in-domain
            # times, then hold the exact endpoint for at least 24 ms.
            times=np.arange(int(np.ceil(total/dt))+1+int(np.ceil(.024/dt)))*dt
            active=times<total
            station=np.full(len(times),s[-1])
            station[active]=copp.interpolation.t_to_s_topp3_samples(s,profile,t_s,times[active])
            position=np.column_stack([np.cos(station),np.sin(station)])
            result=metrics(reference,position,dt=dt,tolerance=tolerance,limits=(1.,1.,1.))
            v=np.diff(position,axis=0)/dt
            acc=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
            jerk=np.diff(np.vstack([np.zeros(2),acc]),axis=0)/dt
            name=f"copp_circle_iteration{iteration}_dt{dt:g}_v4.npz"
            output=ROOT/"audit"/name
            np.savez_compressed(output,t=times,s=station,position=position,
                                velocity=v,acceleration=acc,jerk=jerk,reference=reference)
            rows.append(dict(iteration=iteration,dt_s=dt,continuous_motion_duration_s=total,
                nominal_tolerance_mm=nominal_tolerance,arc_sagitta_reserved_mm=sagitta,
                terminal_formula=terminal_formula,metrics=result,
                sampled_jerk_peak_time_s=float(times[1:][np.argmax(np.linalg.norm(jerk,axis=1))]),
                arrays=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    report=dict(scope="Full prescribed unit-circle probe at two command periods; shared geometry/order, fixed-period backward differences and dwell acceptance. Not project held-out tests or a literature-paper reproduction.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        official_mapping_source="literature/copp_source/src/copp/copp3/interpolation.rs:309-320,370-380",
        runs=rows)
    output=ROOT/"audit"/"copp_time_audit_v4.json"
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps([dict(iteration=x["iteration"],dt=x["dt_s"],success=x["metrics"]["success"],peaks=x["metrics"]["vector_peaks"],reason=x["metrics"]["reason"]) for x in rows],indent=2))


if __name__=="__main__":main()
