"""Read-only comparison of actual gate crossings and final braking phase."""
from pathlib import Path
import json,sys
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha

def main():
    root=OUT/'all34_dominance_research';dest=root/'joint_bend_transit_refit_full/phase_propagation_diagnosis.json'
    assert not dest.exists()
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    paths={name:root/folder for name,folder in [('J3','joint_bend_transit_fit_full'),('J4','joint_bend_transit_refit_full'),('Fixed','fixed_actual_failure_recovery_full')]}
    reports={};sources={}
    for name,folder in paths.items():
        for resolution,dt,sub in [('1ms',.001,''),('half_ms',.0005,'half_ms')]:
            stage=folder/sub;trace=stage/'full_trace.npz';gates_file=stage/'ordered_gate_events.json';result_file=stage/'full_result.json'
            z=np.load(trace);rows=z['rows'];ref=z['reference'];events=json.loads(gates_file.read_text(encoding='utf-8'))
            by_vertex={e['vertex_index']:e for e in events};record=[]
            for corner in geometry['sharp_corners']:
                k=corner['reference_vertex_index'];event=by_vertex[k]
                gate_time=(event['command_segment']+event['fraction'])*dt
                first=int(event['command_segment']);u=ref[k+1]-ref[k];u=u/np.linalg.norm(u)
                after=rows[first:first+int(2/dt),:2];q=(after-ref[k])@u
                ids=np.flatnonzero((q[:-1]<50)&(q[1:]>=50))
                i=int(ids[0]);crossing=(first+i+1+(50-q[i])/(q[i+1]-q[i]))*dt
                record.append(dict(corner=corner['id'],gate_time_s=gate_time,out50_time_s=crossing))
            times=z['decision_times_s'] if 'decision_times_s' in z.files else np.arange(len(z['observations']))*.020
            obs=z['observations'];brake=np.flatnonzero(obs[:,20]>.5)
            last=events[-1];last_gate=(last['command_segment']+last['fraction'])*dt
            result=json.loads(result_file.read_text(encoding='utf-8'))
            reports[name+'_'+resolution]=dict(corners=record,last_gate_time_s=last_gate,complete_time_s=result['complete_time_s'],
                first_braking_observation_time_s=float(times[brake[0]]) if len(brake) else None,
                final_phase_s=result['complete_time_s']-last_gate)
            for p in (trace,gates_file,result_file):sources[str(p)]=sha(p)
    comparisons={}
    for left,right in [('J4_1ms','J3_1ms'),('J4_half_ms','J3_half_ms'),('J4_half_ms','J4_1ms')]:
        a=reports[left];b=reports[right];previous=0;records=[]
        for ac,bc in zip(a['corners'],b['corners']):
            difference=(ac['out50_time_s']-bc['out50_time_s'])*1000
            records.append(dict(corner=ac['corner'],cumulative_delay_ms=difference,increment_since_previous_out50_ms=difference-previous))
            previous=difference
        comparisons[left+'_minus_'+right]=dict(records=records,last_gate_delay_ms=(a['last_gate_time_s']-b['last_gate_time_s'])*1000,
            complete_delay_ms=(a['complete_time_s']-b['complete_time_s'])*1000)
    write_json(dest,dict(reports=reports,comparisons=comparisons,physics_ticks=0,policy_calls=0,source_sha256=sources,
        interpretation='Actual gate and outgoing50 crossing chronology; cumulative phase changes include memory and state propagation. They do not isolate an individual-corner causal effect. Registered corner deficit windows and full time remain unchanged.'))
    for key,c in comparisons.items():
        print(key,'last_gate_ms',c['last_gate_delay_ms'],'complete_ms',c['complete_delay_ms'])
        print(sorted(c['records'],key=lambda r:abs(r['increment_since_previous_out50_ms']),reverse=True)[:6])
    print({k:{a:b for a,b in v.items() if a!='corners'} for k,v in reports.items()})

if __name__=='__main__':main()
