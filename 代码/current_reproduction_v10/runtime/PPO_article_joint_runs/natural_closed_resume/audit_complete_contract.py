"""Audit saved trajectories against the physical task, separate from plot projection."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from geometry_spec import audit_commands
from study_io import write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder');args=ap.parse_args();folder=Path(args.folder)
    result=json.loads((folder/'full_result.json').read_text(encoding='utf-8'))
    trace=np.load(folder/'full_trace.npz');rows=trace['rows'];ref=trace['reference'];start=time.perf_counter()
    geometry,events=audit_commands(ref,np.vstack([ref[0],rows[:,:2]]),epsilon=1.25,closed=True,endpoint_tolerance=1.25)
    sequential=bool([e['gate'] for e in events]==list(range(geometry['gate_count'])))
    command_order=bool(all((a['command_segment'],a['fraction'])<(b['command_segment'],b['fraction']) for a,b in zip(events[:-1],events[1:])))
    m=json.loads((folder/'full_metrics.json').read_text(encoding='utf-8')) if (folder/'full_metrics.json').exists() else {}
    success=bool(result['error'] is None and result['reason']=='success' and geometry['geometry_complete'] and sequential and command_order
        and result['audit']['kinematic_pass'] and result['audit']['rest_pass'] and result.get('analytic',{}).get('analytic_corridor_pass',False)
        and result['ordinary_20ms_hold'] and m.get('all34_windows_complete',False))
    record=dict(kind='independent_saved_command_physical_contract_audit',success=success,
        complete_time_s=len(rows)*.001 if success else None,geometry=geometry,all_gates_in_order=sequential,command_order=command_order,
        physical_audit=result['audit'],analytic=result.get('analytic'),all34_windows_complete=m.get('all34_windows_complete'),
        projection_backward_steps_diagnostic=m.get('projection_raw_backward_steps'),
        qualification='Nearest-polyline projection is only a plotting/window coordinate. Monotonicity of that projection is not the registered ordered-gate criterion. All witnessed gate crossings and physical checks are audited without changing any trajectory or window.',
        historical_full_result_success=result['success'],historical_full_result_sha256=sha(folder/'full_result.json'),
        trace_sha256=sha(folder/'full_trace.npz'),wall_s=time.perf_counter()-start,ppo_updates=0,new_physics_ticks=0,goal_complete=False)
    write_json(folder/'complete_contract_audit.json',record);write_json(folder/'ordered_gate_events_audit.json',events)
    print('CONTRACT_AUDIT',folder.name,success,record['complete_time_s'],geometry,record['wall_s'],flush=True)

if __name__=='__main__':main()
