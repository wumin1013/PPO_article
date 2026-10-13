"""All84 registered windows and exact full-time decomposition from real traces."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compact_control import BASE
from geometry_spec import Corridor, GateTracker
from natural_case_metrics import cruise_metrics

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def analyze(trace,result_file,dt=.001):
    result=json.loads(result_file.read_text(encoding='utf-8'))
    if not result['success']:raise ValueError('Full-time/corner comparison requires a successful full trace')
    geometry=json.loads((ROOT/'dense_closed_case/geometry.json').read_text(encoding='utf-8'))
    reference=np.loadtxt(ROOT/'dense_closed_case/reference.csv',delimiter=',',skiprows=1)
    data=np.load(trace);rows=data['rows'];xy=rows[:,:2];speed=np.linalg.norm(rows[:,2:4],axis=1)
    if 'reference' in data:assert np.array_equal(data['reference'],reference)
    assert abs(len(rows)*dt-result['complete_time_s'])<1e-8
    corridor=Corridor(reference,closed=True);tracker=GateTracker(corridor,reference[0])
    frontiers=np.empty(len(rows),int)
    for i,position in enumerate(xy):
        tracker.update(position);frontiers[i]=tracker.frontier
    assert tracker.snapshot(1.25)['geometry_complete']
    current=np.minimum(frontiers,len(corridor.lengths)-1)
    # Plotting/window coordinate uses nearest of the current and adjacent segments,
    # anchored by actual witnessed gate order, including the closed seam.
    indices=np.stack([np.maximum(current-1,0),current,np.minimum(current+1,len(corridor.lengths)-1)],axis=1)
    relative=xy[:,None,:]-corridor.starts[indices]
    along=np.clip((relative*corridor.tangents[indices]).sum(2),0.,corridor.lengths[indices])
    distance=np.linalg.norm(relative-along[:,:,None]*corridor.tangents[indices],axis=2)
    selected=distance.argmin(1)
    progress=(corridor.arcs[indices]+along)[np.arange(len(rows)),selected]
    events={event['vertex_index']:event for event in tracker.events}
    covered=np.zeros(len(rows),bool);corners=[]
    for c in geometry['sharp_corners']:
        s=c['reference_s_mm'];mask=(progress>=s-c['window_before_mm'])&(progress<=s+c['window_after_mm'])
        assert not np.any(mask&covered);covered|=mask
        k=c['reference_vertex_index'];event=events[k]
        crossing=(event['command_segment']+event['fraction'])*dt
        ids=np.flatnonzero(mask)
        assert len(ids)>2 and (ids[0]+1)*dt<=crossing<=(ids[-1]+1)*dt
        outgoing=corridor.tangents[k];normal=np.sign(c['signed_heading_change_deg'])*np.array([-outgoing[1],outgoing[0]])
        after=ids[(ids+1)*dt>=crossing]
        deviation=(xy[after]-reference[k])@normal
        ss=speed[ids]
        curvature=(rows[ids,2]*rows[ids,5]-rows[ids,3]*rows[ids,4])/np.maximum(ss**3,1e-12)
        corners.append(dict(id=c['id'],angle_deg=c['interior_angle_deg'],window_before_mm=c['window_before_mm'],
            window_after_mm=c['window_after_mm'],duration_s=float(len(ids)*dt),crossing_time_s=float(crossing),
            speed_deficit_mm=float(np.maximum(100-ss,0.).sum()*dt),minimum_speed_mm_s=float(ss.min()),
            cruise99_s=float((ss>=99).sum()*dt),peak_curvature_mm_inv=float(np.max(abs(curvature[ss>5]))) if (ss>5).any() else None,
            outgoing_swing_in_window_mm=float(max(0.,-deviation.min())) if len(deviation) else None,
            first_index=int(ids[0]),last_index=int(ids[-1])))
    traveled=float(speed.sum()*dt)
    deficit=float((100-speed).sum()*dt)
    reconstructed=(traveled+deficit)/100.
    assert abs(reconstructed-len(rows)*dt)<1e-10
    split=[]
    for label,mask in [('all_corner_windows',covered),('outside_corner_windows',~covered)]:
        ss=speed[mask]
        split.append(dict(label=label,time_s=float(len(ss)*dt),executed_length_mm=float(ss.sum()*dt),
            speed_deficit_mm=float((100-ss).sum()*dt),cruise99_s=float((ss>=99).sum()*dt)))
    return dict(trace=str(trace),trace_sha256=sha(trace),success_result=str(result_file),result_sha256=sha(result_file),
        case_registration_sha256=sha(ROOT/'dense_closed_case/registration.json'),physical_dt_s=dt,
        complete_time_s=len(rows)*dt,executed_length_mm=traveled,speed_deficit_mm=deficit,
        time_identity_error_s=reconstructed-len(rows)*dt,cruise=cruise_metrics(speed,dt),
        all84_windows_complete=len(corners)==84,corners=corners,time_partition=split,
        definitions=dict(time='Full execution including final braking and physical rest.',
            identity='T=L_executed/100 + integral(100-|v|)dt/100; isolates path-length and speed-deficit contributions.',
            windows='Fixed before comparison: min(10mm,0.49*adjacent-corner separation), all84, no overlap.',
            swing='Maximum exterior displacement from outgoing line after the witnessed gate inside the fixed speed window; not a full recovery distance.',
            coordinate='Adjacent-segment nearest projection anchored by actual ordered gate frontier. Used only for plots and window indices.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trace',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--dt',type=float,default=.001)
    args=parser.parse_args();value=analyze(args.trace,args.result,args.dt)
    with args.out.open('x',encoding='utf-8') as file:json.dump(value,file,ensure_ascii=False,indent=2);file.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['corners','definitions']},ensure_ascii=False))
