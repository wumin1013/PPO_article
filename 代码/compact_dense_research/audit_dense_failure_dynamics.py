"""Offline failure localization from recorded physical commands, without simulation."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'historical_dense_failure_audits' / stamp
    output.mkdir(parents=True, exist_ok=False)
    geometry = json.loads((ROOT / 'dense_closed_case/geometry.json').read_text(encoding='utf-8'))
    corners = {c['reference_vertex_index']: c for c in geometry['sharp_corners']}
    cases = [('historical', ROOT/'historical_best_dense_initial/trace.npz', ROOT/'historical_best_dense_initial/result.json'),
             ('arc_support', ROOT/'historical_best_dense_local_arc_support/trace.npz', ROOT/'historical_best_dense_local_arc_support/result.json'),
             ('preview_floor3', ROOT/'historical_dense_cluster_preview_probe/floor_3/trace.npz', ROOT/'historical_dense_cluster_preview_probe/floor_3/result.json')]
    pilot = ROOT / 'historical_residual_ppo_seed922301_pilot'
    if (pilot/'evaluations.json').exists():
        evaluations = json.loads((pilot/'evaluations.json').read_text(encoding='utf-8'))
        for evaluation in evaluations:
            label = f"ppo_update{evaluation['update']}"
            cases.append((label, pilot/f"eval_{evaluation['update']:04d}.npz", pilot/f"eval_{evaluation['update']:04d}.json"))
    records = []
    for label, trace, result_file in cases:
        result = json.loads(result_file.read_text(encoding='utf-8'))
        if result['success']:
            continue
        data = np.load(trace)
        rows, reference, executed = data['rows'], data['reference'], data['executed_actions_1ms']
        n = len(rows)
        ids = np.arange(max(0, n-200), n)
        tail = rows[ids]
        speed = np.linalg.norm(tail[:,2:4], axis=1)
        acceleration = np.linalg.norm(tail[:,4:6], axis=1)
        jerk = np.linalg.norm(tail[:,6:8], axis=1)
        heading = np.arctan2(tail[:,9], tail[:,8])
        velocity_heading = np.arctan2(tail[:,3], tail[:,2])
        lag = (heading-velocity_heading+np.pi)%(2*np.pi)-np.pi
        curvature = (tail[:,2]*tail[:,5]-tail[:,3]*tail[:,4])/np.maximum(speed**3,1e-12)
        vertex = result['geometry']['gates_passed']
        corner = corners.get(vertex)
        exterior = np.full(len(ids), np.nan)
        outgoing = np.full(len(ids), np.nan)
        if corner and vertex<len(reference)-1:
            tangent=reference[vertex+1]-reference[vertex]
            tangent=tangent/np.linalg.norm(tangent)
            normal=np.sign(corner['signed_heading_change_deg'])*np.array([-tangent[1],tangent[0]])
            exterior=-(tail[:,:2]-reference[vertex])@normal
            outgoing=(tail[:,:2]-reference[vertex])@tangent
        series=np.column_stack([(ids+1)*.001, tail[:,:2],speed,acceleration,jerk,executed[ids],
                                np.rad2deg(heading),np.rad2deg(velocity_heading),np.rad2deg(lag),curvature,exterior,outgoing])
        series_file=output/(label+'_last200ms.csv')
        columns=['time_s','x_mm','y_mm','speed_mm_s','acceleration_mm_s2','jerk_mm_s3',
                 'requested_steering_rad','requested_feed_mm_s','requested_preview_mm','filtered_heading_deg',
                 'velocity_heading_deg','heading_velocity_lag_deg','signed_curvature_per_mm',
                 'exterior_to_last_gate_outgoing_line_mm','along_last_gate_outgoing_line_mm']
        with series_file.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(columns);writer.writerows(series)
        records.append(dict(label=label,trace=str(trace),trace_sha256=sha(trace),result=str(result_file),
            result_sha256=sha(result_file),failed_after_gate=vertex,last_passed_corner=corner,
            failure_prefix_s=n*.001,complete_time_s=None,last200ms=dict(
                requested_feed_range_mm_s=[float(executed[ids,1].min()),float(executed[ids,1].max())],
                requested_preview_range_mm=[float(executed[ids,2].min()),float(executed[ids,2].max())],
                actual_speed_range_mm_s=[float(speed.min()),float(speed.max())],
                peak_abs_heading_velocity_lag_deg=float(abs(np.rad2deg(lag)).max()),
                acceleration_above99percent_fraction=float(np.mean(acceleration>=1980.)),
                jerk_above99percent_fraction=float(np.mean(jerk>=61875.)),
                final_exterior_to_outgoing_line_mm=float(exterior[-1]),final_along_outgoing_line_mm=float(outgoing[-1])),
            series=str(series_file)))
    report=dict(kind='Offline recorded-motion failure dynamics audit',new_physics_ticks=0,new_ppo_updates=0,
        scope='Last200ms before each recorded failure; outgoing-line exterior is a diagnostic, not a replacement for analytic Euclidean corridor checks.',records=records)
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),records=[{k:v for k,v in x.items() if k in ['label','failed_after_gate','last200ms']} for x in records]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
