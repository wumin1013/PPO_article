"""Bind actual policy requests and executed local motion; no new simulation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT,BASE,CASE,write,sha
from research_policy_loader import from_saved
from natural_case_metrics import ordered_projection


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--index',type=int,required=True);a=p.parse_args()
    folder=a.stage/f'mechanism_{a.index:04d}';folder.mkdir(exist_ok=False)
    checkpoint=a.stage/f'checkpoint_{a.index:04d}.pt'
    policy=from_saved(torch.load(checkpoint,map_location='cpu',weights_only=False))
    torch.set_num_threads(1);records=[];bindings={str(checkpoint):sha(checkpoint)}
    for case,case_path in [('original',BASE/'paper_evidence_review/natural_closed_case'),('dense',CASE)]:
        geom_path=case_path/'geometry.json';geometry=json.loads(geom_path.read_text(encoding='utf-8'))
        bindings[str(geom_path)]=sha(geom_path)
        for key,dt in [('1ms',.001),('half_ms',.0005)]:
            trace=a.stage/f'eval_{a.index:04d}_{case}_{key}.npz';result_path=trace.with_suffix('.json')
            result=json.loads(result_path.read_text(encoding='utf-8'))
            assert result['success'] and result['checkpoint_sha256']==sha(checkpoint) and result['trace_sha256']==sha(trace)
            with np.load(trace) as d:
                ref=d['reference'];rows=d['rows'];x=d['observations'];z=d['latents'];actions=d['actions'];times=d['decision_times_s']
            assert x.shape[1]==14 and z.shape[1]==3 and len(x)==len(actions)==len(times)
            assert np.max(abs(np.diff(times)-.02))<1e-10
            tick=np.rint(times/dt).astype(int)
            position=np.vstack([ref[0],rows[tick[1:]-1,:2]])
            decision_s=ordered_projection(ref,position);progress=ordered_projection(ref,rows[:,:2])
            speed=np.linalg.norm(rows[:,2:4],axis=1)
            curvature=(rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4])/np.maximum(speed,1e-8)**3
            selected_ids=geometry.get('mechanism_columns',[])
            checked=set(range(0,len(x),137));locals_data={};corners=[]
            for corner in geometry['sharp_corners']:
                center=corner['reference_s_mm'];k=corner['reference_vertex_index'];vertex=ref[k]
                incoming=ref[k]-ref[k-1];incoming/=np.linalg.norm(incoming)
                outgoing=ref[k+1]-ref[k];outgoing/=np.linalg.norm(outgoing)
                sign=np.sign(corner['signed_heading_change_deg'])
                normal=sign*np.array([-incoming[1],incoming[0]]);outnormal=sign*np.array([-outgoing[1],outgoing[0]])
                selected=np.flatnonzero(abs(progress-center)<=35.)
                decisions=np.flatnonzero(abs(decision_s-center)<=35.)
                rr=rows[selected];ss=progress[selected]-center
                rel=rr[:,:2]-vertex
                heading=np.rad2deg(np.arctan2(rr[:,2:4]@normal,rr[:,2:4]@incoming))
                within=abs(ss)<=10;entry=(ss>=-10)&(ss<=0);exit_mask=(ss>=5)&(ss<=20)
                onset=np.flatnonzero((ss<0)&(heading>=1.))
                requested=actions[decisions]
                corners.append(dict(id=corner['id'],angle_deg=corner['interior_angle_deg'],
                    requested_action_min=requested.min(0).tolist(),requested_action_max=requested.max(0).tolist(),
                    feed_below99_decisions=int((requested[:,1]<99.).sum()),local_decisions=len(decisions),
                    first_inward_heading_ge1deg_s_mm=float(ss[onset[0]]) if len(onset) else None,
                    maximum_entry_inward_offset_mm=float((rel[entry]@normal).max()),
                    maximum_exit_outward_offset_mm=float(np.maximum(-(rel[exit_mask]@outnormal),0).max()) if exit_mask.any() else None,
                    min_speed_window_mm_s=float(speed[selected][within].min()),
                    speed_deficit_window_mm=float(np.maximum(100-speed[selected][within],0).sum()*dt),
                    maximum_abs_curvature_window_per_mm=float(abs(curvature[selected][within]).max())))
                if corner['id'] in selected_ids:
                    checked.update(decisions.tolist());prefix=corner['id']
                    locals_data[prefix+'_physical']=np.column_stack([ss,rr[:,:2],heading,curvature[selected],speed[selected]])
                    locals_data[prefix+'_requests']=np.column_stack([decision_s[decisions]-center,requested])
            ids=sorted(checked);replayed=np.vstack([policy.act(x[i],True)[0] for i in ids])
            error=float(abs(replayed-z[ids]).max());assert error<=1e-6
            output=folder/f'{case}_{key}_representative_rows.npz';np.savez_compressed(output,**locals_data)
            record=dict(case=case,dt_s=dt,trace=str(trace),trace_sha256=sha(trace),
                action_columns=['steering_request_rad','feed_request_mm_s','preview_request_mm'],
                action_min=actions.min(0).tolist(),action_max=actions.max(0).tolist(),
                feed_below99_fraction=float((actions[:,1]<99).mean()),
                checked_policy_decisions=len(ids),maximum_replayed_latent_error=error,
                representatives=selected_ids,corners=corners,representative_rows=str(output),representative_rows_sha256=sha(output))
            records.append(record);bindings[str(trace)]=sha(trace);bindings[str(result_path)]=sha(result_path)
            print('MECHANISM',case,key,'feed_below99_fraction',record['feed_below99_fraction'],
                  'inward_before_corner',sum(c['first_inward_heading_ge1deg_s_mm'] is not None for c in corners),'of',len(corners),'replay_error',error,flush=True)
    write(folder/'audit.json',dict(records=records,source_sha256=bindings,audit_script_sha256=sha(__file__),
        physical_columns=['relative_reference_s_mm','x_mm','y_mm','signed_incoming_heading_deg','signed_curvature_per_mm','speed_mm_s'],
        request_columns=['relative_reference_s_mm','steering_request_rad','feed_request_mm_s','preview_request_mm'],
        representative_selection='Existing geometry mechanism_columns preserved;no selection by observed performance.',
        scope='Descriptive actual-request/execution evidence;not a causal removal experiment or independent seed repetition.',
        interpretation='Near-constant capped feed on most long straights is expected. Three nonconstant channels in dense motion must not be described as all three changing at every long-path corner.',
        goal_complete=False,method_frozen=False))


if __name__=='__main__':main()
