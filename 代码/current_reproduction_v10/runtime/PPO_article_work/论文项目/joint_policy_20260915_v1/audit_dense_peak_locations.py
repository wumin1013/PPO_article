"""Locate measured peak errors without rerunning a physical trajectory."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from dense_trace_credit_model import coordinates
from learn_trace_resolved_joint_credit import signed_channels
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    stage=BASE/'canonical_gate_band_error_buffered_trace_completion_20260919'
    model=torch.load(stage/'response_model.pt',map_location='cpu',weights_only=False)
    source=BASE/'canonical_gate_band_dense_signed_trace_completion_20260919'
    candidate=read(source/'results.json')['candidates'][0]
    md=torch.load(source/(candidate['tag']+'.pt'),map_location='cpu',weights_only=False)
    z=coordinates(model,md)
    records=[]
    for path in (8,14):
        record=next(r for r in candidate['records'] if r['path']==path)
        with np.load(source/record['trace']) as data:
            ref,rows,actions=[data[k].copy() for k in ('reference','rows','actions')]
        m=model['models'][path]
        actual=signed_channels(ref,rows,m['base']['grid'],np.arange(len(m['base']['incoming'])))
        predicted=m['base']['turn']+m['jac']['turn']@z
        s=m['base']['grid']
        actual_index=int(abs(actual['turn']).argmax())
        predicted_index=int(abs(predicted).argmax())
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(np.vstack([ref[0],rows[:,:2]]),axis=0),axis=1))]
        tick=int(np.searchsorted(arc,s[actual_index]))-1
        i=max(0,min(tick//20,len(actions)-1))
        around=[]
        for decision in range(max(0,i-2),min(len(actions),i+3)):
            lo,hi=decision*20,min(len(rows),(decision+1)*20)
            speed=np.linalg.norm(rows[lo:hi,2:4],axis=1)
            signed_omega=np.sign(m['base']['angle'])*(rows[lo:hi,2]*rows[lo:hi,5]-rows[lo:hi,3]*rows[lo:hi,4])/np.maximum(speed,1e-9)**2
            around.append(dict(start_ms=lo,held_actions=actions[decision].tolist(),
                speed_min_mm_s=float(speed.min()),speed_max_mm_s=float(speed.max()),
                signed_omega_min_rad_s=float(signed_omega.min()),signed_omega_max_rad_s=float(signed_omega.max())))
        item=dict(path=path,trace_sha256=sha(source/record['trace']),
            actual_peak_curvature_mm_inv=float(abs(actual['turn'][actual_index])/.1),
            predicted_peak_curvature_mm_inv=float(abs(predicted[predicted_index])/.1),
            peak_location_actual_mm=float(s[actual_index]),peak_location_predicted_mm=float(s[predicted_index]),
            approximate_peak_time_ms=tick+1,
            predicted_curvature_at_actual_peak_mm_inv=float(abs(predicted[actual_index])/.1),
            actual_curvature_at_predicted_peak_mm_inv=float(abs(actual['turn'][predicted_index])/.1),
            signed_turn_rmse_rad=float(np.sqrt(np.mean((actual['turn']-predicted)**2))),
            around=around)
        records.append(item)
        print('PEAK_LOCALIZATION',item,flush=True)
    write_json(stage/'peak_locations.json',dict(code_sha256=sha(__file__),
        model_sha256=sha(stage/'response_model.pt'),candidate_sha256=sha(source/(candidate['tag']+'.pt')),
        records=records,new_physical_rollouts=0))


if __name__=='__main__':main()
