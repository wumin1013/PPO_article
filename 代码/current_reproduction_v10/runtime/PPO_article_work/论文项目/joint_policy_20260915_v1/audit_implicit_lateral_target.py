"""Audit the actual steering/lookahead coupling on incoming straight segments."""
import time
import argparse
import numpy as np
from audit_current_command_coordinates import command_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',default='canonical_gate_band_physical_cruise_small_learning_20260917')
    parser.add_argument('--commands',default='canonical_gate_band_physical_cruise_small_command_audit_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_implicit_lateral_target_audit_20260917')
    args=parser.parse_args()
    parent=BASE/args.parent
    data=read(parent/'results.json');selected=data['full_candidate'];model=parent/data['chosen']['model']
    commands=BASE/args.commands
    assert read(commands/'results.json')['parent_sha256']==sha(model)
    stage=register(args.stage,
        [__file__,model,parent/'results.json',parent/'receipt.json',commands/'results.json',
         commands/'receipt.json',BASE/'audit_current_command_coordinates.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,training_corners=16),
        'On the incoming straight only, verify LOS=atan2(-y,d) against the recorded filter '
        'inversion, then inspect the implicit lateral equilibrium q=d*tan(delta). '
        'Here d is the actual forward gap to the preview point, including clipped progress. '
        'Negative combined bearing can arise with positive steering when y>q. Separate '
        'q changes caused by steering and preview at decision boundaries using an exact '
        'symmetric two-factor decomposition. This is a saved-data mechanistic audit, '
        'not a trajectory template, new reward, causal proof of training origin or policy change.')
    start=time.perf_counter();records=[]
    for j,r in enumerate(selected['results']):
        path=r['original_path']
        if path>=16:continue
        with np.load(BASE/selected['traces'][j]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        u,v=np.diff(ref,axis=0);length=np.linalg.norm(u);u/=length;v/=np.linalg.norm(v)
        sign=np.sign(u[0]*v[1]-u[1]*v[0]);inward=sign*np.array([-u[1],u[0]])
        position=np.vstack([ref[0],rows[:-1,:2]])
        along=(position-ref[0])@u;y=(position-ref[0])@inward
        held=actions[np.minimum(np.arange(len(rows))//20,len(actions)-1)]
        forward=np.clip(along,0.,length)+held[:,2]-along
        delta=sign*held[:,0];q=forward*np.tan(delta)
        signals=command_signals(ref,rows,actions)
        expected=np.rad2deg(np.arctan2(-y,forward))
        straight=(np.clip(along,0.,length)+held[:,2]<length-.1)&(rows[:,12]==0)&(forward>0.)
        exact=abs(expected-signals['los_deg'])<1e-7
        valid=straight&exact
        negative=valid&(signals['combined_deg']<-.05)
        assert np.all(y[negative]>q[negative])
        onsets=np.flatnonzero(negative&~np.r_[False,negative[:-1]])
        events=[]
        for tick in onsets:
            decision=int(tick//20);anchor=decision*20
            if decision==0 or not valid[anchor] or not valid[anchor-1]:continue
            a,b=actions[decision-1],actions[decision]
            ta,tb=np.tan(sign*a[0]),np.tan(sign*b[0])
            old=a[2]*ta;new=b[2]*tb
            steering=.5*(a[2]+b[2])*(tb-ta)
            preview=.5*(ta+tb)*(b[2]-a[2])
            assert abs(steering+preview-(new-old))<1e-12
            events.append(dict(tick=int(tick),decision=decision,onset_at_decision_start=bool(tick==anchor),
                old_q_mm=float(old),new_q_mm=float(new),q_change_mm=float(new-old),
                steering_contribution_mm=float(steering),preview_contribution_mm=float(preview),
                y_at_decision_start_mm=float(y[anchor]),old_steering_deg=float(np.rad2deg(sign*a[0])),
                new_steering_deg=float(np.rad2deg(sign*b[0])),old_preview_mm=float(a[2]),new_preview_mm=float(b[2])))
        records.append(dict(path=path,verified_straight_ticks=int(valid.sum()),
            excluded_straight_formula_mismatch_ticks=int((straight&~exact).sum()),
            negative_target_ms=int(negative.sum()),events=events))
        np.savez_compressed(stage/f'path{path}.npz',q_mm=q,y_mm=y,valid=valid,
            combined_heading_deg=signals['combined_deg'],forward_gap_mm=forward,actions=actions)
        print('IMPLICIT_TARGET',path,'negative_ms',negative.sum(),'events',events,flush=True)
    write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,
        interpretation='Exact incoming-straight action/guide coupling; learning-origin causality not inferred.',goal_complete=False))
    receipt(stage,dict(status='completed_implicit_lateral_target_coupling_audit',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
