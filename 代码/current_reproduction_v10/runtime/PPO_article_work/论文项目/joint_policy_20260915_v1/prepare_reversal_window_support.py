"""Move the new learning support to the observed onset of residual lateral reversal."""
import copy
import argparse
import time
import numpy as np
import torch
from study_io import BASE,write_json,sha
from search_complete_return import read
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features,state_pair


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--active-window',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter()
    source=BASE/('canonical_gate_band_reversal_window_local_recovery_20260920' if args.active_window else
                'canonical_gate_band_early_geometry_credit_recovery_20260920')
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)==('132e3914433db31aa72c0cf35de80b6d0b010b1a49f82d8ca30ab79776b6f8dd' if args.active_window else
                              '7605e4dd430d529562a430696ea0410d24c0fce839975174ffece414e89febb8')
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    template=torch.load(BASE/'canonical_gate_band_direct_stable_joint_learning_20260920/earlier_geometry_updated_initial_policy.pt',
                        map_location='cpu',weights_only=False)
    with np.load(BASE/'canonical_gate_band_direct_stable_joint_learning_20260920/earlier_geometry_updated_training_states.npz') as z:
        training={k:z[k].copy() for k in z.files}
    records=read(source/'results.json')['candidates'][0]['records']
    # Innovation features are relative to the new outer parent, not its ancestor.
    current=collect_data(parent,parent,[r for r in records if r['path'] in [15,104,207,307]],
                         deadline=started+60.)
    keep=~np.isin(training['path_ids'],np.unique(current['path_ids']))
    training={k:np.concatenate([training[k][keep],current[k]]) for k in training}
    audit=read(source/'saved_lateral_motion_audit.json')
    lateral=next(r['after'] for r in audit['records'] if r['path']==104)
    onset=lateral['intervals'][0]['start_ms'];early=onset//20
    selectors=[early,15,16]
    md=copy.deepcopy(template);md['parent']=parent
    if args.active_window:
        record=next(r for r in records if r['path']==104)
        with np.load(record['trace']) as z:ref,rows=z['reference'],z['rows']
        u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u)
        normal=np.sign(u[0]*v[1]-u[1]*v[0])*np.array([-u[1],u[0]])
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        signed_velocity=rows[:lateral['apex_ms'],2:4]@normal
        peak=int(np.argmin(np.where(speed[:len(signed_velocity)]>5.,signed_velocity,np.inf)))//20
        assert peak>early
        selectors=[early,peak,15,16];indices=[0,0,1,2]
        for key in ('per_center_mean','per_center_std','radii','per_center_distance_weights'):
            md[key]=np.asarray(md[key])[indices].copy()
    raw,_=state_pair(training['observations'],training['innovations'])
    md['centers']=np.array([(raw[np.flatnonzero((training['path_ids']==104)&(training['decisions']==d)).item()]-
        md['per_center_mean'][k])/md['per_center_std'][k] for k,d in enumerate(selectors)])
    md['coefficients']=np.zeros((3*len(selectors),3));md['learning_method']='pre_reversal_supported_joint_complete_return_learning'
    f=features(md,training['observations'],training['innovations'])
    support=[]
    for k in range(len(selectors)):
        active=np.any(f[:,k]!=0.,axis=1)
        support.append(dict(kernel=k,training_selector_ms=selectors[k]*20,
            activations=[dict(path=int(p),decision_ms=int(d)*20,feature=float(abs(v).max()))
                for p,d,v in zip(training['path_ids'][active],training['decisions'][active],f[active,k])]))
    paths=sorted(set(a['path'] for k in support for a in k['activations']))
    selected=[r for r in records if r['path'] in paths]
    assert all('shape' in r for r in selected)
    replay=collect_data(md,parent,selected,include_latents=True,deadline=started+60.)
    replay['held_actions']=np.vstack([np.load(r['trace'])['actions'] for r in selected])
    stem='active_window' if args.active_window else 'reversal_window_exact'
    torch.save(md,source/(stem+'_initial_policy.pt'))
    np.savez_compressed(source/(stem+'_training_states.npz'),**training)
    np.savez_compressed(source/(stem+'_affected_replay.npz'),**replay)
    out=dict(records=support,affected_paths=paths,training_paths=len(records),
        source_model_sha256=sha(parent_file),saved_significant_lateral_onset_ms=onset,
        zero_outer_policy_decoded_actions_bitwise=True,new_physical_runs=0,
        wall_s=time.perf_counter()-started,interpretation='Residual lateral motion persists after the100ms '
        'kernel. Select the last visible training state before that measured event; learn all three '
        'actions jointly with the late geometry support. Time/path select observations offline only; '
        'neither is an actor input. Parent, shape guards and physical executor remain protected.')
    out['parent_relative_innovations_recomputed']=True
    if not args.active_window:out['supersedes']='reversal_window_support_audit.json (ancestor-relative innovations; no learning run)'
    write_json(source/(stem+'_support_audit.json'),out)
    print('REVERSAL_WINDOW_SUPPORT',out,flush=True)


if __name__=='__main__':main()
