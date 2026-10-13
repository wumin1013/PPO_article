"""Read the prepared early states and audit full-route support before optimization."""
import numpy as np
from joint_teacher_state_policy import state_pair,features
from audit_saved_lateral_motion import lateral_motion
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_early_observable_joint_learning_20260920'
    with np.load(stage/'replayed_states.npz') as z:data={k:z[k].copy() for k in z.files}
    baseline=read(stage/'baseline.json')['baseline']['records']
    a,b=state_pair(data['observations'],data['innovations']);both=np.vstack([a,b])
    mean=both.mean(0);std=np.maximum(both.std(0),.05);a,b=(a-mean)/std,(b-mean)/std
    centers=[];anchors=[]
    for path in (206,304):
        r=next(r for r in baseline if r['path']==path);first=lateral_motion(r['trace'])['intervals'][0]
        decision=max(0,first['start_ms']//20-1)
        index=np.flatnonzero((data['path_ids']==path)&(data['decisions']==decision)).item()
        centers.append(a[index]);anchors.append(dict(path=path,decision=decision,
            mirror_distance=float(np.linalg.norm(a[index]-b[index])),wrong_way_interval=first))
    spec=dict(mean=mean,std=std,centers=np.array(centers));output=[]
    shapes={r['path'] for r in baseline if 'shape' in r}
    for width in (.15,.3,.6):
        spec['radii']=np.full(2,width);f=features(spec,data['observations'],data['innovations'])
        paths=sorted(np.unique(data['path_ids'][np.any(f!=0.,axis=(1,2))]).tolist())
        by_head=[sorted(np.unique(data['path_ids'][np.any(f[:,:,h]!=0.,axis=1)]).tolist()) for h in range(3)]
        by_center=[sorted(np.unique(data['path_ids'][np.any(f[:,i,:]!=0.,axis=1)]).tolist()) for i in range(2)]
        item=dict(width=width,paths=paths,noncorner_paths=sorted(set(paths)-shapes),per_head_paths=by_head,per_center_paths=by_center)
        output.append(item);print('EARLY_SUPPORT_AUDIT',item,flush=True)
    write_json(stage/'support_audit.json',dict(anchors=anchors,width_choices=output,
        actual_new_physical_vectors=0,reason='Preparation exceeded the preregistered isolated-corner support scope; no optimization was started.'))
    print('EARLY_ANCHORS',anchors,flush=True)


if __name__=='__main__':main()
