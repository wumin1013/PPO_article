"""Read saved evidence structure and local state ranges; no actor or simulation."""
import json
import numpy as np
from study_io import BASE

data=json.loads((BASE/'paper_evidence_review/candidate_audit.json').read_text(encoding='utf-8'))
c=next(c for c in data['candidates'] if c['label']=='worst_guard')
print('RECORD_KEYS',list(c['records'][0]))
print('FIRST_RECORD_POINTERS',{k:v for k,v in c['records'][0].items() if isinstance(v,str)})
for name in ['public_long/trace.npz','remaining_public_long/dense_corners_large.npz','remaining_public_long/closed_line_arc_large.npz']:
    z=np.load(BASE/'paper_evidence_review'/name)
    ref=z['reference'];x=z['observations'];a=z['actions']
    lengths=np.linalg.norm(np.diff(ref,axis=0),axis=1)
    print(name,'lengths',np.round(lengths[:12],3).tolist(),'last_obs',np.round(x[-1,:22],4).tolist(),
          'last_actions',np.round(a[-8:],4).tolist())
for r in c['records'][:5]:
    for key,value in r.items():
        if isinstance(value,str) and value.endswith('.npz'):
            p=BASE/value
            if p.exists():
                z=np.load(p);print('TRAIN_FIELDS',r['path'],value,z.files,{k:z[k].shape for k in z.files})
