"""Numerical checks for independent rollout segments; no motion simulation."""
import json
import numpy as np
import torch
from evaluate_historical_dense import ROOT, BASE, verify_bound_inputs, sha, write
from joint_policy import advantages
from learning_update_compact_mc import segmented_advantages


def verify():
    r=np.array([1.,2.,3.,4.]);v=np.array([.5,1.5,2.5,3.5])
    n=np.array([1.5,0.,3.5,5.]);t=np.array([False,True,False,False])
    end=np.array([False,True,False,True])
    a,ret=segmented_advantages(r,v,n,t,end,lam=1.)
    np.testing.assert_allclose(ret,[3.,2.,12.,9.])
    for lam in [1.,.98]:
        combined=segmented_advantages(r,v,n,t,end,lam=lam)
        parts=[advantages(r[s],v[s],n[s],t[s],lam=lam) for s in [slice(0,2),slice(2,4)]]
        for j in range(2):np.testing.assert_allclose(combined[j],np.r_[parts[0][j],parts[1][j]])
        for j in range(2):
            np.testing.assert_allclose(segmented_advantages(r,v,n,t,[False]*4,lam=lam)[j],advantages(r,v,n,t,lam=lam)[j])
    # Both boundaries unfinished: bootstrap each separately; no cross-case return.
    t=np.zeros(4,dtype=bool);n=np.array([1.5,7.,3.5,5.])
    _,ret=segmented_advantages(r,v,n,t,end,lam=1.)
    np.testing.assert_allclose(ret,[10.,9.,12.,9.])
    changed=r.copy();changed[2:]+=1000.
    np.testing.assert_allclose(segmented_advantages(changed,v,n,t,end,lam=1.)[1][:2],ret[:2])
    manifest=verify_bound_inputs()
    files=[ROOT/'learning_update_compact_mc.py',ROOT/'learning_update_guarded_mc.py',__file__]
    for p in files:
        from pathlib import Path
        raw=Path(p).read_bytes();assert not raw.startswith(b'\xef\xbb\xbf')
        compile(raw.decode('utf-8'),str(p),'exec')
    return dict(passed=True,checks=['Exact terminal returns','Unfinished segment bootstrap',
        'Independent case isolation','Matches separate single-case GAE','Matches original unsegmented GAE',
        'UTF8 without BOM and syntax','Historical source/model/reference bindings'],
        verified_historical_source_count=len(manifest['imported_source_files']),
        source_sha256={str(p):sha(p) for p in files},simulations=0)


if __name__=='__main__':
    print(json.dumps(verify(),ensure_ascii=False))
