"""Freeze new path combinations without executing or selecting a policy."""
import hashlib
import json
import math
import numpy as np
from study_io import BASE,register,receipt,write_json,sha
from mixed_curriculum import polyline,_check


def fingerprint(ref):
    # Translation, rotation and reflection invariance for duplicate detection.
    segments=np.diff(np.asarray(ref,float),axis=0)
    lengths=np.linalg.norm(segments,axis=1)
    tangents=segments/lengths[:,None]
    dot=np.sum(tangents[:-1]*tangents[1:],axis=1)
    cross=tangents[:-1,0]*tangents[1:,1]-tangents[:-1,1]*tangents[1:,0]
    angles=np.arctan2(cross,dot)
    possible=[]
    for sign in [1.,-1.]:
        possible.append(np.r_[lengths,sign*angles].round(8).tobytes())
    return hashlib.sha256(min(possible)).hexdigest()


if __name__=='__main__':
    stage=register('sealed_test_preparation_v1',[__file__,BASE/'sealed_test_protocol_v1.md',
        BASE/'mixed_curriculum.py',BASE/'geometry_spec.py',BASE/'inspect_geometry_semantics.py',
        BASE/'learning_preparation_v4/paths.json'],
        dict(paths=30,classes=3,candidates_per_path_max=128,seed=915271,
             models=0,physics_ticks=0,training_decisions=0),
        'Generate and freeze held-out path combinations before method/checkpoint selection; no policy execution.')
    data=json.loads((BASE/'learning_preparation_v4/paths.json').read_text(encoding='utf-8'))
    seen={fingerprint(ref) for ref in data['training']+data['validation']}
    rng=np.random.default_rng(915271)
    results=[]
    rejected=0
    for family in ['mixed_lengths','dense_corners','closed_notched']:
        for replicate in range(10):
            for attempt in range(128):
                if family=='closed_notched':
                    width,height=rng.uniform(70.,180.,2)
                    cut=rng.uniform(.38,.67)*width
                    low,high=rng.uniform(.25,.36),rng.uniform(.64,.75)
                    ref=np.array([[width/2,0],[width,0],[width,height],[0,height],
                        [0,high*height],[cut,high*height],[cut,low*height],[0,low*height],
                        [0,0],[width/2,0]])
                else:
                    turns=int(rng.integers(12,21))
                    sign=1. if replicate%2==0 else -1.
                    middle=sign*(-1.)**np.arange(turns)*rng.uniform(35.,75.,turns)
                    angles=np.r_[0.,middle,0.]
                    lengths=rng.uniform(8.,32.,len(angles)) if family=='dense_corners' else rng.uniform(20.,180.,len(angles))
                    ref=polyline(np.deg2rad(angles),lengths)
                signature=fingerprint(ref)
                segments=np.diff(ref,axis=0)
                tangents=segments/np.linalg.norm(segments,axis=1)[:,None]
                angles=np.rad2deg(np.arccos(np.clip(np.sum(tangents[:-1]*tangents[1:],axis=1),-1.,1.)))
                if signature in seen or np.any(angles<30.-1e-8) or np.any(angles>170.+1e-8) or not _check(ref):
                    rejected+=1
                    continue
                seen.add(signature)
                results.append(dict(id=f'{family}_{replicate:02d}',family=family,
                    reference=ref.tolist(),closed=family=='closed_notched',
                    duplicate_signature=signature,geometry_attempts=attempt+1))
                break
            else:
                raise RuntimeError('Finite geometric generation budget exhausted')
    write_json(stage/'paths.json',dict(status='sealed_before_model_evaluation',seed=915271,
        cases=results,training_use_prohibited=True,model_evaluations=0))
    receipt(stage,dict(status='test_geometry_sealed',paths=30,
        counts={family:sum(r['family']==family for r in results) for family in ['mixed_lengths','dense_corners','closed_notched']},
        paths_sha256=sha(stage/'paths.json'),rejected_geometry_candidates=rejected,
        models=0,physics_ticks=0,training_decisions=0,model_selection_frozen=False))
    print('Thirty new paths sealed; no policy evaluation or training. SHA256='+sha(stage/'paths.json'))
