"""Read-only action/support diagnosis at failed independent rollout states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from datetime import datetime,timezone
import numpy as np
import torch
from evaluate_historical_dense import ROOT,BASE,write,sha
from replay_lowdim_training_stage import SUPPLEMENT,SEEDS
from research_policy_loader import from_saved


def main():
    torch.set_num_threads(1)
    frozen_path=ROOT/'direct14_dictionary02_resume_round73/checkpoint_0000.pt'
    frozen=from_saved(torch.load(frozen_path,map_location='cpu',weights_only=False))
    records=[]
    for seed in SEEDS:
        folder=SUPPLEMENT/'independent_training'/f'seed_{seed}'
        checkpoint=folder/'local_features/checkpoint_0004.pt'
        policy=from_saved(torch.load(checkpoint,map_location='cpu',weights_only=False),seed)
        for case in ['original','dense']:
            geometry=BASE/'paper_evidence_review/natural_closed_case/geometry.json' if case=='original' else ROOT/'dense_closed_case/geometry.json'
            corners=json.loads(geometry.read_text(encoding='utf-8'))['sharp_corners']
            for key in ['1ms','half_ms']:
                path=folder/'final_evaluation'/f'eval_0004_{case}_{key}.json'
                result=json.loads(path.read_text(encoding='utf-8'))
                if result['success']:continue
                trace=path.with_suffix('.npz')
                with np.load(trace) as data:
                    x=data['observations'][-1].astype(np.float32)
                    rows=data['rows']
                    actual=data['latents'][-1]
                    ref=data['reference']
                z=policy.act(x,True)[0]
                assert np.array_equal(z,actual)
                zz=frozen.act(x,True)[0]
                with torch.no_grad():
                    f=policy.actor.features(torch.tensor(x)).numpy()
                    local=policy.actor.backbone.features(torch.tensor(x)).numpy()
                nearest=int(np.linalg.norm(ref-rows[-1,:2],axis=1).argmin())
                cumulative=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ref,axis=0),axis=1))]
                near_corner=min(corners,key=lambda c:abs(c['reference_s_mm']-cumulative[nearest]))
                records.append(dict(seed=seed,case=case,dt_s=result['dt_s'],reason=result['reason'],
                    prefix_s=result['elapsed_prefix_s'],nearest_reference_s_mm=float(cumulative[nearest]),
                    nearest_corner=near_corner['id'],active_joint_features=int((f>0).sum()),
                    active_preview_features=int((local>0).sum()),
                    own_raw_action=z.tolist(),frozen_actor_query_same_state=zz.tolist(),
                    clipped_request_difference=(np.clip(z,-1,1)-np.clip(zz,-1,1)).tolist(),
                    own_checkpoint_sha256=sha(checkpoint),trace=str(trace),trace_sha256=sha(trace),
                    scope='Pointwise counterfactual actor query at witnessed failed-state input; not a successful frozen-policy rollout from this arrival state.'))
    write(SUPPLEMENT/'failed_seed_state_diagnosis.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_checkpoint_sha256=sha(frozen_path),records=records,new_physics_steps=0,new_training_updates=0))
    print('FAILURE_DIAGNOSIS',[(r['seed'],r['case'],r['dt_s'],r['nearest_corner'],r['active_joint_features']) for r in records])


if __name__=='__main__':main()
