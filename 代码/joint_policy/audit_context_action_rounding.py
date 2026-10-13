"""Measure actual action quantization in recent joint credit corrections; no rollout."""
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from observable_memory_joint_policy import dynamic_context
from joint_teacher_state_policy import features,OUTPUT_SCALE
from study_io import BASE,write_json
from search_complete_return import read


def corrections(md,replay,coefficients):
    f=features(md,replay['observations'],replay['innovations'])
    context=np.tanh((dynamic_context(replay['observations'])-md['dynamic_context_mean'])/md['dynamic_context_scale'])
    f=np.concatenate([f]+[f*context[:,k,None,None] for k in range(2)],axis=1)
    count=f.shape[1];coef=np.asarray(coefficients).reshape(-1,3);a=np.asarray(md['memory_decays'])[:,None,None]
    values=[];last_path=None
    for path,fi in zip(replay['path_ids'],f):
        if path!=last_path:memory=np.zeros((2,count,3));previous=np.zeros_like(fi);last_path=path
        memory=a*memory+(1.-a)*previous;previous=fi.copy()
        correction=(fi*coef[:count]).sum(axis=-2)*OUTPUT_SCALE
        for index,m in enumerate(memory):
            block=coef[(index+1)*count:(index+2)*count]
            if np.any(block):correction+=(m*block).sum(axis=-2)*OUTPUT_SCALE
        values.append(correction)
    return np.asarray(values)


def main():
    stage=BASE/'canonical_gate_band_zero_margin_context_credit_20260920'
    history=read(stage/'state_joint/results.json')['candidates']
    blocks=read(stage/'optimization_blocks.json');records=[]
    md=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    indexed={c['tag']:c for c in history}
    for block in blocks:
        anchor,proposed=indexed[block['anchor']],indexed[block['tag']]
        md['coefficients']=np.array(anchor['coefficients']).reshape(-1,3)
        replay=collect_data(md,md['parent'],anchor['records'],include_latents=True)
        original=replay['parent_latents'];current=original.astype(float)+corrections(md,replay,anchor['coefficients'])
        assert np.array_equal(current.astype(original.dtype),replay['current_latents'])
        next_action=original.astype(float)+corrections(md,replay,proposed['coefficients'])
        rounded_current=current.astype(original.dtype);rounded_next=next_action.astype(original.dtype)
        # Report active recovery decisions separately from near-zero late memory tails.
        window=(replay['decisions']>=18)&(replay['decisions']<=30)
        difference=next_action-current;intended=(abs(difference)>1e-12)&window[:,None]
        swallowed=intended&(rounded_current==rounded_next)
        ulp=abs(np.spacing(rounded_current)).astype(float)
        records.append(dict(anchor=anchor['tag'],proposed=proposed['tag'],action_dtype=str(original.dtype),
            exact_saved_action_reproduction=True,decisions=len(original),
            recovery_window_ms=[360,600],intended_changed_components=int(intended.sum()),
            rounded_away_components=int(swallowed.sum()),per_head_intended=intended.sum(0).tolist(),
            per_head_rounded_away=swallowed.sum(0).tolist(),
            max_intended_latent_delta=float(abs(difference[window]).max()),
            median_update_to_ulp=float(np.median(abs(difference[intended])/ulp[intended])) if intended.any() else None))
    write_json(stage/'action_rounding_audit.json',dict(records=records,new_physical_rollouts=0,
        interpretation='Counterfactual actions on the same saved states/history, not new '
        'physical outcomes. Exact current actions are verified before examining rounding. '
        'A dtype effect on successful shape recovery requires matched full-loop testing.'))
    for r in records:print('ACTION_ROUNDING',r,flush=True)


if __name__=='__main__':main()
