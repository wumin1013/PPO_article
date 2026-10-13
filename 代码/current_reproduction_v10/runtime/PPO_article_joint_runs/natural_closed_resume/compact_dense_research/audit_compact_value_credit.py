"""Offline actual-return audit of the newly trained compact policy."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
stage=ROOT/'ppo14_action3_allow_seed922301_credit30'
files=sorted(stage.glob('batch_*.npz'))
batches=[dict(np.load(path)) for path in files]
fields={key:np.concatenate([b[key] for b in batches]) for key in batches[0]}
terminal=np.flatnonzero(fields['terminal']);records=[];start=0
for end in terminal:
    ids=np.arange(start,end+1);rewards=fields['reward'][ids]
    actual_return=np.cumsum(rewards[::-1])[::-1]
    values=fields['value'][ids];remaining=fields['observation'][ids,-1]
    errors=actual_return-values
    records.append(dict(first_step=int(start),last_step=int(end),decision_count=len(ids),
        full_observed_return=float(actual_return[0]),critic_at_start=float(values[0]),
        value_rmse=float(np.sqrt(np.mean(errors**2))),value_bias=float(errors.mean()),
        remaining_feature_minmax=[float(remaining.min()),float(remaining.max())]))
    start=end+1
value=dict(kind='Offline audit of actual rewards and stored critic values; no rollouts or updates',
    source_stage=str(stage),complete_training_episodes=len(records),records=records,
    caution='Episodes can cross PPO update boundaries; these are realized trajectory returns, not exact fixed-policy expected values.',
    implication='Large systematic critic error motivates testing Monte Carlo critic targets before PPO advantage construction; this alone does not prove a faster learned policy.',
    physics_ticks=0,ppo_updates=0)
(ROOT/'compact_credit30_value_audit.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(records[-8:],indent=2))
