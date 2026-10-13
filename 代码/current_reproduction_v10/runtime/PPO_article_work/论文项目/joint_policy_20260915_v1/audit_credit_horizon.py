"""Close the registered lambda comparison using receipts and saved diagnostics."""
import json
from pathlib import Path
import numpy as np
from study_io import BASE,register,receipt,write_json,record_progress


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    folders=[BASE/f'credit_horizon_{mode}_pilot_v1' for mode in ('joint','fixed')]
    analyses=[BASE/f'credit_horizon_{n}_analysis_v1' for n in (30000,60000)]
    sources=[Path(__file__),BASE/'credit_horizon_design_v1.md',BASE/'saved_actor_gradient_decomposition_v1/results.json']
    sources += [p/name for p in folders for name in ('registration.json','receipt.json','selection.json','validations.json','training_progress.json','initial_reuse.json')]
    sources += [p/name for p in analyses for name in ('receipt.json','results.json')]
    if not all(p.exists() for p in sources):
        raise RuntimeError('Wait for both completions and both analyses')
    stage=register('credit_horizon_complete_audit_v1',sources,
        dict(model_calls=0,physics_ticks=0,training_decisions=0),
        'Close lambda-one at its original budget, retain all results and decide from matched actual motion.')
    groups=[]
    totals={}
    for folder in folders:
        reg=read(folder/'registration.json')['budget']
        rec=read(folder/'receipt.json')
        progress=read(folder/'training_progress.json')
        vals=read(folder/'validations.json')['validations']
        assert reg['gae_lambda']==1 and reg['observation_dim']==72 and reg['reward_mode']=='original'
        assert reg['seed']==917 and reg['fixed_d']==(3. if reg['mode']=='fixed' else None)
        assert rec['costs']['training_actor_calls']==60000 and rec['wall_s']<=7200
        assert [v['decisions'] for v in vals]==[0,30000,60000]
        candidates=[]
        for v in vals:
            assert sorted(r['path'] for r in v['results'])==list(range(12))
            n=sum(r['success'] for r in v['results'])
            candidates.append(dict(decisions=v['decisions'],tag=v['tag'],successes=n,
                complete_time_sum_s=sum(r['completion_time_s'] for r in v['results']) if n==12 else None))
        best=max(candidates,key=lambda r:(r['successes'],-r['complete_time_sum_s'] if r['complete_time_sum_s'] is not None else -r['decisions']))
        assert best==rec['selected']==read(folder/'selection.json')['selected']
        updates=progress['updates']
        groups.append(dict(mode=reg['mode'],stage=folder.name,selected=best,curves=candidates,
            wall_s=rec['wall_s'],costs=rec['costs'],updates=len(updates),
            early_stops=sum(u['statistics']['actor_early_stop'] for u in updates),
            first_gae_std=updates[0]['statistics']['gae_std'],
            late_gae_std_median=float(np.median([u['statistics']['gae_std'] for u in updates[15:]])),
            completed_training_episodes=len(progress['episodes']),
            training_successes=sum(e['success'] for e in progress['episodes']),
            parameters=rec['parameters']))
        for k,v in rec['costs'].items(): totals[k]=totals.get(k,0)+v
    final=read(analyses[-1]/'results.json')
    result=dict(groups=groups,total_costs=totals,comparisons_60000=final['comparisons'],
        both_modes_support=final['joint_and_fixed_directional_support'],automatic_extension=False,
        state_reward_combination=False,sealed_test_opened=False,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_credit_horizon_budget_selection_audit',model_calls=0,
        physics_ticks=0,training_decisions=0,goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        'λ=1联合/固定两组已按原60k预算完成并审核全部节点、选择、训练成本和失败；未自动追加训练。对照结果与真实梯度诊断需加入新报告和英文稿，强v7及简化负结果保留；最终独立种子、合理选距、长路径/消融/封存评价与20–25页最终全文仍未完成。')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
