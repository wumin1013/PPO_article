"""Read the unchanged-policy prefixes to distinguish reward and feature effects."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from study_io import BASE,register,receipt,write_json,record_progress
from joint_policy import advantages


def main():
    folders={(arm,mode):BASE/f'minimal_{arm}_{mode}_pilot_v1'
             for arm in ('base','state','reward') for mode in ('joint','fixed')}
    sources=[Path(__file__),BASE/'joint_policy.py',BASE/'minimal_factorial_design_v1.md',
             BASE/'spatial_learning_candidate.py',BASE/'train_minimal_factors.py']
    sources += [p/name for p in folders.values() for name in ('registration.json','receipt.json','storage.json')]
    if not all(p.exists() for p in sources):
        raise RuntimeError('Use closed journals from all six completed runs')
    stage=register('minimal_first_batch_credit_v1',sources,
        dict(journal_prefixes=6,rows_per_prefix=2048,model_loads=0,model_calls=0,physics_ticks=0,training_decisions=0),
        'Inspect identical pre-update physical prefixes and zero-value GAE; distinguish representation from reward effects without new learning.')
    groups=[]
    cached={}
    for (arm,mode),folder in folders.items():
        storage=json.loads((folder/'storage.json').read_text(encoding='utf-8'))
        journal=Path(storage['decision_journal'])
        lines=[]
        with gzip.open(journal,'rb') as stream:
            for _ in range(2048):
                line=stream.readline()
                if not line: raise RuntimeError('Incomplete first batch')
                lines.append(line)
        rows=[json.loads(line) for line in lines]
        assert [r['decision'] for r in rows]==list(range(1,2049))
        assert all(r['value']==r['next_value']==0 for r in rows)
        adv,_=advantages([r['reward'] for r in rows],[r['value'] for r in rows],
            [r['next_value'] for r in rows],[r['terminal'] for r in rows])
        scaled=(adv-adv.mean())/(adv.std()+1e-8)
        feed_residual=(np.array([r['latent'][1] for r in rows])-np.float32(math.atanh(-.4)))/.12
        nonterminal=np.array([not r['terminal'] for r in rows])
        rewards=np.array([r['reward'] for r in rows])
        terminals=[r for r in rows if r['terminal']]
        groups.append(dict(arm=arm,mode=mode,journal=str(journal),prefix_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),
            rows=len(rows),terminal_rows=len(terminals),
            terminal_reasons={reason:sum(r['reason']==reason for r in terminals) for reason in sorted({r['reason'] for r in terminals})},
            nonterminal_reward_quantiles=np.quantile(rewards[nonterminal],[.1,.5,.9]).tolist(),
            gae_mean=float(adv.mean()),gae_std=float(adv.std()),
            normalized_gae_feed_residual_mean_product=float(np.mean(scaled*feed_residual)),
            zero_sampling_values=True))
        cached[arm,mode]=(rows,adv)
    pairs=[]
    physical_keys=('decision','episode','path','latent','held','elapsed_s','terminal','reason','failure_remaining_penalty_s')
    for mode in ('joint','fixed'):
        base,base_adv=cached['base',mode]
        for arm in ('state','reward'):
            changed,changed_adv=cached[arm,mode]
            physical_identical=all(all(a[key]==b[key] for key in physical_keys) for a,b in zip(base,changed))
            pairs.append(dict(mode=mode,changed_arm=arm,physical_journal_fields_identical=physical_identical,
                reward_max_difference=max(abs(a['reward']-b['reward']) for a,b in zip(base,changed)),
                gae_max_difference=float(np.max(np.abs(base_adv-changed_adv)))))
    result=dict(groups=groups,pairs=pairs,rows_read=12288,model_loads=0,model_calls=0,
        physical_key_scope=list(physical_keys),
        interpretation='First unchanged-policy batch only. Matching action/outcome log fields does not replace bitwise XY trajectory comparison. The GAE/noise product is a sample score-direction proxy before neural Jacobians, clipping and Adam; it is not an actual policy gradient, independent statistical test or proof of later performance. State-only policies can have identical initial actions and advantages but different random hidden features.',
        sealed_test_opened=False,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_first_unchanged_policy_prefix_analysis',journal_prefixes=6,rows_read=12288,
        model_loads=0,model_calls=0,physics_ticks=0,training_decisions=0,goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        '六组先导首批2048条保存日志的只读信用诊断已完成；首次更新前的价值、动作与结局字段、奖励和GAE分别核对。样本GAE与进给噪声乘积不等于实际梯度或因果性能证据。按完整60k单因素结果及形状选择后继，不自动组合或扩训练；封存测试和最终论文验收仍未完成。')
    print(json.dumps(dict(groups=groups,pairs=pairs),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
