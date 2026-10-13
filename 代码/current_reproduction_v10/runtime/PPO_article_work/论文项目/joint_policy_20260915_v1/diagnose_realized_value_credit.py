"""Uncensored realized-return residuals from preserved pre-reboot batches."""
import json
from pathlib import Path
import numpy as np
from joint_policy import advantages
from resume_codec_v5 import load_resume
from study_io import BASE, register, receipt, sha, write_json

GROUPS=('joint_independent_refinement_v7','joint_gsde_refinement_v7',
        'fixed3_independent_refinement_v7','fixed3_gsde_refinement_v7')


def summary(target,value,gae):
    if not len(target):
        return dict(rows=0)
    residual=target-value
    nonzero=(abs(residual)>1e-7)&(abs(gae)>1e-7)
    return dict(rows=len(target),realized_return_residual_mean=float(residual.mean()),
        realized_return_residual_mae=float(np.mean(abs(residual))),
        realized_return_residual_rmse=float(np.sqrt(np.mean(residual**2))),
        realized_return_residual_abs_quantiles=np.quantile(abs(residual),[.5,.9,.99]).tolist(),
        value_target_explained_variance=float(1.-np.var(residual)/np.var(target)) if np.var(target)>1e-14 else None,
        unnormalized_gae_mc_sign_agreement=float(np.mean(np.sign(residual[nonzero])==np.sign(gae[nonzero]))) if nonzero.any() else None,
        gae_std=float(np.std(gae)),realized_advantage_std=float(np.std(residual)))


def main():
    bindings=[BASE/(name+'_reboot_recovery_v3')/'restoration.json' for name in GROUPS]
    stage=register('realized_value_credit_diagnostic_v3',
        [Path(__file__),BASE/'joint_policy.py',BASE/'resume_codec_v5.py',
         BASE/'minimal_information_credit_audit_v1/receipt.json',BASE/'realized_value_credit_diagnostic_v1/failure.json',
         BASE/'realized_value_credit_diagnostic_v2/failure.json',*bindings],
        dict(preserved_checkpoint_deserializations=4,model_forwards=0,optimizer_updates=0,physics_ticks=0),
        'Compare saved sampling-time values and lambda-.98 credit with realized completed suffix returns; distinguish critic uncertainty from reward formula complexity.')
    records=[]
    for name,binding in zip(GROUPS,bindings):
        saved=json.loads(binding.read_text(encoding='utf-8'))
        snapshot=Path(saved['snapshot'])
        assert sha(snapshot)==saved['snapshot_sha256']
        state=load_resume(snapshot)
        batch=state['batch']
        rewards=np.asarray(batch['reward'],float)
        values=np.asarray(batch['value'],float)
        next_values=np.asarray(batch['next_value'],float)
        terminals=np.asarray(batch['terminal'],bool)
        ends=np.flatnonzero(terminals)
        episodes=state['episodes'][-len(ends):] if len(ends) else []
        gae,_=advantages(rewards,values,next_values,terminals)
        lam1,_=advantages(rewards,values,next_values,terminals,lam=1.)
        targets=np.full(len(rewards),np.nan)
        successful=np.zeros(len(rewards),bool)
        distances=np.zeros(len(rewards),int)
        fragments=[]
        start=0
        for end,episode in zip(ends,episodes):
            idx=slice(start,end+1)
            targets[idx]=np.cumsum(rewards[idx][::-1])[::-1]
            successful[idx]=episode['success']
            distances[idx]=np.arange(end-start,-1,-1)
            fragments.append(dict(episode=episode['episode'],path=episode['path'],success=episode['success'],
                reason=episode['reason'],batch_start_may_be_episode_suffix=bool(start==0),
                **summary(targets[idx],values[idx],gae[idx])))
            start=end+1
        included=np.isfinite(targets)
        error=float(np.max(abs(lam1[included]-(targets[included]-values[included])))) if included.any() else None
        if error is not None:
            assert error<1e-5
        nonterminal_pairs=~terminals[:-1]
        continuity=float(np.max(abs(next_values[:-1][nonterminal_pairs]-values[1:][nonterminal_pairs]))) if nonterminal_pairs.any() else 0.
        groups={}
        for group,mask in [('all_completed',included),('success',included&successful),
            ('failure',included&~successful),('success_at_least_50_steps_before_terminal',included&successful&(distances>=50)),
            ('success_last_50_steps',included&successful&(distances<50))]:
            groups[group]=summary(targets[mask],values[mask],gae[mask])
        records.append(dict(stage=name,decision=state['decision'],batch_rows=len(rewards),
            censored_tail_rows=int((~included).sum()),completed_fragments=len(fragments),
            lambda1_realized_identity_max_error=error,saved_value_continuity_max_error=continuity,
            groups=groups,fragments=fragments,snapshot_sha256=saved['snapshot_sha256']))
    caveat=('A realized return includes future action noise and is not the true conditional expected value. '
        'Residuals are not pure critic approximation errors. Completed-success conditioning excludes failures '
        'and censored tails; rows within each trajectory are dependent. Raw GAE/MC signs precede actor batch '
        'centering. These diagnostics cannot establish that lambda=1 or another critic would improve control.')
    write_json(stage/'results.json',dict(records=records,interpretation_limits=caveat,goal_complete=False))
    lines=['# 保存价值与成功后续回报的差距','',
        '| 保存批 | 成功行数 | 成功残差MAE/s | 成功残差RMSE/s | GAE与完整后续残差信号同号比例 |',
        '|---|---:|---:|---:|---:|']
    for row in records:
        r=row['groups']['success']
        lines.append(f"| {row['stage']} | {r['rows']} | {r['realized_return_residual_mae']:.4f} | {r['realized_return_residual_rmse']:.4f} | {r['unnormalized_gae_mc_sign_agreement']:.3f} |")
    lines += ['', '使用系统重启前已冻结的四份批，不是当前v7末期总体统计。保留批首回合后缀，排除尚未结束批尾。λ=1的代数核对与真实已结束后缀回报一致，没有模型前向、新物理或参数更新。', '',
        '上表的实际回报包含后续动作随机性，残差不等于真实条件价值函数的拟合误差；只在成功后缀上统计还包含结局条件化选择。GAE与Monte Carlo残差同号比例在批中心化之前计算，不能解释成错误梯度比例，也不证明改用λ=1更好。它帮助定位仍应研究价值拟合与长时信用，而不将问题直接归因于奖励项数。']
    (stage/'实际价值残差.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_uncensored_realized_credit_diagnostic',
        checkpoint_deserializations=4,model_forwards=0,physics_ticks=0,training_updates=0))
    print('\n'.join(lines))


if __name__=='__main__':
    main()
