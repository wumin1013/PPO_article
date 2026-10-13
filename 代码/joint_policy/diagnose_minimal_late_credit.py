"""Audit late on-policy suffixes; no fitted model or new physical trajectory."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from joint_policy import advantages
from study_io import BASE, register, receipt, write_json, record_progress


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def moments(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, rmse=None, median=None, q90=None)
    return dict(n=len(a), mean=float(a.mean()), rmse=float(np.sqrt(np.mean(a*a))),
                median=float(np.median(a)), q90=float(np.quantile(a, .9)))


def main():
    folders = [BASE/f'minimal_{arm}_{mode}_pilot_v1'
               for arm in ('base', 'state', 'reward') for mode in ('joint', 'fixed')]
    sources = [Path(__file__), BASE/'joint_policy.py', BASE/'learning_update_v4.py',
               BASE/'train_minimal_factors.py', BASE/'minimal_factorial_complete_audit_v1/receipt.json']
    sources += [p/name for p in folders for name in ('receipt.json', 'storage.json', 'training_progress.json')]
    stage = register('minimal_late_credit_audit_v1', sources,
        dict(closed_journals=6, rows_read_upper=360000, late_rows_analyzed=180000,
             model_loads=0, model_calls=0, optimization_steps=0, physics_ticks=0),
        'Inspect every last-half pilot batch for realized suffix residuals and terminal-credit attenuation; no lambda selection from performance or extra training.')
    groups = []
    for folder in folders:
        progress = read(folder/'training_progress.json')
        path = Path(read(folder/'storage.json')['decision_journal'])
        digest = hashlib.sha256()
        rows = []
        with gzip.open(path, 'rb') as stream:
            for i in range(60000):
                raw = stream.readline()
                if not raw:
                    raise RuntimeError('Unexpected short closed journal')
                digest.update(raw)
                if i >= 30000:
                    row = json.loads(raw)
                    assert row['decision'] == i+1
                    rows.append(row)
            assert not stream.readline()
        boundaries = [u for u in progress['updates'] if u['decisions'] > 30000]
        batches, residuals, absolute, gae_difference, ages, seconds, weights = [], [], [], [], [], [], []
        successful_rows, failed_rows, censored_rows, suffixes, head_suffixes = 0, 0, 0, 0, 0
        sign_n = sign_equal = 0
        lower = 30000
        for u in boundaries:
            upper = u['decisions']
            batch = rows[lower-30000:upper-30000]
            gae, _ = advantages([r['reward'] for r in batch], [r['value'] for r in batch],
                [r['next_value'] for r in batch], [r['terminal'] for r in batch])
            mc, _ = advantages([r['reward'] for r in batch], [r['value'] for r in batch],
                [r['next_value'] for r in batch], [r['terminal'] for r in batch], lam=1.)
            assert np.isclose(gae.std(), u['statistics']['gae_std'], rtol=1e-6, atol=1e-7)
            selected = []
            start = 0
            for end, row in enumerate(batch):
                if not row['terminal']:
                    continue
                part = batch[start:end+1]
                assert len({r['episode'] for r in part}) == 1
                suffixes += 1
                head_suffixes += int(start == 0)
                total, remaining_s = 0., 0.
                for k in range(end, start-1, -1):
                    total += batch[k]['reward']
                    remaining_s += batch[k]['elapsed_s']
                    residual = total-batch[k]['value']
                    assert np.isclose(mc[k], residual, rtol=2e-5, atol=2e-5)
                    if row['reason'] == 'success':
                        residuals.append(residual)
                        absolute.append(abs(residual))
                        gae_difference.append(float(gae[k])-residual)
                        age = end-k
                        ages.append(age)
                        seconds.append(remaining_s)
                        weights.append(.98**age)
                        successful_rows += 1
                        selected.append(k)
                        sign_n += 1
                        sign_equal += int(np.sign(gae[k]) == np.sign(residual))
                    else:
                        failed_rows += 1
                start = end+1
            censored_rows += len(batch)-start
            idx = np.asarray(selected, dtype=int)
            if len(idx):
                x, y = gae[idx].astype(float), mc[idx].astype(float)
                correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
                centered_agreement = float(np.mean(np.sign(x-x.mean()) == np.sign(y-y.mean())))
            else:
                correlation = centered_agreement = None
            batches.append(dict(first_decision=lower+1, last_decision=upper, rows=len(batch),
                success_suffix_rows=len(idx), censored_tail_rows=len(batch)-start,
                gae_std=float(gae.std()), fitted_value_loss=u['statistics']['value_loss'],
                gae_mc_success_correlation=correlation,
                subset_centered_sign_agreement=centered_agreement))
            lower = upper
        assert successful_rows+failed_rows+censored_rows == 30000
        groups.append(dict(stage=folder.name, journal=str(path), journal_uncompressed_sha256=digest.hexdigest(),
            parsed_rows=30000, decompressed_rows=60000, completed_suffixes=suffixes,
            batch_head_suffixes_upper_bound=head_suffixes, success_rows=successful_rows,
            failure_rows=failed_rows, censored_tail_rows=censored_rows,
            success_realized_residual=moments(residuals), success_absolute_residual=moments(absolute),
            gae_minus_success_mc_residual=moments(gae_difference),
            success_unscaled_sign_agreement=sign_equal/sign_n if sign_n else None,
            decisions_before_terminal=moments(ages), physical_time_to_terminal_s=moments(seconds),
            terminal_delta_direct_weight=moments(weights),
            fraction_success_rows_terminal_weight_below_01=float(np.mean(np.asarray(weights)<.1)),
            median_held_feed_mm_s=float(np.median([r['held'][1] for r in rows])),
            batches=batches))
    result = dict(groups=groups, training_decisions=0, model_calls=0, physics_ticks=0,
        interpretation='Realized successful on-policy suffix returns include trajectory randomness and outcome selection; they are not true expected value errors. Censored tails are excluded. GAE/MC correlations and signs are descriptive, not gradient correctness. Lambda-one is an algebraic reference, not a trained candidate. Direct terminal-delta attenuation does not measure total information retained by bootstrap.',
        sealed_test_opened=False, goal_complete=False)
    write_json(stage/'results.json', result)
    lines = ['# 先导后半段价值与终末信用', '',
        '| 臂/模式 | 成功后缀行 | 后缀残差MAE/s | 后缀残差RMSE/s | 原始GAE与后缀残差同号 | 终末直接权重<0.1比例 |',
        '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['stage'].removeprefix('minimal_').removesuffix('_pilot_v1')} | {g['success_rows']} | {g['success_absolute_residual']['mean']:.3f} | {g['success_realized_residual']['rmse']:.3f} | {g['success_unscaled_sign_agreement']:.3f} | {g['fraction_success_rows_terminal_weight_below_01']:.3f} |")
    lines += ['', '读取六份已关闭日志各60k行，仅分析30k后的15个原批，共180k行；未结束批尾排除，跨批回合只使用该策略下已结束后缀。',
        '', '残差包含随机后续动作和成功条件化，不能当成真实期望价值误差。λ=1仅核对后缀回报恒等式，不证明新设置能提速。终末TD权重小不代表bootstrap已丢失同样比例信息；实际梯度还包含网络、PPO裁剪和优化器。',
        '', '结果用于判断是否值得单独检验较长信用范围或价值拟合；不自动改变状态、奖励、失败代价或已完成训练。']
    (stage/'后半段信用诊断.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_late_on_policy_suffix_credit_audit', decompressed_rows=360000,
        parsed_late_rows=180000, model_loads=0, model_calls=0, optimization_steps=0, physics_ticks=0))
    record_progress(stage.name, [stage/'receipt.json', stage/'results.json'],
        '六组先导均已完成，60k状态/奖励简化均无共同效率支持。后半段90批保存数据的实际后缀残差与终末信用诊断已完成；区别样本残差、期望价值误差和梯度。依据该证据确定下一有界问题，保留强固定和负结果，不自动组合；论文、形状图和全PDF验收继续。')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
