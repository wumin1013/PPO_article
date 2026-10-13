"""Decompose frozen GAE dispersion by actual episode outcome, without models."""
import json
from pathlib import Path
import numpy as np
from joint_policy import advantages
from resume_codec_v5 import load_resume
from study_io import BASE, register, receipt, write_json, sha


def main():
    groups = ['joint_independent_exploration_learning_v1', 'fixed3_independent_exploration_learning_v1']
    stage = register('credit_outcome_decomposition_v1', [Path(__file__), BASE/'joint_policy.py',
        BASE/'resume_codec_v5.py', BASE/'saved_action_credit_v1/results.json',
        BASE/'saved_action_credit_v1/external_sources.json'],
        dict(buffer_loads=2, model_calls=0, physics_ticks=0, optimization_steps=0),
        'Follow up the approximately 24-fold GAE scale difference: partition two frozen independent-exploration batches by observed episode outcome, without new models or optimization.')
    bindings = json.loads((BASE/'saved_action_credit_v1/external_sources.json').read_text(encoding='utf-8'))
    records = []
    for name in groups:
        binding = next(r for r in bindings if r['stage'] == name)
        path = Path(binding['path'])
        assert sha(path) == binding['sha256']
        state = load_resume(path)
        batch = state['batch']
        adv, _ = advantages(batch['reward'], batch['value'], batch['next_value'], batch['terminal'])
        n = len(adv)
        batch_start = state['decision']-n+1
        outcome = np.full(n, 'censored', dtype='U12')
        path_id = np.full(n, -1)
        episode_end = 0
        episode_rows = []
        for episode in state['episodes']:
            episode_start = episode_end+1
            episode_end += episode['decisions']
            first = max(episode_start, batch_start)
            last = min(episode_end, state['decision'])
            if first <= last:
                selection = slice(first-batch_start, last-batch_start+1)
                outcome[selection] = 'success' if episode['success'] else 'failure'
                path_id[selection] = episode['path']
                episode_rows.append(dict(episode=episode['episode'], path=episode['path'],
                    success=episode['success'], reason=episode['reason'], rows=last-first+1,
                    raw_advantage_mean=float(adv[selection].mean()),
                    raw_advantage_min=float(adv[selection].min()), raw_advantage_max=float(adv[selection].max())))
        # Saved completed episode decision lengths exactly cover all observed terminals.
        assert len(episode_rows) == sum(batch['terminal'])
        assert np.count_nonzero(outcome == 'censored') == state['environment']['decisions']
        center = float(adv.mean())
        squared = (adv.astype(float)-center)**2
        total = squared.sum()
        partitions = []
        for label in ['success', 'failure', 'censored']:
            mask = outcome == label
            values = adv[mask]
            partitions.append(dict(outcome=label, samples=int(mask.sum()),
                raw_advantage_std=float(values.std()) if len(values) else None,
                raw_advantage_mean=float(values.mean()) if len(values) else None,
                share_of_total_centered_square=float(squared[mask].sum()/total) if total else None))
        records.append(dict(stage=name, batch_rows=n, batch_start_decision=batch_start,
            raw_advantage_std=float(adv.std()), partitions=partitions, observed_episode_suffixes=episode_rows))
        assert sha(path) == binding['sha256']
    write_json(stage/'results.json', dict(groups=records, model_calls=0, physics_ticks=0,
        independent_samples_assumed=False, causal_claim=False))
    lines = ['# 优势尺度与回合结局', '',
        '| 方法 | 批内结局 | 样本 | 组内优势标准差 | 全批中心平方和占比 |', '|---|---|---:|---:|---:|']
    for r in records:
        for p in r['partitions']:
            if p['samples']:
                lines.append(f"| {r['stage']} | {p['outcome']} | {p['samples']} | {p['raw_advantage_std']:.5f} | {p['share_of_total_centered_square']:.2%} |")
    lines += ['', '分组采用原保存回合结局，未完成批尾记为censored。平方和以全批均值为中心，因此包含组间差异；占比不是策略梯度损失比例。成功回合内的优势仍依赖原critic。单一保存批只形成机制线索，不能证明取消归一化、减轻失败处罚或更换价值模型会改善真实性能；不改变v7。']
    (stage/'优势尺度分解.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_outcome_decomposition', buffer_loads=2,
        model_calls=0, physics_ticks=0, optimization_steps=0, training_decisions=0))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
