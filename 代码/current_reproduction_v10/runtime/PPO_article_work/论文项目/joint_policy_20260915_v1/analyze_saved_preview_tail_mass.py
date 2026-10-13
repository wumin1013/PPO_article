"""Measure fresh-draw preview tail mass at frozen saved evaluation states."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr
from study_io import BASE, register, receipt, write_json


def main():
    groups = [('Independent', 'joint_independent_exploration_learning_v1'),
              ('gSDE8', 'joint_gsde_exploration_learning_v1')]
    sources = [Path(__file__), BASE/'joint_runtime_v2.py', BASE/'joint_policy.py',
               BASE/'smooth_policy_candidate.py', BASE/'exploration_learning_analysis_60000_v1/receipt.json']
    for _, name in groups:
        sources += [BASE/name/p for p in ['receipt.json','final.pt','final_validation/3.npz','final_validation/summary.json']]
    stage = register('saved_preview_tail_mass_v1', sources,
        dict(checkpoint_deserializations=2, model_constructions=0, model_forward_calls=0,
             saved_traces=2, fresh_random_draws=0, physics_ticks=0),
        'Quantify marginal fresh-draw exploration mass at the saved 150-degree states; do not infer full trajectory probability or prescribe a distance.')
    results = []
    for label, name in groups:
        folder = BASE/name
        state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
        std = float(torch.exp(state['policy']['log_std'].clamp(-4., -.3))[2])
        with np.load(folder/'final_validation/3.npz') as data:
            ref, rows, actions = data['reference'], data['rows'], data['actions']
        before_index = np.maximum(np.arange(len(actions))*20-1, 0)
        positions = rows[before_index, :2].copy()
        positions[0] = ref[0]
        near = np.linalg.norm(positions-ref[1], axis=1) <= 10.
        assert near.any()
        means = np.arctanh((actions[near, 2]-6.4)/5.6)
        probabilities = []
        for threshold in [4.6, 6., 8.]:
            latent_threshold = np.arctanh((threshold-6.4)/5.6)
            p = ndtr((means-latent_threshold)/std)
            probabilities.append(dict(D_threshold_mm=threshold,
                minimum_probability=float(p.min()), median_probability=float(np.median(p)),
                maximum_probability=float(p.max()), equal_state_mean_probability=float(p.mean())))
        results.append(dict(method=label, saved_pre_macro_states=int(near.sum()),
            latent_D_standard_deviation=std, deterministic_D_range_mm=[float(actions[near,2].min()),float(actions[near,2].max())],
            fresh_draw_marginal_probabilities=probabilities))
    write_json(stage/'results.json', dict(results=results,
        definition='At saved pre-macro states within 10 mm of the 150-degree vertex, invert the known tanh mapping of deterministic saved D to recover its Gaussian mean. Use the frozen global latent std and exact Normal CDF; no model forward or random samples.',
        interpretation='For gSDE this is a fresh noise-matrix draw at a fixed feature state. It is not the conditional distribution given the realized trajectory or retained exploration matrix, and probabilities must not be multiplied to estimate multi-step behavior.',
        larger_D_required_for_deeper_turn=False, causal_performance_conclusion=False))
    lines = ['# 保存状态下的大前瞻探索概率', '',
        '使用两组60k联合模型的150°确定性保存状态，窗口为宏步前位置距尖顶10 mm以内。根据已知tanh映射反求均值，读取保存的全局潜变量标准差，解析计算新噪声抽样的边际尾概率；没有模型前向、随机采样或物理运行。', '',
        '| 方法 | 潜变量D标准差 | D阈值/mm | 等状态平均尾概率 |', '|---|---:|---:|---:|']
    for row in results:
        for p in row['fresh_draw_marginal_probabilities']:
            lines.append(f"| {row['method']} | {row['latent_D_standard_deviation']:.4f} | {p['D_threshold_mm']:g} | {p['equal_state_mean_probability']:.6g} |")
    lines += ['',
        '对gSDE，以上定义为固定特征状态下新抽取探索矩阵的边际概率，不是给定真实动作历史后保留矩阵的条件概率。不能将各步概率相乘来推断轨迹行为，也不能由此推断增加方差一定能改善性能。', '',
        '4.6、6和8 mm只是与已完成距离研究相联系的诊断阈值；转向和速度也影响切弯，不声称较深切弯必须使用其中某个距离。该诊断补充探索覆盖的观察，不改变正在进行的v7设置或预算。']
    (stage/'前瞻探索概率.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_saved_distribution_tail_analysis',
        checkpoint_deserializations=2, model_constructions=0, model_forward_calls=0,
        fresh_random_draws=0, physics_ticks=0, training_decisions=0))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
