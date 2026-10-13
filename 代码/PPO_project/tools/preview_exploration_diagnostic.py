"""Offline numerical and saved-log audit; no environment, training or test use.

The scalar float32 probe reproduces the frozen actor's std initialization,
softplus and clamp exactly. Candidate scales are gradient-flow diagnostics,
not evaluated policies. Epoch and trace statistics are descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

import preview_direct_report as saved


def std_probe(requested_std):
    bias=torch.tensor(math.log(math.expm1(requested_std-.001)),dtype=torch.float32,requires_grad=True)
    raw=torch.nn.functional.softplus(bias)+.001
    scale=torch.clamp(raw,min=.01,max=2.)
    scale.backward()
    return dict(requested_std_u=requested_std,raw_std_u=float(raw.detach()),
        clipped_std_u=float(scale.detach()),gradient_std_wrt_bias=float(bias.grad),
        distance_sigma_mm=3.8*float(scale.detach()),
        scope='Scalar float32 transform diagnostic, not a policy performance result')


def adam_scaling_probe():
    rows=[]
    for common_gradient_scale in (1.,.01,.001):
        p=torch.tensor([1.],requires_grad=True)
        optimizer=torch.optim.Adam([p],lr=1e-5)
        p.grad=torch.tensor([8.*common_gradient_scale])
        optimizer.step()
        rows.append(dict(common_gradient_scale=common_gradient_scale,parameter_delta=float(p.detach()-1.)))
    return dict(rows=rows,scope='One synthetic Adam step, not a training experiment',
        interpretation='Common gradient clipping changes magnitudes; Adam moment normalization prevents inferring a proportional learning-rate reduction from that factor alone')


def audit(base,output=None):
    output=output or base/'exploration_diagnostic';inputs=saved.saved.Inputs()
    protocol=inputs.json(base/'protocol.json')
    if not protocol['version'].startswith('preview-direct') or protocol['geometry_distance_rule_used']:
        raise ValueError('Only rule-free direct studies are supported')
    summaries=[];epochs=[]
    for stage in ('pilot','formal'):
        seeds=[protocol['pilot_seed']] if stage=='pilot' else protocol['formal_seeds']
        for method in protocol[f'{stage}_variants']:
            for seed in seeds:
                run=base/stage/f'{method}_seed{seed}'
                path=run/'actor_update_diagnostics.jsonl'
                if not path.exists():continue
                provenance=inputs.json(run/'provenance.json')
                cfg=yaml.safe_load(inputs.read(run/'config.yaml'))
                if (provenance['source_sha256']!=protocol['source_sha256']
                    or provenance['config_sha256']!=inputs.sha(run/'config.yaml')):
                    raise ValueError(f'Mixed provenance: {run}')
                rows,partial=saved.read_json_lines(path,inputs)
                local=[]
                for update in rows:
                    for epoch in update['epochs']:
                        norm=epoch['actor_gradient_norm_before_clip']
                        factor=min(1.,.5/(norm+1e-6))
                        entry=dict(stage=stage,method=method,seed=seed,step=update['step'],epoch=epoch['epoch'],
                            invocation_id=update.get('invocation_id'),actor_global_norm=norm,
                            common_clip_factor=factor,preview_weight_norm_before=epoch['preview_mu_weight_gradient_norm'],
                            preview_weight_norm_after_estimate=epoch['preview_mu_weight_gradient_norm']*factor,
                            physical_weight_norm_before=epoch['physics_mu_weight_gradient_norm'],
                            joint_ratio_min=epoch['ratio_min'],joint_ratio_max=epoch['ratio_max'],
                            joint_ratio_exceeds_ppo_interval=epoch['ratio_min']<1-cfg['ppo']['eps'] or epoch['ratio_max']>1+cfg['ppo']['eps'])
                        local.append(entry);epochs.append(entry)
                if not local:continue
                last=[r['epochs'][-1] for r in rows if r['epochs']]
                summaries.append(dict(stage=stage,method=method,seed=seed,optimizer_epochs=len(local),
                    latest_logged_step=rows[-1]['step'],partial_jsonl_tail=partial,
                    actor_lr=cfg['ppo']['actor_lr'],ppo_clip_epsilon=cfg['ppo']['eps'],entropy_coefficient=cfg['ppo']['ent_coef'],
                    global_gradient_clipped_epoch_fraction=float(np.mean([r['common_clip_factor']<1 for r in local])),
                    common_clip_factor_median=float(np.median([r['common_clip_factor'] for r in local])),
                    common_clip_factor_min=float(np.min([r['common_clip_factor'] for r in local])),
                    physical_to_preview_weight_gradient_median=float(np.median([r['physical_weight_norm_before']/r['preview_weight_norm_before'] for r in local if r['preview_weight_norm_before']>0])) if method.startswith('direct') else None,
                    joint_ratio_outside_interval_epoch_fraction=float(np.mean([r['joint_ratio_exceeds_ppo_interval'] for r in local])),
                    preview_mu_distance_min_mm=.8+3.8*min(r['preview_mu_min'] for r in last),
                    preview_mu_distance_max_mm=.8+3.8*max(r['preview_mu_max'] for r in last),
                    preview_sigma_distance_mean_mm=3.8*float(np.mean([r['preview_std_mean'] for r in last])),
                    warning='Epoch fraction is not sample clip fraction; saved log lacks per-sample action-dimension ratios'))
    probes=[std_probe(s) for s in (.01,.010001,.05,.1)]
    result=dict(protocol_version=protocol['version'],scope='saved training logs and scalar numerical diagnostics only',
        std_initialization=probes,adam_scaling=adam_scaling_probe(),training=summaries,
        evidence_boundary='No new policy was trained or evaluated; no validation or held-out test files read')
    output.mkdir(parents=True,exist_ok=True)
    saved.saved.write_json(output/'summary.json',result)
    saved.saved.write_csv(output/'optimizer_clip_diagnostics.csv',epochs)
    saved.saved.write_csv(output/'std_initialization_probe.csv',probes)
    review=['# 前瞻探索与优化诊断','',
        '本文件由保存的训练日志和独立 float32 数值探针生成，不执行环境、不训练策略、不读取测试集。',
        '', '## 已确认的标准差初始化死区', '',
        f"本轮 direct/fine 初始化目标 sigma_u=0.01，但 softplus+bias 实际为 {probes[0]['raw_std_u']:.16g}，",
        f"clamp 下限表示为 {probes[0]['clipped_std_u']:.16g}，反向梯度为 {probes[0]['gradient_std_wrt_bias']:g}。",
        '这针对本轮 make_agent 的显式初始化；不能外推到原 main 默认 std_head.bias=0.3 或其他不同初值版本。',
        f"将初值仅设为0.05时，初始 sigma_D={probes[2]['distance_sigma_mm']:.8f} mm，bias梯度={probes[2]['gradient_std_wrt_bias']:.8f}。",
        '0.010001 的探针也恢复梯度，表明数值死区和探索幅度是两个可区分因素；0.05候选同时远离下限并增大初始探索，单凭其结果不能把两者的贡献分开。',
        '', '## 全局梯度裁剪与 Adam', '',
        'AnchoredPPO 对全部 actor 参数统一 clip_grad_norm_(max_norm=0.5)，前瞻头日志记录的是裁剪前梯度。',
        '物理动作和前瞻动作共享特征，物理锚定项可通过特征层影响前瞻，尽管它不直接惩罚第三头。',
        '', '| 阶段 | 配置 | 种子 | 全局梯度裁剪epoch比例 | 裁剪因子中位数 | 物理/前瞻输出权重梯度比中位数 |',
        '|---|---|---:|---:|---:|---:|']
    for r in summaries:
        if r['method'].startswith('direct'):
            review.append(f"|{r['stage']}|{r['method']}|{r['seed']}|{r['global_gradient_clipped_epoch_fraction']:.4f}|{r['common_clip_factor_median']:.6f}|{r['physical_to_preview_weight_gradient_median']:.4f}|")
    review += ['', '不能把共同裁剪因子直接解释为前瞻有效学习率按同比例下降。独立一阶 Adam 数值探针中，梯度统一缩放1、0.01、0.001均产生约-1e-5的首次参数变化。真实动量历史和共享特征仍可能耦合优化。',
        '', '## PPO 联合似然和日志限制', '',
        '代码把活动维度的潜在高斯 log_prob 求和；完整动作的联合 ratio 经过 exp 前[-5,5]限幅，再在 PPO 目标中使用0.1裁剪区间。固定前瞻排除第三维的似然和熵。',
        '因此任一活动维度的策略变化都可能使共同样本进入 PPO 裁剪区域；现有 ratio_min/max 只可说明某个epoch存在越界样本，不能恢复每维贡献或样本裁剪率。',
        '保存日志显示标准差死区与狭窄采样，但没有证明它们是所有失败的原因。每1 ms独立采样意味着短时间内正负扰动可能抵消；没有持续探索消融时不对此作因果结论。',
        '', '## 已注册的最小候选与剩余核验', '',
        '独立 preview_direct_explore 仅把第三头初始sigma_u从0.01改到0.05，保持物理两头、学习率、网络、奖励和完整D映射。固定配置的第三头也相同初始化，但保持inactive。',
        '两物理标准差保留共同初始clamp零梯度限制，需在论文披露。正式优势必须比较匹配beta的独立训练固定基线，检查完成率、约束和配对执行时间。',
        '后续读取同预算训练日志中的mu距离范围、sigma变化及性能；扩大采样范围和非零梯度只证明优化参与，不能代替性能结论。',
        '', '运行命令：`python PPO_project/tools/preview_exploration_diagnostic.py --runs 论文项目/revision_20260908/preview_direct`。',
        '原始优化epoch和插补样本仅作描述，独立训练种子及路径才是性能统计单位。']
    (output/'review.md').write_text('\n'.join(review)+'\n',encoding='utf-8')
    source=Path(__file__);inputs.read(source)
    for path in (saved.ROOT/'PPO_project/src/algorithms/ppo.py',saved.ROOT/'PPO_project/tools/preview_direct_research.py'):
        inputs.read(path)
    outputs=['summary.json','optimizer_clip_diagnostics.csv','std_initialization_probe.csv','review.md']
    saved.saved.write_json(output/'hash_manifest.json',dict(inputs=inputs.hashes,
        outputs={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in outputs},
        python=sys.version,torch=torch.__version__,command=' '.join(sys.argv)))
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=saved.DEFAULT_RUNS)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();audit(args.runs,args.output)


if __name__=='__main__':main()
