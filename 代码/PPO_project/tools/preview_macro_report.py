"""Summarize every completed development run without selecting favorable paths."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import preview_macro_study as study


def report(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior report; choose a new --output')
    output.mkdir(parents=True)
    rows=[];costs=[];effective=[];incomplete=[]
    for run in sorted((study.OUT/'pilot').glob('*_seed*')):
        if not (run/'training_summary.json').exists():incomplete.append(run.name);continue
        cost=study.old.read(run/'training_summary.json')
        if cost['checkpoint_sha256']!=study.old.sha(run/'checkpoint.pt'):raise ValueError('Changed completed model')
        costs.append(dict(run=run.name,**cost))
        cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
        torch.manual_seed(0)
        agent=study.make_agent(cfg,type('Dimensions',(),{'observation_dim':26})())
        action,info=agent.take_action(np.zeros(26,np.float32),deterministic=True,return_info=True)
        effective.append(dict(run=run.name,initial_network_mean=info['mu'].tolist(),
            initial_network_std=info['std'].tolist(),initial_executed_mean=action.tolist(),
            active_action_mask=info['action_mask'].tolist(),gamma=agent.gamma,anchor_beta=agent.anchor_beta,
            actor_lr=agent.actor_optimizer.param_groups[0]['lr'],critic_lr=agent.critic_optimizer.param_groups[0]['lr'],
            probe_state='26 zero entries; warmed actors remain state dependent',
            authoritative_sources=['protocol.json','preview_macro_ppo.py constructor'],
            unused_inherited_descriptors={k:cfg['revision'].get(k) for k in
                ('initialization','preview_initial_std','physical_std_known_limitation')},
            note='These inherited revision descriptions do not configure the macro agent. Original YAML is retained unchanged.'))
        path=run/'validation/summary.json'
        if not path.exists():incomplete.append(run.name+'/validation');continue
        for row in study.old.read(path)['rows']:
            if row['checkpoint_sha256']!=cost['checkpoint_sha256']:raise ValueError('Stale validation model')
            if row['trace_sha256']!=study.old.sha(run/'validation'/f"{row['path_id']}.npz"):raise ValueError('Changed validation trace')
            rows.append(dict(run=run.name,**row))
    fields=['candidate','method','seed','path_id','success','elapsed_time_s','contour_error_max_mm',
            'feed_mean_mm_s','feed_max_mm_s','preview_min_mm','preview_max_mm','failure_reasons']
    with (output/'all_validation.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:row[k] for k in fields})
    groups=[]
    for candidate,method in sorted({(r['candidate'],r['method']) for r in rows}):
        subset=[r for r in rows if r['candidate']==candidate and r['method']==method]
        groups.append(dict(candidate=candidate,method=method,**study.old.backend.summarize_runs(subset)))
    paired=[]
    for candidate in study.CANDIDATES:
        fixed={(r['seed'],r['path_id']):r for r in rows if r['candidate']==candidate and r['method']=='fixed3'}
        adaptive={(r['seed'],r['path_id']):r for r in rows if r['candidate']==candidate and r['method']=='adaptive'}
        keys=set(fixed)&set(adaptive)
        common=[k for k in keys if fixed[k]['success'] and adaptive[k]['success']]
        if not keys:continue
        ft=sum(fixed[k]['elapsed_time_s'] for k in common);at=sum(adaptive[k]['elapsed_time_s'] for k in common)
        paired.append(dict(candidate=candidate,total_pairs=len(keys),common_success_pairs=len(common),
            fixed_success=sum(fixed[k]['success'] for k in keys),adaptive_success=sum(adaptive[k]['success'] for k in keys),
            time_improvement_percent_on_common_successes=None if not ft else 100*(ft-at)/ft,
            single_training_seed_only=True,complete_suite_pass=len(common)==len(keys)))
    value=dict(groups=groups,paired=paired,rows=rows,costs=costs,effective_agent_parameters=effective,incomplete=incomplete,
        generator_sha256=study.old.sha(__file__),study_protocol_sha256=study.old.sha(study.OUT/'protocol.json'),
        total_training_physics_steps=sum(c['physics_steps'] for c in costs),
        total_training_macro_decisions=sum(c['decisions'] for c in costs),
        total_training_command_samples=sum(c['command_steps'] for c in costs),
        total_training_wall_s=sum(c['training_wall_s'] for c in costs),
        scope='Single-seed development validation. No new test or repeatability claim; failures retained.')
    study.old.write(output/'report.json',value)
    study.old.write(output/'effective_agent_parameters.json',effective)
    text=['# 宏步PPO开发验证','',value['scope'],'',
          '|候选|固定成功/总数|自适应成功/总数|共同成功路径数|共同成功时间改善/%|',
          '|---|---:|---:|---:|---:|']
    for p in paired:
        gain='不可比较' if p['time_improvement_percent_on_common_successes'] is None else f"{p['time_improvement_percent_on_common_successes']:.3f}"
        text.append(f"|{p['candidate']}|{p['fixed_success']}/{p['total_pairs']}|{p['adaptive_success']}/{p['total_pairs']}|{p['common_success_pairs']}|{gain}|")
    text+=['','全部逐路径值及失败在 all_validation.csv；不能只凭共同成功子集较快就宣称整套方案更优。',
           f"本表已完成训练物理交互 {value['total_training_physics_steps']} 步，宏决策 {value['total_training_macro_decisions']} 次。额外生成制动采样单独列入 command_samples；不计为PPO决策。",
           '','公共设置及逐项变更：20 ms联合采样、1 ms KCM、取消窄运动锚定；joint_tracking保留转向/进给/距离，preview_tracking只取消独立转向残差，preview_time再将允差内软跟踪/平滑奖励改为进度减实际时间。固定组共享相应变更。',
           '','配置继承的旧initialization/preview_initial_std文字字段未传入宏步agent；实际初始均值、标准差、动作掩码、学习率已由同构agent独立读取并保存 effective_agent_parameters.json。原配置不覆盖。',
           '','恢复训练以checkpoint决策数和training_summary为预算依据；追加式原始更新日志在异常中断后可能包含重试行，不能按日志行数计算训练预算或独立样本。',
           '','未完成项：'+(', '.join(incomplete) if incomplete else '本报告覆盖的运行均完成。')]
    (output/'报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(groups=groups,paired=paired,total_physics_steps=value['total_training_physics_steps']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--study',choices=('initial','refine'),default='initial')
    args=parser.parse_args()
    if args.study=='refine':
        import preview_macro_refine
        preview_macro_refine.install()
    report(args.output)
