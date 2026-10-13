"""Record inspected front pages and analyze completed v5 data without rollouts."""
import json
import gzip
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from study_io import BASE, register, receipt, write_json, record_progress, sha
from summarize_conditioned_training import distance_to_commands


def main():
    names = ['joint_remaining_credit_training_v5', 'fixed3_remaining_credit_training_v5',
             'fixed4p6_remaining_credit_training_v5']
    sources = [Path(__file__), BASE/'summarize_conditioned_training.py']
    for name in names:
        sources += [BASE/name/f for f in ['receipt.json','selection.json','training_progress.json',
                    'decisions.jsonl.gz','final.pt','final_validation/summary.json']]
        sources += [BASE/name/'final_validation'/f'{i}.npz' for i in range(12)]
    stage = register('remaining_credit_analysis_v5', sources,
        dict(saved_trace_files=36,checkpoint_deserializations=3,training_journal_rows=180000,
             model_forward_calls=0,physics_ticks=0,training_decisions=0),
        'Analyze all completed v5 checkpoints and saved final commands; identify performance gaps without repeating experiments.')
    results = []
    for name in names:
        directory=BASE/name
        selection=json.loads((directory/'selection.json').read_text(encoding='utf-8'))
        progress=json.loads((directory/'training_progress.json').read_text(encoding='utf-8'))
        state=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
        std=state['policy']['log_std'].clamp(-4.,-.3).exp().numpy()
        episodes=progress['episodes']
        traces=[]
        for i in range(12):
            with np.load(directory/'final_validation'/f'{i}.npz') as data:
                ref,rows,actions=data['reference'],data['rows'],data['actions']
            stats=dict(path=i,feed_median=float(np.median(actions[:,1])),
                D_quantiles=np.quantile(actions[:,2],[.1,.5,.9]).tolist(),
                D_mean_absolute_step=float(np.mean(np.abs(np.diff(actions[:,2])))),
                speed_median=float(np.median(np.linalg.norm(rows[:,2:4],axis=1))))
            if i<4:
                near=np.linalg.norm(rows[:,:2]-ref[1],axis=1)<=10.
                stats.update(nearest_corner_mm=distance_to_commands(ref[1],np.vstack([ref[0],rows[:,:2]])),
                    corner_speed_median=float(np.median(np.linalg.norm(rows[near,2:4],axis=1))))
            traces.append(stats)
        held=[]
        jumps=[]
        previous=None
        n=0
        with gzip.open(directory/'decisions.jsonl.gz','rt',encoding='utf-8') as file:
            for line in file:
                row=json.loads(line)
                n+=1
                if row['decision']>45000:
                    held.append(row['held'])
                    if previous is not None and previous['episode']==row['episode']:
                        jumps.append(abs(row['held'][2]-previous['held'][2]))
                previous=row
        assert n==60000
        held=np.asarray(held)
        results.append(dict(stage=name,candidates=selection['candidates'],selected=selection['selected'],
            final_latent_std=std.tolist(),episodes=len(episodes),successful_episodes=sum(r['success'] for r in episodes),
            last100_successes=sum(r['success'] for r in episodes[-100:]),last100_count=len(episodes[-100:]),
            last15000_held_quantiles=np.quantile(held,[.1,.5,.9],axis=0).tolist(),
            last15000_D_step_quantiles=np.quantile(jumps,[.5,.9,.99]).tolist(),final_traces=traces,
            mean_value_loss_last8=float(np.mean([r['statistics']['value_loss'] for r in progress['updates'][-8:]]))))
    write_json(stage/'results.json',dict(groups=results,causal_exploration_test=False,
        superiority_established=False,models_executed=0,physics_ticks=0))
    lines=['# v5完整开发结果及前瞻探索诊断','',
        '全部三组完成各60,000个新决策和五轮开发验证。此处只读取原始结果、训练日志和保存轨迹，没有新模型执行、物理重放或训练。','',
        '| 方法 | initial | 15k | 30k | 45k | 60k | 训练成功回合 | 最后100回合成功 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        cells=[f"{c['complete_time_sum_s']:.3f} s" if c['complete_time_sum_s'] is not None else f"{c['successes']}/12" for c in r['candidates']]
        lines.append('| '+r['stage']+' | '+' | '.join(cells)+f" | {r['successful_episodes']}/{r['episodes']} | {r['last100_successes']}/100 |")
    lines += ['', '三组final均12/12成功。自适应33.009 s，比固定3的27.678 s慢19.26%，比固定4.6的29.331 s慢12.54%；目前没有自适应优势。固定3在各次检查点持续改善，60k不能被称为已充分收敛。','',
        '| 方法 | 最终潜变量标准差(转向/速度/D) | 末15k相邻D变化中位数/mm | 末8次critic损失均值 |',
        '|---|---|---:|---:|']
    for r in results:
        lines.append(f"| {r['stage']} | {r['final_latent_std']} | {r['last15000_D_step_quantiles'][0]:.4f} | {r['mean_value_loss_last8']:.4f} |")
    lines += ['', '| 方法 | 150°最近尖顶距离/mm | 150°近角速度中位数/(mm/s) | 150°全程D中位数/mm |', '|---|---:|---:|---:|']
    for r in results:
        t=r['final_traces'][3]
        lines.append(f"| {r['stage']} | {t['nearest_corner_mm']:.4f} | {t['corner_speed_median']:.4f} | {t['D_quantiles'][1]:.4f} |")
    lines += ['', 'D每20 ms重新采样，较大的独立探索扰动可能使联合策略面对更不稳定的转向目标，并通过失败代价形成偏保守的速度选择。现有日志只支持这一待检验机制，不能证明其为唯一原因；三组状态访问分布也不同。', '',
        '后继应从当前较强的固定3模型共同初始化，先用较小D探索检验能否保持原可行性和速度，再给予三组相同的新训练预算及选优机会。公共几何、奖励、执行器和限值保持一致；不增加逐角D表、角度限速或离尖顶奖励。固定3仍持续变好，必须继续保留其训练和较好检查点，不能只延长自适应训练。', '',
        '固定D合理搜索、独立种子、长路径、封存测试和完整论文仍未完成。此报告不改变封存测试状态，也不表示投稿目标完成。']
    (stage/'开发结果与探索诊断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_saved_data_analysis',groups=3,validation_summaries=15,
        saved_traces=36,checkpoint_deserializations=3,models=0,physics_ticks=0,superiority_established=False))
    proof=BASE/'front_sections_proof_v3'
    assert sha(proof/'front_sections.pdf')=='51340d3eeca0b2eff1988c8e97cf83d6983f588019484b0105c995cfa04f30d6'
    qa=proof/'visual_qa.json'
    if qa.exists():
        raise FileExistsError(qa)
    figures=[BASE/'method_figure1_v4/Figure_1.pdf',BASE/'method_figures_v2/Figure_2.pdf',BASE/'method_figures_v2/Figure_3.pdf']
    write_json(qa,dict(status='nine_front_pages_and_three_embedded_figures_visually_passed',
        pdf_sha256=sha(proof/'front_sections.pdf'),pages=9,
        visual_observation='All nine rendered page PNGs were viewed in the preceding active work before context handoff; this record is written afterwards without repeating rendering.',
        rendered_pages=[dict(path=f'D:/PPO_article_joint_runs/front_sections_proof_v3/page-{i}.png',sha256=sha(Path(f'D:/PPO_article_joint_runs/front_sections_proof_v3/page-{i}.png'))) for i in range(1,10)],
        figures=[dict(path=str(p),sha256=sha(p),svg_sha256=sha(p.with_suffix('.svg'))) for p in figures],
        latex_warnings=0,whole_manuscript_acceptance=False,goal_complete=False,
        recorded_utc=datetime.now(timezone.utc).isoformat()))
    backup=BASE/'before'/('v5_analysis_and_front_qa_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False)
    readme=BASE/'manuscript_joint_v1/README.md'
    report=BASE/'逐图保留与修改报告_进行中.md'
    for p in [readme,report]:
        shutil.copy2(p,backup/p.name)
    readme.write_text(readme.read_text(encoding='utf-8')+'\n## 当前可查看稿\n\n当前编译入口为`front_sections_proof_v3/front_sections.tex`，PDF为同目录`front_sections.pdf`（相对于joint_policy_20260915_v1）。它含引言、问题定义、方法、三张方法图及引用文献，共9页，全部9页目视检查通过，日志无排版或引用告警；验收见同目录`visual_qa.json`。前半稿使用实际三动作学习和v5奖励/终点候选语义，不含未获得的性能结论。\n\n`method_proof.pdf`是旧5页历史摘录；旧入口不再作为当前编译入口。v2前半稿保留，其引用处主张错误已经在v3修正，见`citation_claim_corrections_v1.md`。当前9页稿仍不是20–25页整篇稿。\n',encoding='utf-8')
    text=report.read_text(encoding='utf-8')
    replacements={
        '原构图保留，具体标签待逐项核查，尚未生成新最终图':'已生成`method_figure1_v4/Figure_1.svg/pdf`，保留左右对照构图，更新三输出、离散插补点和向量KCM；SVG实际转PDF，独立图及9页稿内目视通过',
        '几何定义及25项合成检查完成；原外侧斜接Pl/Pr与新等距域不等价，不能原样标作新审核域；最终图待制作':'已生成`method_figures_v2/Figure_2.svg/pdf`，130°有限折线欧氏域、单侧ε及正确裁剪的有向门；绿色圆弧明确为几何示意；独立图及稿内目视通过',
        '方法正文已据实现写成；原构图保留，旧标量线/角约束和不适用的奖励标签需修订，最终图待制作':'已生成`method_figures_v2/Figure_3.svg/pdf`，保留闭环构图，更新72维、三动作、双周期、向量KCM及真实奖励；独立图及稿内目视通过',
        'v2/v3开发结果已全量汇总；当前最强全量成功为固定4.6、33.732 s。v4三组仍训练中，尚不能制作自适应获胜的最终图':'v5三组各60k已完成且final均12/12；自适应33.009 s、固定3为27.678 s、固定4.6为29.331 s；固定仍较强。v4联合与固定4.6继续原预算，固定3异常保留；尚不能制作自适应获胜图'}
    for old,new in replacements.items():
        assert old in text
        text=text.replace(old,new)
    text+='\n## 当前前三图及前半稿验收\n\n采用Fig.1 v4、Fig.2/3 v2。三张图的SVG/PDF及哈希和九页逐页PNG绑定见`front_sections_proof_v3/visual_qa.json`。9页v3稿全页已检查，排版和引用告警为0；v2稿的引用处主张修正另有记录。旧5页摘录和旧失败布局均保留。前三图完成不代表Fig.4–9和20–25页全文验收完成。\n'
    report.write_text(text,encoding='utf-8')
    record_progress('v5_complete_and_front9_qa_v1',[stage/'receipt.json',qa]+[BASE/n/'receipt.json' for n in names],
        'v5三组各60000步已完成，final均12/12：联合33.009 s、固定3为27.678 s、固定4.6为29.331 s，固定仍较强。前三图及9页前半稿已全页目视验收；目标未完成。继续v4原预算，结合保存日志检验前瞻探索与速度保守的关系，准备公平的有界后继；合理固定D、独立种子、长路径、封存测试及20–25页全文仍待完成。')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
