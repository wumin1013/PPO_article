"""Reconcile final receipts and previously analyzed checkpoints, without evaluations."""
import csv
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json, sha, record_progress
from analyze_v7_checkpoint import GROUPS


def main():
    nodes=(60000,120000,180000,240000,300000,359999)
    analysis=[BASE/f'v7_checkpoint_{n}_analysis_v1' for n in nodes]
    sources=[Path(__file__),BASE/'learning_design_v7.md',BASE/'resume_interrupted_v7.py']
    sources += [folder/name for folder in analysis for name in ('receipt.json','results.json')]
    for _,name in GROUPS:
        sources += [BASE/name/f for f in ('receipt.json','selection.json','validations.json','training_progress.json')]
        sources += [BASE/(name+'_reboot_recovery_v3')/f for f in ('receipt.json','restoration.json')]
    if not all(p.exists() for p in sources):
        raise RuntimeError('All original completions, recovered completions and common analyzed nodes must exist')
    stage=register('v7_complete_curve_audit_v1',sources,
        dict(saved_evaluation_reanalysis=0,model_loads=0,physics_ticks=0,training_decisions=0,
             optional_saved_episode_pair_reads=2),
        'Audit the completed registered v7 curves, selected checkpoints, recovery endpoint and total cost; retain stronger earlier results.')
    curves=[]
    for node,folder in zip(nodes,analysis):
        results=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        assert all(row['decisions']==node for row in results['groups'])
        curves += [dict(stage=row['stage'],label=row['label'],decisions=node,successes=row['successes'],
            full_time_s=row['full_time_s'],line_time_s=row['line_time_s'],
            corner150_distance_mm=row['cases'][3]['diagnostics']['corners'][0]['minimum_vertex_distance_mm'])
            for row in results['groups']]
    selected=[]
    optimization=[]
    total_costs={}
    for label,name in GROUPS:
        folder=BASE/name
        rec=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
        recovery=json.loads((BASE/(name+'_reboot_recovery_v3')/'receipt.json').read_text(encoding='utf-8'))
        validation=json.loads((folder/'validations.json').read_text(encoding='utf-8'))['validations']
        selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
        progress=json.loads((folder/'training_progress.json').read_text(encoding='utf-8'))
        updates=progress['updates']
        assert all(left['decisions']<right['decisions'] for left,right in zip(updates,updates[1:]))
        for lower,upper in ((0,359999),(240000,359999)):
            rows=[r for r in updates if lower<r['decisions']<=upper]
            stats=[r['statistics'] for r in rows]
            optimization.append(dict(stage=name,label=label,after_decision=lower,through_decision=upper,
                batches=len(rows),actor_early_stop_batches=sum(bool(s['actor_early_stop']) for s in stats),
                actor_minibatches=sum(s['actor_minibatches'] for s in stats),
                critic_minibatches=sum(s['critic_minibatches'] for s in stats),
                median_batch_value_loss=float(np.median([s['value_loss'] for s in stats])),
                median_batch_mean_approximate_kl=float(np.median([s['approximate_kl'] for s in stats]))))
        pointer=Path('D:/PPO_article_joint_runs/rolling_states')/name/'resume_pointer.json'
        state=json.loads(pointer.read_text(encoding='utf-8'))
        assert state['training_complete'] and state['decision']==359999
        assert recovery['through_decision']==359999
        assert recovery['original_receipt_sha256']==sha(folder/'receipt.json')
        assert rec['costs']['training_actor_calls']==359999
        assert rec['wall_s'] <= 18000
        assert sorted(v['decisions'] for v in validation)==[0,*nodes]
        choices=[]
        for v in validation:
            successes=sum(r['success'] for r in v['results'])
            choices.append(dict(decisions=v['decisions'],tag=v['tag'],successes=successes,
                complete_time_sum_s=sum(r['completion_time_s'] for r in v['results']) if successes==12 else None))
        best=max(choices,key=lambda r:(r['successes'],
            -r['complete_time_sum_s'] if r['complete_time_sum_s'] is not None else float('-inf')))
        assert best==selection['selected']
        initial=next(v for v in validation if v['decisions']==0)
        initial_results=initial['results']
        initial_complete=all(r['success'] for r in initial_results)
        initial_lines_complete=all(r['success'] for r in initial_results[:8])
        curves.append(dict(stage=name,label=label,decisions=0,
            successes=sum(r['success'] for r in initial_results),
            full_time_s=sum(r['completion_time_s'] for r in initial_results) if initial_complete else None,
            line_time_s=sum(r['completion_time_s'] for r in initial_results[:8]) if initial_lines_complete else None,
            corner150_distance_mm=None))
        chosen=next(row for row in curves if row['stage']==name and row['decisions']==best['decisions'])
        selected.append(dict(chosen,tag=best['tag'],costs=rec['costs'],
            active_runtime_upper_s=rec['wall_s'],registered_decisions_upper=360000,
            actual_saved_training_decisions=359999,uncommitted_training_decisions_upper=1))
        for key,value in rec['costs'].items():
            total_costs[key]=total_costs.get(key,0)+value
    # A previously written uncommitted episode was explicitly retained during recovery.
    restoration=json.loads((BASE/(GROUPS[0][1]+'_reboot_recovery_v3')/'restoration.json').read_text(encoding='utf-8'))
    old_value=restoration['original_uncommitted_episode_preserved']
    old=Path(old_value) if old_value is not None else None
    new=Path(restoration['recovered_episode_directory'])/old.name if old is not None else None
    episode_comparison=dict(old=str(old) if old is not None else None,
        recovered=str(new) if new is not None else None,
        exists=old is not None and old.exists() and new.exists())
    if episode_comparison['exists']:
        with np.load(old) as left,np.load(new) as right:
            identical=all(np.array_equal(left[k],right[k]) for k in ('reference','rows'))
            episode_comparison.update(arrays_identical=identical,old_rows=len(left['rows']),
                recovered_rows=len(right['rows']),old_sha256=sha(old),recovered_sha256=sha(new),
                interpretation='Saved-array consistency only; no physical audit repeated and no removal of reserved interruption uncertainty.')
    same_sampler=[]
    for sampler in ('independent','gsde'):
        joint=next(r for r in selected if r['stage']==f'joint_{sampler}_refinement_v7')
        fixed=next(r for r in selected if r['stage']==f'fixed3_{sampler}_refinement_v7')
        same_sampler.append(dict(sampler=sampler,joint_successes=joint['successes'],fixed_successes=fixed['successes'],
            joint_minus_fixed_twelve_time_s=joint['full_time_s']-fixed['full_time_s'] if joint['full_time_s'] is not None and fixed['full_time_s'] is not None else None))
    curves.sort(key=lambda r:(r['decisions'],[name for _,name in GROUPS].index(r['stage'])))
    result=dict(selected=selected,curves=curves,total_costs=total_costs,optimization=optimization,
        optimization_interpretation='Saved PPO summaries only. Mean minibatch approximate KL is not an independent full-policy KL; critic fit loss is not true future-return prediction error.',
        uncommitted_costs_reserved_upper=dict(training_actor_calls=4,training_physics_ticks=8480,ppo_updates=4),
        episode_comparison=episode_comparison,same_sampler_comparison=same_sampler,
        s2_eight_path_time_s=results['s2_eight_path_time_s'],new_training_decisions=0,
        development_only=True,independent_replicates=1,shared_pretraining_lineage=True,
        final_method_frozen=False,sealed_test_opened=False,goal_complete=False)
    write_json(stage/'results.json',result)
    with (stage/'learning_curve.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(curves[0]))
        writer.writeheader();writer.writerows(curves)
    lines=['# v7完成后的全曲线审核','',
        '| 方法 | 所选决策 | 完整成功 | 十二条总时间/s | 同八条总时间/s | 150°尖顶距离/mm |',
        '|---|---:|---:|---:|---:|---:|']
    fmt=lambda value: '未全量排名/未复算' if value is None else f'{value:.3f}'
    for row in selected:
        lines.append(f"| {row['label']} | {row['decisions']} | {row['successes']}/12 | {fmt(row['full_time_s'])} | {fmt(row['line_time_s'])} | {fmt(row['corner150_distance_mm'])} |")
    lines += ['', '## 后期PPO更新是否仍频繁提前停止', '',
        '| 方法 | 240k之后更新批 | 提前停止批 | actor小批数 | critic小批数 | 批价值损失中位数 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in optimization:
        if row['after_decision']==240000:
            lines.append(f"| {row['label']} | {row['batches']} | {row['actor_early_stop_batches']} | {row['actor_minibatches']} | {row['critic_minibatches']} | {row['median_batch_value_loss']:.6f} |")
    lines += ['', '以上只读取保存更新摘要。无提前停止但性能停滞时，不能继续将问题全归于KL截断；也不能据此证明学习率或当前探索已最优。critic拟合损失不同于真实后续回报预测误差。', '']
    lines += ['',f"S2同八条为{result['s2_eight_path_time_s']:.3f} s。六个后续共同节点全部保留，选择同时包括原初始模型；不只拿final进行比较。",'',
        '四组实际保存决策均为359999，另为重启前最后未提交一步保留预算；没有重跑已完成训练或评价。共同预训练谱系与同一开发种子不构成四个独立复本。新状态/奖励先导将使用输入匹配的新网络和独立抽样，不能把跨阶段差异全部归因于状态或奖励。', '',
        '下一步按预备设计登记base、只改状态和只去势差的有限配对先导。固定D每臂同等学习转向/速度，强v7和S2继续保留；先导不用于宣称固定D最优或最终自适应优势。']
    (stage/'全曲线与后继依据.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_all_v7_curve_and_budget_audit',saved_evaluation_reanalysis=0,
        model_loads=0,physics_ticks=0,training_decisions=0,goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        '四组v7按原预算完成并已核对359999实际终点与全部开发检查点选择；强固定、旧负结果、原恢复资料保留。接下来按最小信息单因素设计登记六组60k先导，共享base，分别只改状态/只去势差，固定D同等训练。30条封存测试未打开，最终独立种子、三类长路径、匹配消融和20–25页英文全文逐页验收仍未完成。')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
