"""Reconcile the completed single-factor pilot without model or motion execution."""
import csv
import json
from pathlib import Path
from study_io import BASE,register,receipt,write_json,record_progress


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    folders=[BASE/f'minimal_{arm}_{mode}_pilot_v1'
             for arm in ('base','state','reward') for mode in ('joint','fixed')]
    analyses=[BASE/f'minimal_factors_{n}_analysis_v1' for n in (30000,60000)]
    sources=[Path(__file__),BASE/'minimal_factorial_design_v1.md']
    sources += [p/name for p in folders for name in
                ('registration.json','receipt.json','selection.json','validations.json','training_progress.json','initial_completed.json')]
    sources += [p/name for p in analyses for name in ('receipt.json','results.json')]
    if not all(p.exists() for p in sources):
        raise RuntimeError('Wait for six completion receipts and both common-node analyses')
    stage=register('minimal_factorial_complete_audit_v1',sources,
        dict(models=0,physics_ticks=0,training_decisions=0),
        'Audit all completed single-factor curves, selection, cost and optimization; no automatic combination or budget increase.')
    analyzed={n:read(p/'results.json') for n,p in zip((30000,60000),analyses)}
    curves=[]
    groups=[]
    totals={}
    for p in folders:
        reg=read(p/'registration.json')['budget']
        rec=read(p/'receipt.json')
        progress=read(p/'training_progress.json')
        vals=read(p/'validations.json')['validations']
        selection=read(p/'selection.json')
        assert reg['training_decisions']==rec['costs']['training_actor_calls']==60000
        assert rec['wall_s']<=reg['wall_s']==7200
        assert reg['seed']==917 and reg['fixed_d']==(3. if reg['mode']=='fixed' else None)
        assert reg['observation_dim']==(22 if reg['arm']=='state' else 72)
        assert [v['decisions'] for v in vals]==[0,30000,60000]
        choices=[]
        for v in vals:
            assert sorted(r['path'] for r in v['results'])==list(range(12))
            n=sum(r['success'] for r in v['results'])
            total=sum(r['completion_time_s'] for r in v['results']) if n==12 else None
            choice=dict(decisions=v['decisions'],tag=v['tag'],successes=n,complete_time_sum_s=total)
            choices.append(choice)
            detail=next((r for r in analyzed[v['decisions']]['groups'] if r['stage']==p.name),None) if v['decisions'] else None
            corner=detail['cases'][3]['diagnostics'] if detail else None
            curves.append(dict(stage=p.name,arm=reg['arm'],mode=reg['mode'],**choice,
                vertex150_mm=corner['corners'][0]['minimum_vertex_distance_mm'] if corner else None,
                median150_speed_mm_s=corner['speed_quantiles_mm_s'][1] if corner else None,
                median150_preview_mm=corner['preview_quantiles_mm'][1] if corner else None))
        best=max(choices,key=lambda r:(r['successes'],
            -r['complete_time_sum_s'] if r['complete_time_sum_s'] is not None else -r['decisions']))
        assert best==selection['selected']==rec['selected']
        assert all(a['decisions']<b['decisions'] for a,b in zip(progress['updates'],progress['updates'][1:]))
        optimization=[]
        for lower,upper in ((0,30000),(30000,60000)):
            stats=[r['statistics'] for r in progress['updates'] if lower<r['decisions']<=upper]
            optimization.append(dict(after=lower,through=upper,batches=len(stats),
                early_stop_batches=sum(s['actor_early_stop'] for s in stats),
                actor_minibatches=sum(s['actor_minibatches'] for s in stats),
                critic_minibatches=sum(s['critic_minibatches'] for s in stats)))
        episodes=progress['episodes']
        groups.append(dict(stage=p.name,arm=reg['arm'],mode=reg['mode'],observation_dim=reg['observation_dim'],
            parameters=rec['parameters'],selected=best,wall_s=rec['wall_s'],costs=rec['costs'],
            component_seconds=rec['component_seconds'],optimization=optimization,
            completed_training_episodes=len(episodes),training_successes=sum(r['success'] for r in episodes),
            training_failures=sum(not r['success'] for r in episodes)))
        for key,value in rec['costs'].items():
            totals[key]=totals.get(key,0)+value
    result=dict(groups=groups,curves=curves,total_costs=totals,
        primary_60000_comparisons=analyzed[60000]['comparisons'],
        primary_60000_directional_support=analyzed[60000]['paired_directional_support'],
        timing_scope='policy includes actor and critic; physics includes env.step; checkpoint covers rolling saves only; none is isolated deployment latency',
        shape_acceptance_pending=True,independent_seeds_per_arm=1,
        automatic_combination=False,automatic_budget_extension=False,sealed_test_opened=False,goal_complete=False)
    write_json(stage/'results.json',result)
    with (stage/'curves.csv').open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(curves[0]))
        writer.writeheader();writer.writerows(curves)
    lines=['# 单因素先导完整审核','',
        '| 臂 | 模式 | 输入维数 | 参数数 | 所选决策 | 所选完整成功 | 所选十二条时间/s |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in groups:
        b=r['selected']
        t='未全量排名' if b['complete_time_sum_s'] is None else f"{b['complete_time_sum_s']:.3f}"
        lines.append(f"| {r['arm']} | {r['mode']} | {r['observation_dim']} | {r['parameters']} | {b['decisions']} | {b['successes']}/12 | {t} |")
    lines += ['', '主比较仍为同60k、同模式内state/reward对base；所选模型表不能代替这一单因素比较。全部18个节点、失败与真实成本在JSON/CSV保留。',
        '', '60k共同方向性支持：'+json.dumps(result['primary_60000_directional_support'],ensure_ascii=False),
        '', '形状需结合真实XY和局部速度审核。单个开发种子不证明最小充分性、固定距离最优或最终自适应优势。没有自动组合或扩预算。',
        '', '分项计时包含训练用critic和整个env.step；不能将其写成单独部署actor延迟或纯物理执行时间。']
    (stage/'完整先导与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_six_arm_curve_selection_cost_audit',models=0,physics_ticks=0,training_decisions=0,goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        '六组状态/奖励单因素先导按60k/7200秒上限完成并已核全曲线、选择与实际成本；分别按同模式同60k对共享base判断，保留强v7与负结果。后继需依据真实支持和形状诊断，不自动组合、扩预算或打开封存测试；合理固定选距、独立完整种子、最终长路径、匹配消融和20–25页英文全文仍未完成。')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
