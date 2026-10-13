"""Compare a new common single-factor checkpoint using saved evaluations only."""
import argparse
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json, sha, record_progress
from summarize_conditioned_training import diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--decisions',type=int,choices=(30000,60000),required=True)
    args=parser.parse_args()
    tag='final_validation' if args.decisions==60000 else 'validation_30000'
    groups=[(arm,mode,BASE/f'minimal_{arm}_{mode}_pilot_v1')
            for arm in ('base','state','reward') for mode in ('joint','fixed')]
    sources=[Path(__file__),BASE/'minimal_factorial_design_v1.md',
        BASE/'summarize_conditioned_training.py',BASE/'s2_line_development_v2/results.json']
    available=[]
    for arm,mode,folder in groups:
        if not (folder/'registration.json').exists():
            raise RuntimeError(f'Unstarted arm: {folder.name}')
        summary=folder/tag/'summary.json'
        failure=folder/'failure.json'
        if not summary.exists() and not failure.exists():
            raise RuntimeError(f'Wait for the common node: {folder.name}')
        if args.decisions==60000 and not (folder/'receipt.json').exists() and not failure.exists():
            raise RuntimeError(f'Wait for final completion receipt: {folder.name}')
        sources += [folder/'registration.json']
        sources += [p for p in (folder/'receipt.json',folder/'failure.json',folder/'selection.json') if p.exists()]
        if summary.exists():
            sources += [summary,*[folder/tag/f'{i}.npz' for i in range(12)]]
        available.append((arm,mode,folder,summary.exists()))
    stage=register(f'minimal_factors_{args.decisions}_analysis_v1',sources,
        dict(saved_trace_files_upper=72,models=0,physics_ticks=0,training_decisions=0),
        'Matched single-factor development comparison; retain failed stages and all fixed-preview opportunities.')
    records=[]
    traces=0
    for arm,mode,folder,present in available:
        record=dict(arm=arm,mode=mode,stage=folder.name,decisions=args.decisions,
            checkpoint_available=present,training_failed=(folder/'failure.json').exists(),
            training_completed=(folder/'receipt.json').exists(),cases=[])
        if present:
            summary=json.loads((folder/tag/'summary.json').read_text(encoding='utf-8'))
            assert len(summary['results'])==12
            for result in summary['results']:
                path=folder/tag/f"{result['path']}.npz"
                with np.load(path) as data:
                    detail=diagnostics(data['reference'],data['rows'],data['actions'])
                traces+=1
                record['cases'].append(dict(result=result,diagnostics=detail,trace_sha256=sha(path)))
            complete=all(r['success'] for r in summary['results'])
            line_complete=all(r['success'] for r in summary['results'][:8])
            record.update(successes=summary['successes'],
                full_time_s=sum(r['completion_time_s'] for r in summary['results']) if complete else None,
                line_time_s=sum(r['completion_time_s'] for r in summary['results'][:8]) if line_complete else None)
        else:
            record.update(successes=None,full_time_s=None,line_time_s=None)
        records.append(record)
    comparisons=[]
    for arm in ('state','reward'):
        for mode in ('joint','fixed'):
            base=next(r for r in records if r['arm']=='base' and r['mode']==mode)
            changed=next(r for r in records if r['arm']==arm and r['mode']==mode)
            full=base['successes']==changed['successes']==12
            common=[]
            if base['checkpoint_available'] and changed['checkpoint_available']:
                a={r['result']['path']:r['result'] for r in base['cases']}
                b={r['result']['path']:r['result'] for r in changed['cases']}
                for index in sorted(a):
                    if a[index]['success'] and b[index]['success']:
                        common.append(dict(path=index,base_s=a[index]['completion_time_s'],
                            changed_s=b[index]['completion_time_s'],
                            changed_minus_base_s=b[index]['completion_time_s']-a[index]['completion_time_s']))
            comparisons.append(dict(factor=arm,mode=mode,base_successes=base['successes'],
                changed_successes=changed['successes'],both_all_complete=full,
                full_time_difference_s=changed['full_time_s']-base['full_time_s'] if full else None,
                common_success_cases=common,
                directional_efficiency_support=full and changed['full_time_s']<base['full_time_s']))
    support={arm:all(r['directional_efficiency_support'] for r in comparisons if r['factor']==arm)
             for arm in ('state','reward')}
    result=dict(groups=records,comparisons=comparisons,paired_directional_support=support,
        primary_budget_endpoint=args.decisions==60000,shape_acceptance_pending=True,
        independent_seeds_per_arm=1,development_only=True,combination_authorized_by_this_analysis=False,
        sealed_test_opened=False,goal_complete=False)
    write_json(stage/'results.json',result)
    fmt=lambda x:'未全量完成/无该节点' if x is None else f'{x:.3f}'
    lines=[f'# 状态与奖励单因素：共同{args.decisions}决策','',
        '| 臂 | 执行模式 | 完整成功 | 十二条时间/s | 同八条时间/s |',
        '|---|---|---:|---:|---:|']
    for r in records:
        count='无该节点' if r['successes'] is None else f"{r['successes']}/12"
        lines.append(f"| {r['arm']} | {r['mode']} | {count} | {fmt(r['full_time_s'])} | {fmt(r['line_time_s'])} |")
    lines += ['', '只在相同模式内将state/reward与base比较。失败前缀不计为快的完成；JSON完整保留共同成功路径集合、所有角点、运动速度及前瞻诊断。',
        '', '共同改善的方向性筛选：'+json.dumps(support,ensure_ascii=False)+
        '。这不等于形状验收、充分性证明、独立确认或自适应相对合理选距固定D的最终优势。',
        '', '60k为主比较预算，30k为曲线补充；全部所选模型另按已登记规则保留。没有单因素支持时不得自动启动组合或追加训练。']
    (stage/'单因素比较.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_saved_single_factor_checkpoint_analysis',decisions=args.decisions,
        saved_trace_files=traces,models=0,physics_ticks=0,training_decisions=0))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        f'状态/奖励单因素共同{args.decisions}节点已由保存评价分析；分别与共享base比较，固定D获得同样设计机会，失败与强v7保留。方向性筛选不替代完整形状、独立确认、合理选距或最终优势。按真实结果决定有界后继，不自动组合/扩预算；封存测试未用于选择，20–25页英文全文及逐图全PDF验收尚未完成。')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
