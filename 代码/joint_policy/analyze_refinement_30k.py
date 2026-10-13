"""Analyze the first complete paired refinement checkpoint."""
import json
import numpy as np
from study_io import BASE,register,receipt,write_json
from summarize_conditioned_training import diagnostics


def main():
    names=['joint_refinement_training_v6','fixed3_refinement_training_v6','fixed4p6_refinement_training_v6']
    sources=[__file__,BASE/'summarize_conditioned_training.py',BASE/'s2_line_development_v2/results.json']
    for name in names:
        sources += [BASE/name/'validation_30000/summary.json']
        sources += [BASE/name/'validation_30000'/f'{i}.npz' for i in range(12)]
    stage=register('refinement_30000_analysis_v6',sources,dict(traces=36,models=0,physics_ticks=0),
        'Analyze all 12 development paths at the first paired refinement checkpoint, including every corner.')
    groups=[]
    for name in names:
        folder=BASE/name/'validation_30000'
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        cases=[]
        for i,result in enumerate(summary['results']):
            with np.load(folder/f'{i}.npz') as data:
                detail=diagnostics(data['reference'],data['rows'],data['actions'])
            cases.append(dict(result=result,diagnostics=detail))
        groups.append(dict(stage=name,cases=cases,successes=summary['successes'],
            complete_total_s=sum(r['result']['completion_time_s'] for r in cases) if summary['successes']==12 else None,
            first8_complete_s=sum(r['result']['completion_time_s'] for r in cases[:8]) if all(r['result']['success'] for r in cases[:8]) else None))
    write_json(stage/'results.json',dict(groups=groups,final_selection=False,training_continues=True))
    lines=['# v6首个检查点：完整配对与角点形状','',
        '| 方法 | 成功数 | 十二条总时间/s | 前八条直线拐角路径/s |', '|---|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['stage']} | {g['successes']}/12 | {g['complete_total_s']:.3f} | {g['first8_complete_s']:.3f} |")
    lines += ['', '同八条S2已完成总时间9.493 s，仅与上述前八条比较；十二条总时间含S2这批接口未评价的圆弧/闭合离散路径，不混用。','',
        '| 转角 | 自适应最近尖顶/mm | 固定3/mm | 固定4.6/mm | 自适应D中位数/mm |', '|---|---:|---:|---:|---:|']
    for i,angle in enumerate([50,95,125,150]):
        ds=[g['cases'][i]['diagnostics']['corners'][0]['minimum_vertex_distance_mm'] for g in groups]
        preview=groups[0]['cases'][i]['diagnostics']['preview_quantiles_mm'][1]
        lines.append(f'| {angle}° | {ds[0]:.3f} | {ds[1]:.3f} | {ds[2]:.3f} | {preview:.3f} |')
    lines += ['', '目前自适应在本检查点十二条总时间上优于两个固定距离，但仍须比较所有后续检查点和固定D搜索；不能据此宣称稳定或统计显著优势。更大的角点距离也不能替代完整时间和运动审核。此分析未改变任何运行中的源码或训练计划。']
    (stage/'阶段结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='paired_development_checkpoint_analyzed',traces=36,models=0,physics_ticks=0,goal_complete=False))
    print('\n'.join(lines))


if __name__=='__main__':main()
