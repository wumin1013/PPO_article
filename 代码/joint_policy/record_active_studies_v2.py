"""Keep bounded studies and negative results visible during ongoing training."""
import json
import shutil
from datetime import datetime,timezone
from study_io import BASE,register,receipt,write_json,record_progress


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    grid=['fixed0p8_distance_search_v1','fixed1p5_distance_search_v1','fixed6_distance_search_v1',
          'fixed8_distance_search_v1','fixed12_distance_search_v1']
    refinements=['joint_refinement_training_v6','fixed3_refinement_training_v6','fixed4p6_refinement_training_v6']
    source_paths=[BASE/'smooth_exploration_physical_probe_v1/receipt.json',BASE/'smooth_exploration_physical_probe_v1/results.json',
        BASE/'joint_curriculum_training_v4/receipt.json',BASE/'fixed4p6_curriculum_training_v4/receipt.json',
        BASE/'fixed3_curriculum_training_v4/failure.json',BASE/'joint_long_mixed_storage_recovery_v5/receipt.json',
        BASE/'fixed3_long_mixed_storage_recovery_v5/receipt.json',BASE/'fixed3_long_mixed_storage_recovery_v5/evaluation.json',
        BASE/'nnc_saved_command_new_geometry_v1/registration.json',BASE/'fixed_distance_search_v1.md']
    source_paths += [BASE/n/'registration.json' for n in grid]
    source_paths += [BASE/n/'validation_60000/summary.json' for n in refinements]
    stage=register('ongoing_research_snapshot_v2',[__file__,*source_paths],dict(models=0,physics_ticks=0),
        'Observe completed supporting studies and registered remaining training; do not replay experiments.')
    probe=read(BASE/'smooth_exploration_physical_probe_v1/results.json')['results']
    groups=[]
    for name in ['independent','gsde8']:
        rows=[r for r in probe if r['sampler']==name]
        groups.append(dict(sampler=name,runs=len(rows),successes=sum(r['success'] for r in rows),
            equal_run_mean_absolute_D_step=sum(r['mean_absolute_D_step'] for r in rows)/len(rows)))
    validations=[]
    for name in refinements:
        s=read(BASE/name/'validation_60000/summary.json')
        validations.append(dict(stage=name,successes=s['successes'],
            complete_time_s=sum(r['completion_time_s'] for r in s['results']) if s['successes']==12 else None))
    candidates=[]
    for name in grid:
        s=read(BASE/name/'initial_validation/summary.json')
        reg=read(BASE/name/'registration.json')
        candidates.append(dict(stage=name,distance=reg['budget']['fixed_d'],initial_successes=s['successes'],
            initial_complete_time_s=sum(r['completion_time_s'] for r in s['results']) if s['successes']==12 else None,
            training_decisions_budget=120000))
    value=dict(observed_utc=datetime.now(timezone.utc).isoformat(),v6_60000=validations,
        exploration_probe=groups,additional_fixed_candidates=candidates,
        v4_completed={'joint_final_successes':10,'fixed4p6_final_successes':10,'fixed3_exception_preserved':True},
        long_v5={'joint_success':False,'joint_prefix_s':91.750,'fixed3_success':True,'fixed3_complete_s':534.338,'s2_reused_complete_s':323.367},
        nnc_audit='registered_and_running_at_observation; actual receipt takes precedence',
        gSDE_training_started=False,goal_complete=False)
    write_json(stage/'results.json',value)
    lines=['# 进行中研究状态与下一步','',
        'v4联合与固定4.6已各完成120,000步，final均10/12；固定3的原制动候选异常保留，未恢复。已结束阶段不重启。','',
        'v6三组继续原120k预算。60k联合12/12、22.952 s，固定3为12/12、23.880 s，固定4.6为11/12；后者30k完整24.045 s仍参与检查点选择。','',
        'v5长mixed：联合在91.750 s越界，完整时间为空；固定3完整534.338 s，仍慢于S2已保存的323.367 s。两组首20 ms存储错误均保留，并在不重复物理前缀的情况下重建状态续接完成。','',
        '| 探索方式 | 完整成功 | 各运行平均D逐步变化的均值/mm |','|---|---:|---:|']
    for g in groups:lines.append(f"| {g['sampler']} | {g['successes']}/{g['runs']} | {g['equal_run_mean_absolute_D_step']:.4f} |")
    lines += ['', '平滑探索降低动作抖动，但本探测的完整成功反而减少；它可能改变探索到的行为，不能直接等同于训练后性能改善。当前没有启动gSDE训练，不据平滑性直接替换主方法。','',
        '| 新固定D/mm | 初始化成功数 | 新决策预算 |','|---|---:|---:|']
    for c in candidates:lines.append(f"| {c['distance']} | {c['initial_successes']}/12 | {c['training_decisions_budget']} |")
    lines += ['', '完整固定集合为0.8、1.5、3、4.6、6、8、12 mm。五个新增值继续其已登记预算，3/4.6直接复用v6相同条件。不能以初始化失败否定后续学习，也不能把末次退化结果当固定对照。', '',
        '下一步先收齐v6全部检查点与固定范围搜索，判断效率、急弯形状、可行率及学习曲线。NNC只对已保存命令补新几何审核，用于原图保留，不重新执行其模型。合理固定D、充分训练/独立种子、最终联合长路径、匹配无KCM和30条封存测试仍待完成。前三图及9页前半稿已有全页QA；完整20–25页和Fig.4–9仍不能标完成。']
    (stage/'状态与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=BASE/'探索与切角研究补充_20260915.md'
    backup=BASE/'before'/('report_update_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False);shutil.copy2(report,backup/report.name)
    text=report.read_text(encoding='utf-8')
    text=text.replace('它尚未训练或执行任何物理轨迹，也没有写成主方法。','它尚未训练，也没有写成主方法；随后已完成下述有限探索探测。')
    text+='\n## 后续有限探测与当前训练\n\n36次探索探测已完成，固定相同策略均值和潜变量幅度，使用六条既有训练路径及三个采样种子。独立探索完整15/18，gSDE每8步重采样完整12/18；D逐步变化的逐运行均值分别1.4652和0.2964 mm。平滑性改善没有同时改善该探测的完成率，不能宣传成性能提升。尚未启动gSDE训练。\n\nv6的60k检查点：联合12/12、22.952 s，固定3为12/12、23.880 s，固定4.6为11/12；所有较好固定检查点保留。v4两组120k已结束且均未达到12/12，不恢复。完整固定D集合已预定为0.8、1.5、3、4.6、6、8、12 mm，补五组与v6同初始化/预算的训练，不按结果挑选较弱距离。固定3的v5长mixed评价已完整成功534.338 s，仍慢于常规S2的323.367 s。\n'
    report.write_text(text,encoding='utf-8')
    receipt(stage,dict(status='supporting_results_and_active_studies_recorded',models=0,physics_ticks=0,goal_complete=False))
    record_progress('fixed_grid_and_v4_closed_snapshot_v1',[stage/'receipt.json',report],
        'v4联合/固定4.6已完成120k且final10/12，固定3原异常保留；不重启。v6三组继续原120k，60k联合12/12、22.952 s，固定3为23.880 s，固定4.6为11/12且较好30k结果保留。新增固定0.8/1.5/6/8/12各120k已启动，同条件补完整七值搜索。gSDE36次探索探测完成但成功率未改善，尚未训练。NNC仅补原命令新几何审核。目标仍缺稳定优势、充分训练/多种子、最终长路径、消融、封存测试和20–25页全PDF。')
    print('\n'.join(lines))


if __name__=='__main__':main()
