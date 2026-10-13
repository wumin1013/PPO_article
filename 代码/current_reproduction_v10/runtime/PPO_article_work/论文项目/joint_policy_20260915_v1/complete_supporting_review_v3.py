"""Consolidate saved evidence and the completed corner-figure visual review."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from study_io import BASE, register, receipt, write_json, record_progress, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    names = ['fixed0p8_distance_search_v1', 'fixed1p5_distance_search_v1',
             'fixed6_distance_search_v1', 'fixed8_distance_search_v1',
             'fixed12_distance_search_v1', 'joint_independent_exploration_learning_v1',
             'joint_gsde_exploration_learning_v1', 'fixed3_independent_exploration_learning_v1',
             'fixed3_gsde_exploration_learning_v1', 'joint_long_mixed_development_v6',
             'fixed3_long_mixed_development_v6']
    evidence = [BASE / p for p in [
        'refinement_final_analysis_v6/receipt.json', 'refinement_final_analysis_v6/selected_checkpoints.json',
        'refinement_later_analysis_v6/receipt.json', 'learned_macro_action_response_v6/receipt.json',
        'front_and_settings_proof_v4/visual_qa.json', 's2_figure_layer_v2/visual_qa.json',
        'nnc_saved_command_new_geometry_v1/receipt.json', 'smooth_training_integration_validation_v1/receipt.json',
        'v6_selected_corner_comparison_figure_v1/receipt.json', 'smooth_learning_probe_plan_v1.md',
        'long_transfer_diagnostic_v6_plan.md']]
    evidence += [BASE / n / 'registration.json' for n in names]
    stage = register('ongoing_research_snapshot_v3', [Path(__file__), *evidence],
                     dict(model_calls=0, physics_ticks=0, training_decisions=0),
                     'Consolidate completed v6 results, visual acceptance and bounded active studies without restarting anything.')
    figure = BASE / 'v6_selected_corner_comparison_figure_v1'
    qa = figure / 'visual_qa.json'
    if qa.exists():
        raise FileExistsError(qa)
    write_json(qa, dict(status='single_pdf_page_visually_checked_pass',
        pdf_sha256=sha(figure/'v6_selected_corner_comparison.pdf'),
        svg_sha256=sha(figure/'v6_selected_corner_comparison.svg'),
        preview='D:/PPO_article_joint_runs/v6_selected_corner_comparison_figure_v1/preview.png',
        panels=4, all_four_isolated_corners_shown=True, true_xy_ratio=True,
        labels_legible=True, overlapping_or_clipped_labels=False,
        selected_checkpoints={'joint':120000, 'fixed3':120000, 'fixed4p6':90000},
        final_manuscript_figure=False, goal_complete=False))
    observations = []
    for name in names:
        folder = BASE/name
        pointer = Path('D:/PPO_article_joint_runs/rolling_states')/name/'resume_pointer.json'
        validations = folder/'validations.json'
        rows = []
        if validations.exists():
            for item in read(validations)['validations']:
                results = item['results']
                count = sum(r['success'] for r in results)
                rows.append(dict(decisions=item['decisions'], tag=item['tag'],
                    successes=count, total=len(results),
                    complete_time_sum_s=sum(r['completion_time_s'] for r in results) if count==len(results) else None))
        observations.append(dict(stage=name, observed_utc=datetime.now(timezone.utc).isoformat(),
            pointer=read(pointer) if pointer.exists() else None,
            completed_receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else None,
            failure=read(folder/'failure.json') if (folder/'failure.json').exists() else None,
            validations=rows))
    selected = read(BASE/'refinement_final_analysis_v6/selected_checkpoints.json')
    write_json(stage/'results.json', dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        selected_v6=selected, active_stage_observations=observations,
        goal_status_last_actually_read='active', goal_complete=False,
        sealed_test_opened=False, new_models=0, new_physics_ticks=0))
    text = ['# 研究状态与下一步', '',
        '三组v6已全部完成120,000个新决策，不重启。按各组所有检查点选择：联合final为12/12、22.712 s，固定3 final为12/12、21.040 s，固定4.6的90k为12/12、23.051 s。固定4.6 final只有11/12，不能拿退化末次结果代替较好的90k。固定3尚在持续改善，充分收敛尚无证据。', '',
        '同八条直线路径，联合11.428 s、固定3为10.600 s，原S2为9.493 s。联合还没有达到目标要求的效率优势。150°联合final距尖顶约0.800 mm，固定3约0.921 mm，固定4.6所选90k约1.822 mm。任务允许合理深内切；允许、学到和整体更快是三个需要分别检验的事实。', '',
        '| 阶段 | 已保存决策 | 保存阶段 | 完成收据 |', '|---|---:|---|---|']
    for row in observations:
        pointer = row['pointer'] or {}
        text.append(f"| {row['stage']} | {pointer.get('decision', '未记录')} | {pointer.get('phase', '未记录')} | {'有' if row['completed_receipt'] else '无'} |")
    text += ['',
        '上表为读取时快照；不能仅凭旧PID或pointer判断此后仍运行。九组训练、两组长mixed评价的原预算继续，未新增或重复启动。四组探索均从同一v6固定3所选final初始化，60k新决策；固定七值搜索的3/4.6复用v6，其余五值各120k。初始策略等价时沿用已完成评价。', '',
        '动作响应诊断已完成：12个保存状态、84个局部宏，1680物理步中240为已登记基准重放，1440为新扰动；0模型/训练。D具有状态相关独立响应，不是全局冗余或收益证明。禁止重复该诊断。', '',
        '11页英文前半稿已逐页验收；前三图、S2五联图层、所选v6四角诊断图已验收。原14维NNC三份保存命令共3,034,807段完成新几何审核，均未通过完整任务；不作为新72维匹配消融。', '',
        '先收齐已登记研究的全部检查点与长mixed结果，比较完整性、时间、急弯形状和D变化。再据结果判断探索方式是否值得采用；不自动扩同一候选预算。仍需充分训练和独立种子、三类最终长路径、匹配无KCM、封存测试、Fig.4–9、20–25页全文及全PDF逐页验收。封存30条测试未运行。', '',
        '论文继续沿用指定提交稿的英文术语、句式和论述组织；正文围绕方法与研究结果，运行日志与恢复过程留在研究记录。目标保持active。']
    (stage/'状态与下一步.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    report = BASE/'逐图保留与修改报告_进行中.md'
    supplement = BASE/'探索与切角研究补充_20260915.md'
    backup = BASE/'before'/'supporting_review_v3'
    backup.mkdir(parents=True, exist_ok=False)
    for file in [report, supplement]:
        shutil.copy2(file, backup/file.name)
    content = report.read_text(encoding='utf-8')
    content = content.replace('v4已结束。v6截至90k，联合较好完整检查点为60k的22.952 s，固定3为90k的22.444 s，固定4.6为90k的23.051 s；七值搜索按预定预算继续，尚无稳定自适应优势',
        'v6三组全部完成。所选联合120k为22.712 s、固定3 120k为21.040 s、固定4.6 90k为23.051 s，均12/12；七值搜索与四组探索学习继续，尚无稳定自适应优势')
    content += '\n## v6完成与所选急弯图\n\n`v6_selected_corner_comparison_figure_v1`采用联合/固定3的120k和固定4.6的90k，全部四种单角均展示。SVG实际转换PDF，四个面板、真实XY比例、欧氏允差边界、图例与数值注释已目视通过，见同目录visual_qa.json。该图是开发诊断，不替代最终Fig.9。v6联合与固定3的长mixed评价已启动，只有完整审核收据才能更新Fig.4/5结论。最新汇总见ongoing_research_snapshot_v3。\n'
    report.write_text(content, encoding='utf-8')
    content = supplement.read_text(encoding='utf-8')
    content = content.replace('v6首个开发检查点', 'v6完整开发结果')
    content = content.replace('当前120k预算仍在运行，不能根据30k就宣布收敛或启动无限扩展。',
        'v6三组120k现已完成。固定3仍持续改善，不能根据早期优势或预定预算结束就宣布收敛。')
    content = content.replace('它尚未训练，也没有写成主方法；随后已完成下述有限探索探测。',
        '随后已完成有限探索探测，并启动下述四组配对学习；尚未采用为最终主方法。')
    content = content.replace('尚未启动gSDE训练。', '四组60k配对学习现已启动，学习效果以其实际检查点为准。')
    content += '\n## v6完整结果与当前判断\n\n三组v6均结束120,000个新决策。联合所选final为12/12、22.712 s；固定3所选final为12/12、21.040 s；固定4.6所选90k为12/12、23.051 s，final退化为11/12但保留。相同八条直线路径上，联合11.428 s、固定3为10.600 s、S2为9.493 s。早期联合的时间领先没有在最终检查点保持，因此目前不能宣称第三动作已经带来稳定优势。\n\n150°联合最近尖顶距离由60k的0.960 mm降至final约0.800 mm。所选四角图显示其轨迹仍靠近尖顶，并存在入弯、出弯的方向调整。固定4.6较远离尖顶，但局部速度更低。这些结果支持继续研究几何选择与速度调度的联合学习，而不支持把离尖顶距离单独视为性能。\n\n12个真实保存状态的三动作局部响应诊断已完成。D扰动的执行响应在去除转向与速度响应张成空间后仍有残差，其相对比例随状态变化；150°的接近、转弯附近、离开分别约0.116、0.109、0.366。该结果排除了这些局部状态下D完全无响应的说法，却不能证明全局不可替代或更高效率。无需再重复同一敏感性诊断。\n\n四组新学习共同使用v6固定3 final初始化，比较联合/固定3与独立/gSDE8探索，每组60,000个新决策；D初值3 mm，潜变量标准差0.03、0.10、0.20。七值固定距离搜索继续原预算。v6联合与固定3各一次长mixed评价也已启动，原S2为323.367 s，直接复用。中途经过时间不作为完整效率结果。\n\n11页英文前半稿已经逐页检查，S2五联图层及v6四角诊断图通过SVG到PDF与目视验收。最终仍缺稳定优势、独立种子和封存验证，以及后半篇实验结果、Fig.4–9与20–25页整篇PDF验收。\n'
    supplement.write_text(content, encoding='utf-8')
    receipt(stage, dict(status='completed_evidence_snapshot_and_visual_review',
        model_calls=0, physics_ticks=0, training_decisions=0, goal_complete=False,
        design_principles='Reuse evidence and one tracker (DRY); one bounded consolidation step (KISS/YAGNI); keep evidence observation separate from training and audit (single responsibility).'))
    record_progress('joint_completed_v6_and_active_followups_v3',
        [stage/'receipt.json', stage/'results.json', qa, report, supplement, *evidence[:9]],
        'v6三组120k全部结束：所选联合22.712 s、固定3为21.040 s、固定4.6 90k为23.051 s，均12/12；不重启。继续核实五组固定范围搜索、四组60k探索学习及两组v6长mixed的实际收据。11页稿/S2图层/v6四角图QA完成，动作响应和NNC审核完成。目标仍缺真实稳定优势、充分训练与独立种子、最终三类长路径、匹配消融、封存测试、Fig.4–9及20–25页全PDF验收。')
    print(stage/'状态与下一步.md')


if __name__ == '__main__':
    main()
