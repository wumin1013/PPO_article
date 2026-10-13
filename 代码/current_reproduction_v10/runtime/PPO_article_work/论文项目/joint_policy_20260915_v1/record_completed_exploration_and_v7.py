"""Publish verified research-update QA and the current bounded continuation."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from study_io import BASE, register, receipt, write_json, record_progress, sha


def main():
    paper = BASE/'research_update_pdf_v4'
    manifest = json.loads((paper/'build_manifest.json').read_text(encoding='utf-8'))
    qa = paper/'visual_qa.json'
    assert not qa.exists()
    write_json(qa, dict(status='all_seven_pdf_pages_visually_checked_pass', pages=7,
        pdf_sha256=sha(paper/'联合策略研究更新.pdf'), actual_file_size=(paper/'联合策略研究更新.pdf').stat().st_size,
        rendered_pages=[dict(page=i+1, path=p, sha256=sha(p), visual_pass=True)
                        for i,p in enumerate(manifest['rendered_pages'])],
        small_tables_kept_together=True, missing_glyphs=0, overfull_boxes=0,
        unresolved_references=0, overlaps_or_clipped_labels=False,
        remaining_non_layout_package_notices=['obsolete fixltx2e','xeCJK family redefinition','microtype footnote patch unavailable'],
        final_manuscript=False, goal_complete=False))
    write_json(BASE/'research_update_pdf_v3/visual_qa.json', dict(
        status='all_seven_pages_reviewed_pagination_revision_required',
        reason='The three-row long-path comparison table split across pages 4 and 5.',
        superseded_for_delivery_by='research_update_pdf_v4', old_pdf_preserved=True))
    names = ['joint_independent_refinement_v7', 'joint_gsde_refinement_v7',
             'fixed3_independent_refinement_v7', 'fixed3_gsde_refinement_v7']
    evidence = [BASE/p for p in [
        'fixed_distance_search_results_v1/receipt.json', 'fixed_distance_search_results_v1/results.json',
        'exploration_learning_analysis_30000_v1/receipt.json', 'exploration_learning_analysis_60000_v1/receipt.json',
        'exploration_learning_analysis_60000_v1/results.json', 'ppo_update_size_diagnosis_v2/receipt.json',
        'selected_turning_dynamics_v6/receipt.json', 'long_time_regions_v6/receipt.json',
        'joint_long_mixed_development_v6/receipt.json', 'fixed3_long_mixed_development_v6/receipt.json',
        'learning_design_v7.md', 'front_and_settings_proof_v4/visual_qa.json']]
    evidence += [BASE/n/'registration.json' for n in names]
    stage = register('ongoing_research_snapshot_v4', [Path(__file__), qa, *evidence],
        dict(models=0, physics_ticks=0, training_decisions=0),
        'Record completed fixed search, four-arm exploration, long evaluations, offline diagnosis, full report QA and four bounded v7 workers.')
    ids = dict(zip(names, [(93076,81146), (74748,88909), (33836,44171), (71936,44521)]))
    observed = []
    for name in names:
        path = Path('D:/PPO_article_joint_runs/rolling_states')/name/'resume_pointer.json'
        observed.append(dict(stage=name, observed_utc=datetime.now(timezone.utc).isoformat(),
            pointer=json.loads(path.read_text(encoding='utf-8')), observed_worker_pid=ids[name][0],
            exec_session_id=ids[name][1], new_decision_budget=360000, wall_seconds_budget=18000,
            current_failure_file_exists=(BASE/name/'failure.json').exists()))
    write_json(stage/'results.json', dict(v7_at_observation=observed,
        fixed_search_complete=True, selected_fixed_distance_mm=3.,
        selected_exploration_times_s={'joint_independent_60k':20.512,'joint_gsde_60k':20.657,
                                    'fixed3_independent_60k':20.598,'fixed3_gsde_30k':20.827},
        long_v6_times_s={'joint':438.506,'fixed3':405.570,'s2_reused':323.367},
        saved_buffer_diagnosis={'real_buffer_loads':4,'offline_ppo_updates':12,'actor_minibatches':278,'critic_minibatches':432,
                                'physics_ticks':0,'training_decisions':0},
        research_report_pages_visually_accepted=7, front_manuscript_pages_visually_accepted=11,
        sealed_test_opened=False, goal_complete=False))
    lines = ['# 目标续接：已完成搜索与v7细化', '',
        '七值固定D搜索全部结束，选择3 mm；0.8/1.5/3/4.6所选十二条时间26.252/23.227/21.040/23.051 s，均12/12；6/8/12各较好完成数9/5/4，不作全量时间排名。3/4.6复用v6，没有重复训练。', '',
        '四组60k探索全部结束：所选联合独立20.512 s、联合gSDE20.657 s、固定独立20.598 s、固定gSDE保留30k的20.827 s，均12/12；固定gSDE final为21.181 s，不以较弱final替代30k。联合独立同八条10.328 s，仍慢于S2的9.493 s；150°最近尖顶0.816 mm，深切仍未学好。', '',
        'v6两条新长mixed均完整成功：联合438.506 s、固定3为405.570 s；原S2为323.367 s。各通过1373道门、允差、公共V/A/J、端点与停稳。三者时间分区已经只读分析，联合约97.955 s时差发生在尖角15 mm邻域以外的剩余路径；不能归为纯直线或单部件因果。完成评价不重跑。', '',
        'PPO更新幅度诊断完成：四份59,999步原批、12次离线更新，0物理/新决策；原学习率各3/2/12/4 actor小批，3e-5各36，边际KL均小于0.009。原v1来源登记错误发生在模型和批加载之前，0更新，保留failure；v2完成后不重做。', '',
        'v7四组按learning_design_v7.md运行，共同从固定独立60k final初始化，实际初始12/12、20.598 s。Actor LR3e-5，其他PPO设置沿用；每组360k、18000秒，每60k评价，四组均处于同一既有学习谱系，不是独立完整种子。所有训练回合NPZ、决策日志、滚动状态在D盘。', '',
        '| 阶段 | 观察决策 | Worker PID | exec session |', '|---|---:|---:|---:|']
    for row in observed:
        lines.append(f"| {row['stage']} | {row['pointer']['decision']} | {row['observed_worker_pid']} | {row['exec_session_id']} |")
    lines += ['',
        '先读实际pointer/receipt并核Get-Process及绑定invocation，不按此旧表猜存活或重复启动。V7恢复仍须joint_compact_v5，已结束节点不恢复。完整七值搜索、四组60k、v6长mixed、动作响应、转向动态、时间分区和学习率诊断均只复用。', '',
        '七页《联合策略的过弯能力与效率》研究更新已全PDF逐页验收，入口research_update_pdf_v4/联合策略研究更新.pdf，SHA256='+sha(paper/'联合策略研究更新.pdf')+'。原15页报告、当前11页英文前半稿保留；最终20–25页、Fig.4–9、匹配无KCM、独立完整种子、最终三类长路径和封存30条测试仍未完成。', '',
        '最新三个工程问题：register只接受ROOT内来源，D盘指针另存路径/哈希绑定；LaTeX包装器在GBK和UTF8两种设置下均因混合输出解码失败，底层PDF已生成，后续用既有XeLaTeX并捕获bytes；中文不能整体放入nolinkurl，只将ASCII路径段放入URL命令。旧失败PDF保留。小表改为不可拆分的tabular，v4全页通过，不重编已验收版本。', '',
        '原则应用：复用已完成证据和唯一writer，避免重复实验；将只读诊断、训练和论文发布分开；只按新证据登记有限后继，未增加逐角执行规则。目标仍active，不因报告或前半稿完成而标完成。']
    (stage/'状态与下一步.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    report = BASE/'逐图保留与修改报告_进行中.md'
    supplement = BASE/'探索与切角研究补充_20260915.md'
    backup = BASE/'before'/'completed_exploration_and_v7'
    backup.mkdir(parents=True, exist_ok=False)
    for file in [report, supplement]:
        shutil.copy2(file, backup/file.name)
    with report.open('a', encoding='utf-8') as file:
        file.write('\n## 七值搜索与新研究报告\n\n七值固定前瞻和四组60k探索全部完成。所选固定D=3 mm；当前较强联合独立20.512 s、固定独立20.598 s仅属单种子开发结果。v6长mixed联合438.506 s、固定3为405.570 s均完整，但仍慢于S2的323.367 s；这些新来源可供开发图层使用，尚未冻结为最终Fig.4–9模型。\n\nresearch_update_pdf_v4/联合策略研究更新.pdf共7页、四角诊断图及全部表格已逐页验收。原v1/v2编译包装器编码失败、v1缺字与v3小表断页均保留；v4没有缺字、溢出或未解析引用。该报告不是20–25页投稿论文。v7四组较小学习率细化已启动，后续图件仍按真实所选模型更新。最新状态见ongoing_research_snapshot_v4。\n')
    with supplement.open('a', encoding='utf-8') as file:
        file.write('\n## 最新完整研究更新\n\n当前完整论述与后续方案见《联合策略研究更新.md》及research_update_pdf_v4/联合策略研究更新.pdf（7页，全页验收）。七值固定搜索、四组60k探索、两条v6长mixed和离线PPO更新幅度诊断均已完成；此前段落中的进行中表述是历史。七值选择3 mm；当前较强联合独立20.512 s与固定独立20.598 s的微小差异未证明稳定优势，150°深切目标仍未达到。v7四组各360k已登记并启动，具体范围见learning_design_v7.md，不重启旧完成阶段。\n')
    receipt(stage, dict(status='completed_report_qa_and_v7_continuation_snapshot', models=0,
        physics_ticks=0, training_decisions=0, goal_complete=False))
    record_progress('completed_exploration_and_v7_snapshot_v4',
        [stage/'receipt.json', stage/'results.json', qa, report, supplement, *evidence],
        '七值固定搜索与四组60k探索全部完成，选择D3，较强联合20.512 s/固定20.598 s未建立稳定优势；v6长mixed两组完整438.506/405.570 s仍慢于S2。PPO保存批诊断完成；v7四组各360k、actor LR3e-5、每60k验证已启动，先核其实际pointer/receipt，不重跑完成节点。最新续接ongoing_research_snapshot_v4。7页研究更新全PDF已验收，11页英文前半稿保留；独立种子、最终长路径/消融/封存测试、Fig.4–9及20–25页全文验收仍未完成。')
    print(stage/'状态与下一步.md')


if __name__ == '__main__':
    main()
