"""Record the reviewed manuscript and scoped completion, preserving prior state."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from study_io import BASE, sha, write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    paper=BASE/'manuscript_joint_v1';case=BASE/'paper_evidence_review/natural_closed_case'
    latest=read(paper/'natural_pdf_qa/latest.json');qa=BASE/latest['directory']
    verify=read(paper/'natural_case_supplements/final_evidence_verification.json')
    assert verify['status']=='pass' and len(verify['hash_checks'])==132
    assert latest['pdf_sha256']=='e670f8314653574bcf9f3533e9cc0e98c9dca4473a08ebdff3528102cdc4b1d8'
    pdf=paper/'build_natural/PPO_article.pdf';assert sha(pdf)==latest['pdf_sha256']
    history=paper/'natural_case_supplements/before_delivery';history.mkdir(exist_ok=True)
    for p in (BASE/'当前执行状态.json',case/'execution_continuation.json',case/'execution_state_receipt.json',case/'method_freeze.json'):
        backup=history/p.name
        if not backup.exists():shutil.copy2(p,backup)
    final_pdf=paper/'PPO_article.pdf'
    if final_pdf.exists() and sha(final_pdf)!=sha(pdf):
        shutil.copy2(final_pdf,history/('PPO_article_'+sha(final_pdf)[:12]+'.pdf'))
    shutil.copy2(pdf,final_pdf)
    page_topics=['标题、摘要与研究边界','方法框架与动机','相关工作与正确的 S2 来源','欧氏走廊、向量约束与完整完成',
        '案例、34 角覆盖及静态口径','指标、99 主阈值及首次转向定义','72 维观察、支持修复及复合模型',
        '三动作与实际执行链','规则执行器、向量约束和制动','PPO/拟合/搜索来源与成本','S2 几何和向量时间律适配',
        '对照选择、预算差异及重复性口径','完整 XY/V/A/J 和停稳','三代表 XY 与真实速度形状表',
        '三列六行机制图','全部 34 角降速和全程巡航','固定 D 机会与唯一条件消融','完整效率、巡航、无虚假误差条',
        '0.5 ms 物理积分敏感性','长/短直线和圆弧分类、95/98 辅助及残余修正','单模型/单开发案例限制',
        '可复现材料、数据可用性与限定结论','九条参考文献和 arXiv 标识']
    visual=dict(pdf_sha256=sha(final_pdf),page_count=23,reviewed_pages=list(range(1,24)),
        visual_review_complete=True,review_method='Every page viewed as a rendered contact pair; nine standalone figures also viewed. '
        'After the final arXiv note change, pages1-22 were pixel-identical to the reviewed rendering and page23 was viewed separately.',
        prior_full_review_directory='manuscript_joint_v1/natural_pdf_qa/f66abba17da2',
        final_render_directory=str(qa.relative_to(BASE)),unresolved_visual_issues=[],
        page_checks=[dict(page=i,topic=topic,status='pass') for i,topic in enumerate(page_topics,1)],
        fixes=['LaTeX macro names containing digits','Title line-break syntax','Figure3 text/arrow overlaps',
               'Figure6 acute S2 curvature clipping','Figure5 onset labels crossing curves','Two orphan overflow pages',
               'S2 citation identity mismatch','PPO arXiv identifier missing from rendered bibliography'])
    write_json(qa/'visual_review.json',visual)
    report=['# PPO_article 全 PDF 逐页验收','',f"最终文件：`PPO_article.pdf`；23 页；SHA-256：`{sha(final_pdf)}`。",'',
        '结论：23 页全部实际渲染并目视查看，九幅独立主图同时检查。最终文献页补充 arXiv 编号后，第 1–22 页与已检查版逐像素一致，第 23 页重新单页查看。无未解决裁切、文字叠压、孤立短段或缺失图件。','',
        '自动核验：TeX Live 编译成功；未定义引用、字体缺字、overfull/underfull 警告均为 0；文本无页面越界；字体嵌入；132 项模型、路径、收据、图源绑定通过；9 条引文身份核验；正文数字与统一绑定一致。', '',
        'PDF 文字提取器将少数数学字形导出为 XML 非法控制字符，解析时仅在副本中替换；原 PDF 不受改动，公式已目视核查。文本提取不替代图面验收。','',
        '| 页码 | 检查内容 | 结果 |','|---|---|---|']
    report += [f'| {i} | {topic} | 通过 |' for i,topic in enumerate(page_topics,1)]
    report += ['', '已修复：数字宏名、标题换行、执行链文字重叠、S2 锐角曲率坐标裁切、首次转向标签遮线、两处跨页孤段、S2 引用错指及 PPO 标识漏显示。', '',
        '研究验收边界：完整成功、局部可观察协调与对强固定前瞻 1.007477% 的条件收益已成立；S2 更快，三独立训练种子与新封存测试未建立。24/25 页的先前渲染缓存保留为历史，当前页面应从 `natural_pdf_qa/latest.json` 定位。', '',
        '本文是研究稿交付，不含真实机床/连续伺服保证，也未执行投稿或公开发布。']
    (paper/'PDF逐页验收.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    fr=read(paper/'natural_case_figures/receipt.json')
    fr.update(visual_qa_complete=True,visual_qa_record=str((qa/'visual_review.json').relative_to(BASE)))
    write_json(paper/'natural_case_figures/receipt.json',fr)
    filelist=[final_pdf,paper/'PPO_article.tex',paper/'natural_references.bib',paper/'逐图说明与证据边界.md',paper/'PDF逐页验收.md']
    filelist+=list(paper.glob('natural_*.tex'))
    filelist+=list((paper/'natural_case_figures').glob('Figure_*.*'))
    filelist+=[paper/'natural_case_figures/figure_contract.json',paper/'natural_case_figures/receipt.json',qa/'visual_review.json',qa/'structural_audit.json']
    filelist+=list((paper/'natural_case_supplements').glob('*.csv'))
    filelist+=[paper/'natural_case_supplements'/n for n in ('citation_verification.json','引用核验.md','final_evidence_verification.json','derived_data_receipt.json')]
    filelist += [BASE/n for n in ('bind_natural_paper_results.py','natural_case_metrics.py','draw_natural_paper_figures.py',
        'prepare_natural_manuscript.py','compile_natural_manuscript.py','render_natural_pdf_qa.py','verify_natural_delivery.py','finalize_natural_delivery.py')]
    binding=read(case/'paper_result_binding.json')
    filelist += [case/'paper_result_binding.json',case/'reference.csv',case/'geometry.json',case/'geometry_reporting.json',case/'geometry_audit.json']
    for entry in binding['primary_methods'].values():filelist += [BASE/entry[k] for k in ('trace','result','metrics')]
    model=BASE/'canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt'
    filelist.append(model)
    freeze=read(case/'method_freeze.json')
    freeze.update(paper_delivery_complete=True,scoped_case_study_complete=True,goal_complete=True,
        completion_scope='Single-case single-composite-model manuscript and evidence; no general superiority or independent training repeatability claim.',
        remaining='No required work for scoped delivery. Publication administration and any broader confirmatory study require a separate scope.',
        final_pdf_sha256=sha(final_pdf))
    write_json(case/'method_freeze.json',freeze);filelist.append(case/'method_freeze.json')
    manifest=dict(completed_utc=datetime.now(timezone.utc).isoformat(),project='PPO_article',new_project_number=False,
        status='scoped_research_manuscript_delivered',page_count=23,main_figures=9,citations=9,
        model_and_path_scope=dict(paths=1,main_models=1,independent_end_to_end_training_replications=0),
        core_result=dict(joint_time_s=450.119,strong_fixed_time_s=454.7,s2_time_s=445.185,
            joint_gain_over_fixed_percent=100*(454.7-450.119)/454.7,joint_slower_than_s2_s=450.119-445.185),
        evidence_binding_sha256=sha(case/'paper_result_binding.json'),
        files={str(p.relative_to(BASE)):sha(p) for p in dict.fromkeys(filelist)},
        no_cleanup_commit_push_or_new_agents=True,real_machine_or_servo_claim=False)
    write_json(paper/'delivery_manifest.json',manifest)
    continuation=dict(updated_utc=datetime.now(timezone.utc).isoformat(),method_frozen=True,scoped_delivery_complete=True,
        manuscript=str(final_pdf.relative_to(BASE)),manuscript_sha256=sha(final_pdf),
        delivery_manifest=str((paper/'delivery_manifest.json').relative_to(BASE)),delivery_manifest_sha256=sha(paper/'delivery_manifest.json'),
        evidence_binding=str((case/'paper_result_binding.json').relative_to(BASE)),
        original_research_goal_preserved=True,independent_end_to_end_training_replications=0,
        supported_claim='Complete feasible coordination on the registered case, 1.007477% faster than the strengthened conditional fixed-preview control.',
        unsupported_claims=['faster than S2','general-path superiority','three independent PPO seeds','all-soft-metric dominance','industrial servo guarantee'],
        next_action='Preserve method and evidence. Use the delivered manuscript for author review and submission preparation; no automatic retraining or cleanup.',
        historical_limits_superseded_by_latest_user_authorization=True)
    write_json(case/'paper_delivery_continuation.json',continuation)
    (BASE/'PPO_article_论文交付与续接.md').write_text('''# PPO_article 论文交付与续接

沿用原项目与研究目标，没有新增编号。以本文件及 `paper_evidence_review/natural_closed_case/paper_delivery_continuation.json` 为本次执行的最新续接入口；20260919/20260920 文档和静态收据保留为历史，不把其“尚未运行”或“两轮总上限”误作当前状态。

## 已完成并冻结

同一自然闭合案例，主方法 450.119 s、强固定前瞻 D=4.6 为 454.700 s、S2 为 445.185 s，三者均完整有序覆盖、解析欧氏允差及向量 V/A/J 审核、制动和停稳通过。主方法相对固定快 4.581 s（1.007477%），相对 S2 慢 4.934 s。0.5 ms 执行检查完整成功 450.118 s；决策仍为 20 ms。

主模型权重未变，新增适配只截断剩余距离观察；历史 PPO、离线拟合及参数搜索分别披露。固定法获得七值选距、十个附加残差候选、34 角选择检查；这是共同骨干条件比较，不是三独立 PPO 重训。历史强固定 PPO 模型也保留完整比较。

## 交付入口

- `manuscript_joint_v1/PPO_article.pdf`：23 页英文稿，含九幅主图。
- `manuscript_joint_v1/逐图说明与证据边界.md`：九图主张、来源、解释和限制。
- `manuscript_joint_v1/PDF逐页验收.md`：全 23 页实际渲染验收与修复记录。
- `manuscript_joint_v1/natural_case_supplements/引用核验.md`：9 条正文引用身份核验及全文支持层级。
- `manuscript_joint_v1/natural_case_supplements/all_34_corners_three_methods.csv`：34×3 完整角点统计。
- `manuscript_joint_v1/delivery_manifest.json`：稿件、图源、模型、轨迹和审核的 SHA-256 清单。

## 不扩大结论

这是单个开发案例和单个复合主模型证据；三独立端到端 PPO 重复尚未建立，S2 更快，原长路径失败保留，原封存数据已接触且本次未用于选择。仅一项条件前瞻消融；没有转向/进给重训消融。数字执行审核不等于连续伺服或工业交付。

目前限定案例论文交付完成，普遍优越性和独立训练重复性仍未证实。冻结后不继续围绕软指标训练；下一步仅作作者审阅和投稿准备。更广的确认性研究另立范围，不能把既有数据重新标为未见测试。所有中间文件保留，未经新指令不清理、不提交、不推送。

KISS/YAGNI：满足限定证据后停训。DRY：正文、表格和图件统一绑定。SOLID：训练、执行、数据派生、绘图、引文和 PDF 验收职责分开。主要困难是长路径观察分布、强固定机会、引用错指和版面溢出，均以有限修复与可追溯记录处理。
''',encoding='utf-8')
    print('DELIVERY_RECORDED',final_pdf,sha(final_pdf))


if __name__=='__main__':main()
