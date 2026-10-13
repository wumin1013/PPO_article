"""Finalize observed report QA and append this evidence through the unified writer."""
from datetime import datetime, timezone
import json
from pathlib import Path
from study_io import BASE, write_json, sha, register, receipt, record_progress


def main():
    pdf_stage=BASE/'minimal_research_report_pdf_v2'
    manifest=json.loads((pdf_stage/'build_manifest.json').read_text(encoding='utf-8'))
    qa=dict(status='passed_full_page_visual_review',reviewed_utc=datetime.now(timezone.utc).isoformat(),
        pdf_sha256=manifest['pdf_sha256'],pages=[dict(page=i+1,path=p,sha256=sha(p),
        result='Read and visually inspected: no clipping, overlap, missing glyphs, or broken table rows.')
        for i,p in enumerate(manifest['rendered_pages'])],
        long_tables_continue_with_headers=True,overfull_boxes=0,missing_characters=0,
        undefined_references=0,nonblocking_package_warnings=['fixltx2e obsolete','xeCJK font redefinition','microtype footnote patch'],
        final_manuscript=False,goal_complete=False)
    write_json(pdf_stage/'visual_qa.json',qa)
    write_json(BASE/'minimal_research_report_pdf_v1/visual_qa.json',dict(
        status='reviewed_not_selected_due_to_four_small_table_overflows',pages_reviewed=list(range(1,11)),
        overfull_hbox_pt=3.64777,next_version='minimal_research_report_pdf_v2',goal_complete=False))
    sources=BASE/'minimal_state_sources_v1'
    included=[dict(file='state_abstraction_verified_title_candidate.pdf',
        title='Towards a Unified Theory of State Abstraction for MDPs',
        url='http://anytime.cs.umass.edu/aimath06/proceedings/P21.pdf',
        identity_verification='PDF title/authors plus saved OpenAlex record',
        read_scope='Abstract, sections 3.1–3.4 and related counterexamples; PDF page 4 definition 3 visually inspected'),
        dict(file='gae.pdf',title='High-Dimensional Continuous Control Using Generalized Advantage Estimation',
        url='https://arxiv.org/abs/1506.02438',
        identity_verification='arXiv identifier and PDF title/authors; v6 dated 2018-10-20, ICLR 2016 paper',
        read_scope='Sections 2–4; page 5 equations 16–20 visually inspected; page 6 value-shift relation read')]
    for item in included:
        item['sha256']=sha(sources/item['file'])
    write_json(sources/'curated_reading_manifest.json',dict(
        status='curation_of_completed_reading_not_a_new_download',included=included,
        excluded=[dict(file='state_abstraction.pdf',sha256=sha(sources/'state_abstraction.pdf'),
            actual_title='Fast Hierarchical Goal Schema Recognition',reason='Wrong initial proceedings file; not cited as Li/Walsh/Littman')],
        preserved_original_fetch_records=True,unread_material=['GAE full appendix','Tucker supplementary ZIP']))
    receipts=[BASE/name/'receipt.json' for name in ('minimal_information_credit_audit_v1',
        'compact_information_counterexamples_v1','event_observation_contract_v1',
        'event_horizon_information_probe_v2','v7_checkpoint_120000_analysis_v1',
        'v7_checkpoint_180000_analysis_v1','v7_checkpoint_240000_analysis_v1')]
    receipts += [BASE/'event_horizon_information_probe_v1/failure.json',pdf_stage/'receipt.json',
        pdf_stage/'visual_qa.json',sources/'curated_reading_manifest.json']
    stage=register('minimal_diagnostics_and_report_observation_v1',[Path(__file__),*receipts],
        dict(model_calls=0,physics_ticks=0,training_decisions=0),
        'Record completed diagnostics and ten-page visual acceptance without rerunning any experiment.')
    workers=[]
    for name in ('joint_independent_refinement_v7','joint_gsde_refinement_v7',
                 'fixed3_independent_refinement_v7','fixed3_gsde_refinement_v7'):
        pointer=Path('D:/PPO_article_joint_runs/rolling_states')/name/'resume_pointer.json'
        workers.append(dict(stage=name,pointer=json.loads(pointer.read_text(encoding='utf-8')),
            receipt_exists=(BASE/name/'receipt.json').exists(),failure_exists=(BASE/name/'failure.json').exists()))
    write_json(stage/'results.json',dict(observed_utc=datetime.now(timezone.utc).isoformat(),workers=workers,
        app_goal_observed_in_current_task=None,diagnostic_macros_total=15,diagnostic_ticks_total=300,
        diagnostic_training_updates=0,training_workers_separately_accounted=True,
        events16_not_advanced_to_training=True,next_state_candidate='spatial22, pending training admission',
        new_state_reward_training_started=False,report_pages=10,report_visual_pass=True,goal_complete=False))
    receipt(stage,dict(status='completed_evidence_observation',model_calls=0,physics_ticks=0,goal_complete=False))
    target=record_progress('minimal_diagnostics_and_report_v1',[stage/'receipt.json',stage/'results.json',*receipts],
        '已完成最小状态/奖励源码与保存数据诊断：稀疏18维和事件16维分别发现转移/即时执行信息别名，保留负结果；空间22维为待验证候选，27维不是下限。奖励成功排序正确，四批4601行的势函数与价值平移产生相同GAE，不能按奖励项数归因。v7四组已续接并推进，共同240k全部12/12，联合20.184/20.467s、固定19.132/18.932s；原设置与上限不改，下一共同300k只分析新保存评价。10页深入研究报告v2已全页QA，无溢出或缺字。尚未启动状态/奖励单因素新训练，待v7最终359999节点及全曲线后登记；固定方法同等设计/训练/选距机会。不打开封存测试。最终独立种子、长路径、匹配消融、20–25页英文及逐图全文验收未完成。')
    print(target)


if __name__=='__main__':
    main()
