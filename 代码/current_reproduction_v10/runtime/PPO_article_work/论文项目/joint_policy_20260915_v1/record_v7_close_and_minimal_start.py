"""Record completed v7 evidence and actually started single-factor workers."""
import json
from pathlib import Path
from study_io import BASE,register,receipt,write_json,sha,record_progress


def main():
    figure=BASE/'v7_complete_curve_figure_v2'
    write_json(figure/'visual_qa.json',dict(pages=1,all_pages_inspected=True,accepted=True,
        pdf_sha256=sha(figure/'v7_complete_curve.pdf'),
        rendered_page='D:/PPO_article_joint_runs/v7_complete_curve_figure_v2/preview.png',
        checks=['All 28 audited nodes retained','Actual final x coordinate 359999',
                'S2 line and label clear of selected endpoints','Legends axes units and page edges readable',
                'No invented uncertainty; initial corner metric omitted'],
        manuscript_final_figure=False))
    groups=[BASE/f'minimal_{arm}_{mode}_pilot_v1'
            for arm in ('base','state','reward') for mode in ('joint','fixed')]
    sources=[Path(__file__),figure/'visual_qa.json',BASE/'v7_complete_curve_audit_v1/receipt.json',
             BASE/'v7_complete_curve_audit_v1/results.json']
    sources += [folder/name for folder in groups for name in ('registration.json','started.json','initial_completed.json')]
    stage=register('v7_closed_minimal_started_v1',sources,
        dict(model_calls=0,physics_ticks=0,training_decisions=0),
        'Record complete v7 audit, accepted curve and six actual bounded single-factor starts; no result inferred from registration.')
    observed=[]
    for folder in groups:
        start=json.loads((folder/'started.json').read_text(encoding='utf-8'))
        pointer=Path('D:/PPO_article_joint_runs/rolling_states')/folder.name/'resume_pointer.json'
        observed.append(dict(stage=folder.name,start=start,
            saved_pointer=json.loads(pointer.read_text(encoding='utf-8')),
            failure_exists=(folder/'failure.json').exists(),receipt_exists=(folder/'receipt.json').exists()))
    write_json(stage/'results.json',dict(observed=observed,v7_complete=True,
        single_factor_training_budget_upper=360000,sealed_test_opened=False,goal_complete=False))
    receipt(stage,dict(status='completed_readonly_launch_and_curve_record',workers_started=6,
        model_calls=0,physics_ticks=0,training_decisions=0,goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json',figure/'visual_qa.json'],
        '四组v7真实359999终点及全曲线审核完成；所选联合19.856/19.968 s，固定18.833/18.464 s，均12/12，联合优势仍未建立。曲线v2已整页验收，v1标注邻近问题保留。六组最小状态/奖励单因素先导已真实启动，每组60k/7200秒，共享base，22维和去势差分别与同模式72维原奖励比较；禁止重复启动或运行中改设置。下一共同30k/60k按保存数据分析；最终独立种子、合理选距、长路径、消融、封存测试和20–25页全文未完成。')
    print(stage.name)


if __name__=='__main__':
    main()
