"""Record completed visual checks and current supporting research evidence."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from study_io import BASE,write_json,record_progress,sha


def main():
    stage=BASE/'v5_corner_comparison_figure_v1'
    qa=stage/'visual_qa.json'
    assert not qa.exists()
    write_json(qa,dict(status='single_pdf_page_visually_checked_pass',
        pdf_sha256=sha(stage/'v5_corner_comparison.pdf'),svg_sha256=sha(stage/'v5_corner_comparison.svg'),
        preview='D:/PPO_article_joint_runs/v5_corner_comparison_figure_v1/preview.png',
        panels=4,all_four_development_corners_shown=True,overlaps_or_clipped_data_labels=False,
        true_xy_ratio=True,final_manuscript_figure=False,goal_complete=False))
    backup=BASE/'before'/('supporting_records_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
    backup.mkdir(parents=True,exist_ok=False)
    report=BASE/'逐图保留与修改报告_进行中.md'
    addition=BASE/'探索与切角研究补充_20260915.md'
    for file in [report,addition]:shutil.copy2(file,backup/file.name)
    with report.open('a',encoding='utf-8') as file:
        file.write('\n新增开发诊断图`v5_corner_comparison_figure_v1/v5_corner_comparison.svg/pdf`比较50°、95°、125°、150°全部四种单角，显示三组真实完整轨迹及最近尖顶距离/局部速度。已实际SVG转PDF并目视检查，详见同目录`visual_qa.json`；它不是最终Fig.9，也没有自适应全面占优的含义。\n')
    text=addition.read_text(encoding='utf-8')
    old='当前尚未实现或执行这些候选，也没有把它们写成主方法。'
    assert old in text
    text=text.replace(old,'当前已实现一个归一化策略特征的gSDE候选，并通过边际分布、三动作梯度、固定头屏蔽及跨重采样状态恢复的合成检查。它尚未训练或执行任何物理轨迹，也没有写成主方法。合成检查采用每8个策略决策重采样一次，仅表示实现检查通过，不表示该间隔最优。')
    text+='\n## 长路径诊断的新增观察\n\nv5联合在mixed_lengths_large的91.750 s处越过欧氏允差边界，公共V/A/J通过；完整时间为空，不能参与成功时间排名。失败处是150°角后面的75°连接，实际速度约33.284 mm/s。保存的失败前状态能够逐位复现原动作。剩余长度特征虽然超出训练范围（0.8522对训练最大0.3158），单独改变此分量时速度意图变化不到0.5 mm/s、转向约0.23°，因此不足以支持简单裁剪此输入就能修复失败的推断。详见`long_observation_diagnosis_v5_binding_fix/长路径观测诊断.md`。\n\n长评价的首宏保存曾误用训练buffer编解码器，两组各已执行20物理步的命令保留。修正后从保存命令核对恢复：20次引导重建及位置更新逐位一致，没有重复模型动作或物理前缀。该恢复没有原始RNG快照，评价策略是确定性的，恢复记录明确区分原检查点与重建状态。原失败目录和费用保留。\n'
    addition.write_text(text,encoding='utf-8')
    sources=[qa,BASE/'refinement_30000_analysis_v6/receipt.json',
        BASE/'joint_long_mixed_storage_recovery_v5/receipt.json',
        BASE/'long_observation_diagnosis_v5_binding_fix/receipt.json',
        BASE/'smooth_exploration_candidate_validation_v1/receipt.json',
        BASE/'temporal_exploration_fulltext_v1/receipt.json',addition]
    record_progress('v6_first_checkpoint_and_supporting_studies_v1',sources,
        'v6三组原定120k训练继续；30k均12/12，联合23.620 s、固定3为25.373 s、固定4.6为24.045 s；同八条拐角仍慢于S2，深内切尚未实现。v5联合长mixed在91.750 s越界，固定3长评价以实际收据为准。9页前半稿与前三图、v5四角诊断图已目视验收。平滑探索候选仅通过合成检查，尚未训练或物理评价。合理固定D、多种子、长路径、封存测试及20–25页论文全PDF仍未完成。')
    print('Supporting evidence recorded; goal remains active.')


if __name__=='__main__':main()
