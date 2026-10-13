"""Record inspected corner figures and close the bounded exploration candidate."""
import json
import shutil
from pathlib import Path
from study_io import BASE, register, receipt, write_json, record_progress, sha


def main():
    figure=BASE/'conditional_exploration_corners_20260916'
    analysis=BASE/'conditional_exploration_results_20260916'
    credit=BASE/'conditional_credit_geometry_20260916'
    previews=[figure/'conditional_corners.png',
              Path('D:/PPO_article_joint_runs/conditional_exploration_corners_20260916/preview.png')]
    assert not (figure/'visual_qa.json').exists()
    write_json(figure/'visual_qa.json',dict(accepted=True,
        inspection='Both complete original PNG and rendered SVG-derived PDF preview were directly inspected in this turn.',
        previews=[dict(path=str(p),sha256=sha(p)) for p in previews],
        pdf_sha256=sha(figure/'conditional_corners.pdf'),svg_sha256=sha(figure/'conditional_corners.svg'),
        checks=['Four corners shown; equal XY aspect; 1.25 mm Euclidean band.',
                'Legend, labels, axes and all twelve metric rows legible with no clipping or overlaps.',
                'Outside approach and shallow late turning remain visible; figure does not establish intended inner transition.'],
        scope='Common 60k development endpoints, including reused fixed independent control; not final manuscript figure or independent test.',
        final_submission=False,goal_complete=False))
    reports=[BASE/'最小充分状态与简洁奖励研究报告_进行中.md',BASE/'逐图保留与修改报告_进行中.md']
    stage=register('conditional_results_recorded_20260916',[Path(__file__),analysis/'receipt.json',
        analysis/'results.json',credit/'receipt.json',credit/'results.json',figure/'visual_qa.json',*reports],
        dict(model_calls=0,physics_ticks=0,training_decisions=0),
        'Bind completed 60k evidence and actual visual inspection, preserve reports before appending, stop candidate under its gate.')
    before=stage/'before';before.mkdir()
    for p in reports:shutil.copy2(p,before/p.name)
    content='''

## 20260916条件探索完成：部分改善，整体采用门槛未通过

两组各60,000新决策现已完成，进程正常退出，最终模型SHA、注册、完整收据、0/60k选择及结束指针一致。新24条开发路径全部完整成功；联合总时间20.236 s，相比原独立60k的21.140 s减少0.904 s（4.28%），也优于共同初值20.598 s。固定条件探索20.404 s，反而慢于原固定独立60k的20.018 s。两者均选60k。预注册整体采用条件未通过，停止此候选，不自动扩训；联合19.856 s、固定18.464 s的强v7及S2继续保留。

联合D≥4.6 mm最长连续19决策，D≥6 mm最长4决策，仍未访问8 mm。训练探索持续性增加没有转化为显著自适应前瞻：确定性开发D中位数约2.94–2.96 mm。95/125/150度入弯前10 mm的有符号偏移分别−0.035/−0.112/−0.168 mm，负值为这三条左转路径的外侧。四角图中局部速度提高，但外侧入弯和接近尖顶才形成转向仍可见；150度距离还从0.947降至0.833 mm。不能把本次时间改善归因于期望内侧形状已学会。远离尖角的可用开发区间速度中位数仍约78–82 mm/s，这一几何筛选不等于已证明具备高速巡航动态条件。

本轮新增训练合计120,000决策、2,531,099物理tick；24条终点评价共1,921次actor决策、40,640物理tick。两组墙钟1066.9/1073.8秒，均在7200秒上限内。保存日志分析120,000唯一行，复用覆盖函数两遍共解压240,000行；日志条件logp误差均小于1.4e−6。旧13组训练未重跑。

新增两份冻结59,999批共1,214行，2模型、4批actor前向、2次autograd，无物理或优化器更新。恒定潜动作均值平移下，连续步条件Fisher相对同边际方差独立高斯仅(1−ρ)/(1+ρ)=0.06242；这是该特定方向的解析敏感度，不是整个网络学习率或效率收益。联合批进给bias的GAE上升信号+0.215，使用原GAE尺度的完整后缀回报为+0.327；固定分别+0.061/+0.011。两者直线子组仍指向降速，不能用一次改λ或统一增大学习率直接解决。critic可能缺少噪声上下文，但现有诊断未证明这是主要原因，因此不据此添加状态或新价值头。

接下来回到动作序列的可达性与回报：用少量预注册的持续转向/进给/D组合干预，检查更内侧、更快的完整后缀是否在当前动作接口下可行，以及单动作和协同改变有何区别。该探针只做离线机制诊断，不把逐角干预部署成在线模板、不增加形状奖励、不打开封存测试。状态无维数下限，奖励仍逐项审查必要性。

本次复用冻结运行循环、完整审核和已验收图型，采样/分析/记录各自承担单一职责（KISS、DRY、SOLID）；不自动加训练、价值头或几何控制模块（YAGNI）。挑战是区分持续探索覆盖与真正学得行为，现用真实轨迹、确定性D及完整时间共同核对。21页中文v10、23页英文v13继续保留其冻结工作稿身份；此次新证据尚未写入这些旧PDF，最终论文与研究目标均未完成。
'''
    with reports[0].open('a',encoding='utf-8') as out:out.write(content)
    with reports[1].open('a',encoding='utf-8') as out:out.write('''

## 20260916：条件探索四角图

新增`conditional_exploration_corners_20260916`，沿用用户已确认的等比例四角图型，全部50/95/125/150度及独立联合/条件联合/独立固定3 mm三线均保留。12份原保存轨迹，0新评价；输出可编辑SVG、实际SVG转换PDF、PNG和source_data.csv。原PNG与PDF渲染预览已分别整页直接查看，图例、坐标和12行数据无裁切或重叠，QA绑定文件SHA。

联合四角速度增加，95/125/150度入弯仍偏外侧，150度最近尖顶距离反而减小；图形不支持提前内侧形状已经实现。20.236 s只是共同60k开发节点结果，仍慢于强v7联合19.856 s、强固定18.464 s。固定条件探索变慢导致整体采用门槛未通过，按规则停止。该图仅为开发诊断，最终Fig.4–9继续待定；原图和23页英文v13、21页中文v10均不覆盖，不冒称旧PDF已包含本次结果。
''')
    receipt(stage,dict(status='conditional_candidate_closed_with_partial_joint_improvement',
        candidate_adopted=False,automatic_extension=False,visual_qa_passed=True,
        report_sha256={p.name:sha(p) for p in reports},goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',analysis/'receipt.json',credit/'receipt.json',figure/'visual_qa.json'],
        '条件探索联合/固定各60k真实完成，24/24完整；20.236/20.404秒，相对原同模式21.140/20.018，整体采用门槛未通过，停止此候选不自动扩训。四角PNG和SVG转PDF整页已实查，外侧入弯与浅切仍在。两份冻结批1214行的条件信用诊断完成，无新物理/更新。下一步仅以有预算的持续三动作组合后缀探针区分可达性和信用，不部署模板、不加形状奖。旧13组、强v7/S2和全部负结果保留。云端总量未验收，最终研究/论文未完成。')
    print('Conditional results, figure QA, reports and unified progress recorded.')


if __name__=='__main__':main()
