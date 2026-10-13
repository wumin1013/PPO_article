"""Preserve tested reachability, rejected noise candidate and newer storage evidence."""
import shutil
from pathlib import Path
from probe_coordinated_suffix import read
from study_io import BASE,ROOT,register,receipt,write_json,sha,record_progress


def main():
    figure=BASE/'coordinated_suffix_figures_20260916'
    assert not (figure/'visual_qa.json').exists()
    previews=[figure/'suffix_probes.png',Path('D:/PPO_article_joint_runs/coordinated_suffix_figures_20260916/preview.png')]
    write_json(figure/'visual_qa.json',dict(accepted=True,
        inspection='Original PNG and full SVG-derived PDF preview both directly inspected in this turn.',
        checks=['All three geometries and original actor/2-degree/8-degree intervention curves shown.',
            'Equal XY scale, 1.25 mm Euclidean band, nine metric rows and explanatory panel legible.',
            'No clipped or overlapping labels; probes explicitly identified as counterfactuals.',
            '95/125 degree positive approach median and still slightly negative 150 degree median accurately shown.'],
        previews=[dict(path=str(p),sha256=sha(p)) for p in previews],
        pdf_sha256=sha(figure/'suffix_probes.pdf'),svg_sha256=sha(figure/'suffix_probes.svg'),final_submission=False,goal_complete=False))
    names=['coordinated_suffix_probe_20260916','inner_approach_amplitude_probe_20260916',
           'coordinated_exploration_coverage_20260916','student_exploration_validation_20260916',
           'student_exploration_physical_probe_20260916']
    reports=[BASE/'最小充分状态与简洁奖励研究报告_进行中.md',BASE/'逐图保留与修改报告_进行中.md']
    stage=register('suffix_findings_recorded_20260916',[Path(__file__),figure/'visual_qa.json',*reports,
        *[BASE/n/f for n in names for f in ('receipt.json','results.json')]],
        dict(model_calls=0,physics_ticks=0,training_updates=0),
        'Record completed finite approach/coverage tests and rejected Student candidate; reconcile newer complete cloud inventory against stale queued snapshot.')
    before=stage/'before';before.mkdir()
    for p in reports:shutil.copy2(p,before/p.name)
    audit=Path('D:/PPO_storage_cleanup_20260915')
    files=['cloud_final_verified_inventory.json','cloud_inventory_checkpoint.json','cloud_cleanup_current_status.json',
           'cloud_obsolete_cache_unlock_receipt.json','cloud_core_package_receipt.json','cloud_core_upload_receipts.jsonl']
    sources=[dict(path=str(audit/n),sha256=sha(audit/n)) for n in files]
    latest=read(audit/files[0]);old=read(audit/files[1])
    assert latest['time']>old['time'] and latest['file_count']==len(latest['files'])==238
    assert latest['directories_read']==len(latest['directories'])==17
    assert sum(r['bytes'] for r in latest['files'])==latest['bytes']==2394877183
    assert len(latest['old_nested_remaining_files'])==1 and latest['old_nested_remaining_files'][0]['bytes']==0
    write_json(stage/'storage_evidence_reconciliation.json',dict(sources=sources,
        newer_complete_inventory_time=latest['time'],verified_snapshot_bytes=latest['bytes'],files=238,directories=17,
        scope='PPO cloud only; account total not measured. This is the 03:29 saved complete server inventory, not a fresh network measurement.',
        stale_inventory_time=old['time'],stale_queued_count=len(old['queued']),
        correction='The 02:59 queue is historical before subsequent deletion; do not resume its 2437 directories as if current cloud content remained unenumerated.',
        unresolved=latest['old_nested_remaining_files'],lock_status='Prior DELETE/UNLOCK 409 FileBeingLocked; no repeated unlock attempt in this stage.',
        storage_target_at_verified_snapshot=True,all_cleanup_complete=False,new_deletions=0))
    text='''

## 20260916动作配合机制：可行改善与实际探索之间仍有缺口

先登记三条公开开发角95/125/150度，在原条件联合60k到达入弯前约10 mm的真实状态分支；前缀物理直接复用，只重建引导和有序门记忆并逐位核对。转向+2°、进给+5 mm/s、D+2 mm的七个非零组合各持续8决策，随后恢复原actor至完整制动停稳。21个新后缀全部合规，三动作组合分别缩短40/40/60 ms。较强内切不能仅用最近尖顶距离表述，三者入弯段中位有符号y仍为负。

基于这一结果，仅再登记一次更大转向幅度：相同组合中的转向改为+8°，其余不变，新增3个后缀全部合规。结果如下，不能作为已学会或最优策略：

| 转角 | 原actor时间/s | +8°协同干预时间/s | 入弯段中位y：原→干预/mm | 最近尖顶：原→干预/mm |
|---|---:|---:|---:|---:|
| 95° | 0.833 | 0.793 | −0.207→+0.191 | 0.280→1.569 |
| 125° | 0.873 | 0.813 | −0.351→+0.047 | 0.635→1.720 |
| 150° | 0.893 | 0.853 | −0.360→−0.045 | 0.833→1.744 |

正y为这些左转路径的内侧。150°的+8°比+2°组合还慢20 ms，说明更大转向并非处处更快。24个后缀共12,332个新物理tick、511次actor调用，前缀重建1,960tick，无训练更新。成功后缀的实际回报差与负完整时间差数值一致，现有时间目标能认可这些有限改善；没有增加形状奖励或在线逐角规则。

原条件高斯训练日志再查60,000行，180条已结束单角训练轨迹，扣除跨PPO参数更新无法精确重建均值的1,235行。在2,299个有效入弯决策中，同时满足向内转向增量≥2°、进给增量≥5 mm/s、D增量≥2 mm仅出现1步，最长1步；≥8°组合为0步，两种均无连续8步。这里是实际样本覆盖，不是连续动作序列的概率估计，也不排除其他改善方式。

据此先检验共享尺度的多元Student-t(ν=5)条件创新噪声。其三动作尾部幅度相关，保持对称方向和与原AR高斯相同的条件协方差；历史条件均值、actor、状态、奖励及物理规则不变。SciPy密度、解析协方差及真实bias梯度检查通过，最大密度误差1.65e−6。随后固定同一actor、16条预选训练几何和相同种子规则，做高斯/Student各16次物理探索，共2,494次策略决策、52,837tick，无训练更新。两组均16/16完整，但都没有目标协同组合的连续8步，因此未通过预注册推进条件，停止该噪声候选，不启动新训练。小样本未出现事件不能证明理论概率为零。

当前较具体的差距是：可行且获时间奖励的协同序列存在，但当前探索没有持续访问它们；重尾噪声的这一小预算方案也未解决。下一轮应检查有时间结构的动作表示及其正确的序列信用，先验证采样分布/似然/回报归因，再登记训练。不能把当前探针变成逐角在线控制模板，不能原样扩大噪声或重复60k训练。状态无预设维数下限；此次未更改状态或奖励，相关旧负结果继续保留。

图`coordinated_suffix_figures_20260916`的原PNG和SVG转PDF渲染均已直接整页查看。三种真实曲线、全部三角、允差带及诊断身份清楚，未替代最终稿Fig.4–9。代码复用原actor、物理审核和前缀重建，采样器/探针/记录职责分离；没有为未通过的候选扩建训练流水线（KISS、DRY、SOLID、YAGNI）。NumPy 2二维cross弃用提示出现在只读覆盖统计中，未影响本次结果，冻结来源保留。

### 存储状态纠正

新核到03:29的完整云端清单，比02:59的2437目录排队快照更新：17目录、238文件、2,394,877,183 bytes（2.395 GB），仅PPO范围，未测整个账户总量。14核心包和manifest已有回读SHA验证。旧排队清单为后续删除前的历史，不能继续当作当前云端尚有2437目录未核。仍有零字节`基于强化学习的轨迹平滑/_ipdf2/edge_probe/lockfile`，此前删除及解锁返回409 FileBeingLocked；本阶段未重复网络操作或删除，故“目标体积已在该完整快照达到”和“残留清理全部完成”分开记录。上述新证据纠正本报告较早的云端总量未验收表述，历史正文保留。
'''
    with reports[0].open('a',encoding='utf-8') as out:out.write(text)
    with reports[1].open('a',encoding='utf-8') as out:out.write('''

## 20260916：动作后缀可达性图

新增`coordinated_suffix_figures_20260916/suffix_probes`，全部95/125/150度，原actor与+2°/+8°两种协同干预真实XY等比例展示，1.25mm允差带保留。9份保存轨迹，无新评价；SVG、实际转换PDF、PNG和CSV均有来源绑定。原PNG及PDF渲染已整页实查，无裁切或重叠。95/125度内侧过渡增强，150度入弯中位偏移仍略外侧；+8°在150度比+2°慢20ms。明确标识有限诊断干预，不是学得方法、最优解或最终图号。旧21页中文和23页英文工作稿仍冻结，未冒称含这些新图。
''')
    receipt(stage,dict(status='finite_reachability_and_rejected_noise_candidate_recorded',
        new_successful_suffixes=24,student_candidate_adopted=False,new_training_started=False,
        report_sha256={p.name:sha(p) for p in reports},goal_complete=False))
    record_progress(stage.name,[stage/'receipt.json',figure/'visual_qa.json',stage/'storage_evidence_reconciliation.json'],
        '24个协同后缀全部合规：95/125度+8°组合由外侧转内侧并快40/60ms；150度仍略外侧，快40ms且慢于+2°组合。真实训练2299个入弯决策仅1步小组合、0步大组合，无8步连续。Student5条件噪声数值验证和32次冻结策略物理比较均完成，覆盖门槛未通过，未启动新训练。下一步研究动作序列表示及正确序列信用，先验证再定有限训练，禁止重复已完成候选和部署逐角模板。两张新图已目视验收，强v7/S2保留，最终研究/论文未完成。存储新证据纠正：03:29完整云端快照2.395GB/238文件/17目录，旧2437队列已过时；仍有零字节跨机器锁未清，未测账户总量。')
    print('Evidence, visual QA, report updates and newer storage reconciliation recorded.')


if __name__=='__main__':main()
