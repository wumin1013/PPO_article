"""Check terminal credit using completed records and finite algebra examples."""
import json
import numpy as np
from study_io import BASE,register,receipt,write_json


if __name__=='__main__':
    names=['joint_conditioned_training_v3','fixed3_conditioned_training_v3','fixed4p6_conditioned_training_v3']
    sources=[__file__,BASE/'joint_runtime.py',BASE/'joint_runtime_v3.py',BASE/'conditioned_curriculum_v3.py']
    for name in names:sources += [BASE/name/'results.json',BASE/name/'paths.json',BASE/name/'receipt.json']
    stage=register('failure_credit_analysis_v1',sources,
        dict(completed_training_groups=3,synthetic_return_checks=6,models=0,physics_ticks=0),
        'Diagnose how the current complete-time reward ranks failures; test a proposed remaining-progress terminal penalty without changing active training.')
    reports=[]
    for name in names:
        original=json.loads((BASE/name/'results.json').read_text(encoding='utf-8'))
        paths=json.loads((BASE/name/'paths.json').read_text(encoding='utf-8'))['training']
        failures=[r for r in original['episodes'] if not r['success']]
        max_error=0.
        pairs=[]
        for row in failures:
            ref=np.array(paths[row['path']],float)
            length=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
            # The environment rounds its episode limit to a whole physical tick.
            limit=round(max(5.,length/20.+1.)/.001)*.001
            expected=-row['elapsed_s']+length/100.-(limit+1.)
            max_error=max(max_error,abs(expected-row['return_sum']))
        assert max_error<1e-8
        for index in sorted({r['path'] for r in failures}):
            rows=sorted([r for r in failures if r['path']==index],key=lambda r:r['elapsed_s'])
            if len(rows)>=2 and rows[-1]['elapsed_s']>rows[0]['elapsed_s']+1e-6:
                first,last=rows[0],rows[-1]
                assert first['return_sum']>last['return_sum']
                pairs.append(dict(path=index,earlier_failure_time=first['elapsed_s'],
                    earlier_failure_return=first['return_sum'],later_failure_time=last['elapsed_s'],
                    later_failure_return=last['return_sum']))
        reports.append(dict(stage=name,failure_episodes=len(failures),
            maximum_return_identity_error=max_error,paths_with_observed_early_failure_preference=len(pairs),pairs=pairs))
    checks=[]
    def ret(length,limit,time,progress,success):
        return -time+length/100.-(0. if success else limit+1.+max(length-progress,0.)/20.)
    # Success ordering remains exactly complete-time ordering on the same path.
    for length in [56.,118.,30000.]:
        limit=max(5.,length/20.+1.)
        slow,fast=.8*limit,.7*limit
        assert abs((ret(length,limit,fast,length,True)-ret(length,limit,slow,length,True))-(slow-fast))<1e-9
        # Even the slowest permitted success exceeds the best possible failure.
        assert ret(length,limit,limit,length,True)>ret(length,limit,0.,length,False)
        checks.append(dict(length=length,successful_time_order_preserved=True,success_dominates_all_failures=True))
    for progress,time in [(10.,.25),(20.,.4),(50.,1.)]:
        assert ret(100.,6.,time,progress,False)>ret(100.,6.,0.,0.,False)
        checks.append(dict(progress=progress,time=time,additional_progress_above_20mm_s_improves_failure_return=True))
    write_json(stage/'results.json',dict(completed_record_checks=reports,synthetic_checks=checks,
        proposed_failure_penalty='episode_limit_s + 1 + remaining_ordered_reference_mm / 20',
        applied_to_current_training=False,causal_bottleneck_proven=False))
    text=['# 失败回合的信用分配','',
        '当前完整奖励在失败时也把势函数设为零。同一路径的失败回报因此等于−失败经过时间+L/100−(时限+1)。这意味着在都失败的回合之间，越早失败反而回报越高。以下由已完成v3记录直接核对，未运行模型或物理。','',
        '| 训练组 | 失败回合 | 实测同路径早失败偏好的路径数 | 回报等式最大误差 |',
        '|---|---:|---:|---:|']
    for r in reports:text.append(f'| {r["stage"]} | {r["failure_episodes"]} | {r["paths_with_observed_early_failure_preference"]} | {r["maximum_return_identity_error"]:.2e} |')
    text += ['',
        '这证明现有奖励怎样排序失败回合，尚不能证明它是当前学习困难的唯一原因。若策略在困难角上很少成功，这种回报可能无法引导它逐步增加有效完成进度。',
        '', '候选只修改失败终端奖励：在原罚项之外，增加未完成有序参考长度/20 mm/s的罚项。所有成功轨迹的回报完全不变，成功时间排序保留；同一路径中，任何时限内成功仍优于任何失败。失败之间，新增进度在平均20 mm/s以上时可以抵消额外经过时间。该20 mm/s是辅助失败评分尺度，与既有回合时限尺度一致，不是执行速度限制或角点控制规则。',
        '', '六项有限代数检查已通过。它不证明训练一定改善，也没有改变正在运行的v4奖励、模型或记录。若采用，须在v4完成后另立共同训练版本，各固定D和联合策略使用同一奖励；完整评价仍以原允差/门序/V/A/J/停稳判据和成功时间为准，不用失败分数制造性能排名。']
    (stage/'失败信用分析.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    receipt(stage,dict(status='failure_return_identity_and_candidate_algebra_checked',
        completed_training_groups=3,synthetic_checks=6,models=0,physics_ticks=0,
        active_training_modified=False,learning_benefit_established=False))
    print('\n'.join(text))
