"""Append current evidence while retaining the earlier implementation record."""
import json
import shutil
from datetime import datetime,timezone
from study_io import BASE,sha,write_json,record_progress


if __name__=='__main__':
    stages=[('joint_remaining_credit_training_v5',90156,93640),
        ('fixed3_remaining_credit_training_v5',83776,40284),
        ('fixed4p6_remaining_credit_training_v5',80924,77551)]
    observed=[]
    sources=[]
    for name,pid,session in stages:
        stage=BASE/name
        summary=json.loads((stage/'initial_validation/summary.json').read_text(encoding='utf-8'))
        observed.append(dict(stage=name,pid_observed=pid,exec_session=session,
            initial_successes=summary['successes'],
            initial_complete_time_sum_s=sum(r['completion_time_s'] for r in summary['results']) if summary['successes']==12 else None,
            registration_sha256=sha(stage/'registration.json')))
        sources += [stage/'registration.json',stage/'initial_validation/summary.json']
    for name in ['terminal_candidate_diagnosis_v1','snapshot_codec_profile_v1',
        'compact_resume_validation_v5','corner_learning_diagnosis_v4','reward_runtime_validation_v5']:
        sources.append(BASE/name/'receipt.json')
    for name in ['joint_curriculum_training_v4','fixed4p6_curriculum_training_v4']:
        sources.append(BASE/name/'validation_60000/summary.json')
    sources.append(BASE/'fixed3_curriculum_training_v4/failure.json')
    observation=BASE/'v5_launch_observation_v1.json'
    if observation.exists():
        raise FileExistsError(observation)
    write_json(observation,dict(observed_utc=datetime.now(timezone.utc).isoformat(),stages=observed,
        launcher='Direct exec_command sessions; run_logged wrapper was not used for these v5 launches.',
        initial_evaluation_cost='Each new protocol arm was actually evaluated; coincident joint/fixed4.6 initial results are not independent performance replicates.',
        old_v4_scientific_sources_changed=False,old_fixed3_exception_resumed=False,
        sealed_tests_executed=False,goal_complete=False))
    sources.append(observation)
    manuscript_record=BASE/'实施研究进展_20260915.md'
    backup=BASE/'before'/('v5_launch_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True,exist_ok=False)
    shutil.copy2(manuscript_record,backup/manuscript_record.name)
    original=manuscript_record.read_text(encoding='utf-8')
    insert=['## 最新状态补充：v4六万步与独立v5启动','',
        '本段优先于后面的历史运行状态。v4联合和固定4.6在60,000步开发验证仍均8/12成功，无法作十二条完整时间排名，两组继续原120,000步上限。固定3已在第33,007次模型动作、该宏任何物理步之前因有限终点制动候选搜索异常退出；33006步完整状态另存并核哈希，原失败和所有验证保留，未恢复。','',
        'terminal_candidate_diagnosis_v1复现原候选异常（0物理步），并在独立候选接口执行20个新物理步，仍在允差内且独立V/A/J通过。新接口只拒绝不可用的候选而继续原控制，不放松终态条件。corner_learning_diagnosis_v4复用4份保存命令：联合30k在150°弯近角速度中位数70.607 mm/s、D4.489 mm，越界；成功的v3固定4.6近角速度40.452 mm/s。允许内切并不表示联合策略已稳定学会。','',
        'snapshot_codec_profile_v1和compact_resume_validation_v5完成恢复状态的递归逐项往返一致性检查，紧凑格式内存保存约7–9 ms，旧格式约37–41 ms；这是存储微测量，不是完整训练提速结论。reward_runtime_validation_v5完成4个新合成诊断：2成功/2真实越界，共4489物理步、0模型、0训练，成功回报恒等式与失败剩余进度罚均匹配。','',
        '按learning_design_v5.md明确补充的时序条件，v4两组60k仍非全量成功后，独立启动v5联合、固定3、固定4.6各60,000个新决策。三组共同从v3固定4.6 final初始化、D头重置4.6、速度偏置不增加，D潜变量探索标准差改为0.30，采用同64条训练路径、共同新失败奖励及终端候选接口。旧v4不改动、不终止，后来的较好检查点仍保留。','',
        '| v5组 | 已观察初始成功 | 初始完整总时间/s |', '|---|---:|---:|']
    for r in observed:
        insert.append(f'| {r["stage"]} | {r["initial_successes"]}/12 | {r["initial_complete_time_sum_s"]:.3f} |')
    insert += ['',
        '各臂初始化评价是实际执行成本；联合与固定4.6共同初始策略产生相同时间，不作为独立效应重复。新训练是否改善尚待15k/30k/45k/60k检查点。运行PID和exec会话仅为启动观察，后续必须核实存活，不能按旧PID重启。v5启动直接使用exec_command，未使用旧run_logged wrapper，相关事实保留于v5_launch_observation_v1.json。','',
        '论文仍以指定提交稿英文叙述风格为准，运行故障和存储细节保留在外部记录。当前新方法工作稿5页已验收，整篇20–25页、所有新实验图、逐图报告和全PDF验收未完成；封存30条测试没有打开评价。','']
    position=original.index('## 最新状态：v3完成，v4进行中')
    manuscript_record.write_text(original[:position]+'\n'.join(insert)+'\n'+original[position:],encoding='utf-8')
    record_progress('v5_launch_and_supporting_evidence_v1',sources,
        'v4联合和固定4.6在60000步仍8/12，继续原120000步；固定3在33007次动作终点候选搜索异常停止，原状态和失败保留未恢复。独立v5三组各60000步已启动，初始均12/12，使用共同失败进度奖励和已检查终端候选接口，紧凑完整宏恢复在D盘。读取各自15k/30k/45k/60k验证再判断，不能预写优势；封存测试、合理固定D、多种子、长路径及英文20–25页全文/逐图报告/全PDF验收仍待完成。')
    print(observed)
