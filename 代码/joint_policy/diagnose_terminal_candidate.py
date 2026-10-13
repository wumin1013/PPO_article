"""One stopped v4 state, one reproduced planner exception, one new macro."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from study_io import BASE, ROOT, register, receipt, write_json, sha, record_progress


if __name__ == '__main__':
    capture = BASE/'v4_fixed3_terminal_capture_v1'
    capture.mkdir(exist_ok=False)
    origin = Path('D:/PPO_article_joint_runs/rolling_states/fixed3_curriculum_training_v4')
    pointer = json.loads((origin/'resume_pointer.json').read_text(encoding='utf-8'))
    assert pointer['decision'] == 33006 and pointer['phase'] == 'training'
    original = origin/f'resume_{pointer["slot"]}.pt'
    shutil.copy2(original, capture/'state.pt')
    assert sha(original) == sha(capture/'state.pt')
    write_json(capture/'capture.json', dict(origin=str(original), sha256=sha(original), pointer=pointer))
    prior = BASE/'fixed3_curriculum_training_v4'
    sources = [__file__, capture/'state.pt', capture/'capture.json', prior/'failure.json',
        prior/'registration.json', BASE/'learning_preparation_v4/paths.json']
    sources += [BASE/name for name in ['joint_runtime.py','joint_runtime_v2.py','joint_runtime_v3.py',
        'joint_policy.py','geometry_spec.py','terminal_candidate_runtime_v1.py','study_io.py']]
    stage = register('terminal_candidate_diagnosis_v1',sources,
        dict(snapshot_loads=1,models=1,actor_calls=1,original_planner_queries=1,
             new_runtime_macros=1,physics_ticks_max=20,training_updates=0,wall_s=180),
        'Diagnose the preserved fixed-D v4 exception; test declining the unavailable brake candidate in a separate runtime, without resuming v4.')
    started=time.perf_counter()
    from conditioned_curriculum_v3 import environment_factory
    from terminal_candidate_runtime_v1 import TerminalCandidateRuntime
    from joint_policy import JointPolicy
    torch.set_num_threads(1)
    state=torch.load(capture/'state.pt',map_location='cpu',weights_only=False)
    paths=json.loads((BASE/'learning_preparation_v4/paths.json').read_text(encoding='utf-8'))
    ref=np.asarray(paths['training'][state['path_index']],float)
    original_env=environment_factory(ref,fixed_d=3.)
    original_env.restore(state['environment'])
    policy=JointPolicy(state['observation_dim'],mode='fixed')
    policy.load_state_dict(state['policy'])
    torch.set_rng_state(state['torch_rng'])
    action,logp,value=policy.act(original_env.observation())
    before=original_env.state()
    try:
        original_env._consider_terminal()
        raise AssertionError('Captured candidate search unexpectedly succeeded')
    except ValueError as error:
        message=str(error)
        assert message=='No feasible nonreversing vector cubic brake within registered horizon'
    assert original_env.ticks==before['ticks']
    assert np.array_equal(original_env.position,before['position'])
    candidate=TerminalCandidateRuntime(ref,fixed_d=3.,max_time=original_env.max_ticks*.001,
        closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-9))
    candidate.restore(before)
    _,reward,done,info=candidate.step(action)
    rows=info['rows']
    assert 0<len(rows)<=20
    position=np.vstack([before['position'],rows[:,:2]])
    velocity=np.vstack([before['velocity'],np.diff(position,axis=0)/.001])
    acceleration=np.vstack([before['acceleration'],np.diff(velocity,axis=0)/.001])
    jerk=np.diff(acceleration,axis=0)/.001
    peaks=[float(np.linalg.norm(x,axis=1).max()) for x in [velocity,acceleration,jerk]]
    assert np.all(np.array(peaks)<=np.array([100.,2000.,62500.])*1.00001)
    np.savez_compressed(stage/'new_macro.npz',reference=ref,rows=rows,initial_position=before['position'],
        initial_velocity=before['velocity'],initial_acceleration=before['acceleration'],latent=action)
    result=dict(captured_decision=state['decision'],path_index=state['path_index'],
        position=before['position'].tolist(),velocity=before['velocity'].tolist(),
        acceleration=before['acceleration'].tolist(),speed=float(np.linalg.norm(before['velocity'])),
        velocity_dot_acceleration=float(before['velocity']@before['acceleration']),
        planner_error=message,original_exception_physics_ticks=0,new_macro_physics_ticks=len(rows),
        independent_vaj_peaks=peaks,after_reason=info['reason'],after_inside=candidate.tracker.inside,
        new_macro_reward=reward,full_success_established=False,original_stage_resumed=False)
    write_json(stage/'results.json',result)
    text=['# 终点制动候选异常诊断','',
        '固定3 mm的v4训练在第33007次动作后、该动作任何物理步之前，因为有限三次制动族找不到可行候选而抛出异常。原第33006步完整状态、优化器、随机数和未更新数据已另存并核对哈希；原失败与训练文件保持不变。','',
        '此异常只说明当前有限候选族无解，不能推断满足公共限制的所有制动都无解。新候选接口仅将这一特定搜索失败视为未接入终端制动，继续由原策略和KCM控制；不降低允差、动力学或停稳要求，也不吞掉其他异常。','',
        f'原候选搜索复现1次，物理步0。独立候选执行1个宏步、{len(rows)}个物理步，独立差分V/A/J峰值为{peaks}，公共动力学审核通过。局部执行后的原因为{info["reason"]}，仍在允差域内为{candidate.tracker.inside}。此项不代表完整终点或完整训练成功。','',
        '候选未用于正在运行的v4，未恢复固定3组，也未重跑其已有验证。后续若采用须以共同运行接口重新登记各比较臂，并保留v4中断的真实成本。']
    (stage/'终点候选诊断.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    receipt(stage,dict(status='preserved_exception_diagnosed_separate_candidate_macro_checked',
        snapshot_loads=1,models=1,actor_calls=1,physics_ticks=len(rows),training_updates=0,
        original_stage_resumed=False,wall_s=time.perf_counter()-started))
    record_progress('v4_terminal_candidate_diagnosis_v1',[stage/'receipt.json',prior/'failure.json'],
        'v4联合与固定4.6继续原120000步预算；固定3在33007次动作的终点候选搜索异常停止，保留33006步完整状态和所有验证，不自动恢复。新候选接口只完成一次独立20ms检查，未改变活动v4。继续分析急弯与失败信用，后继须共同登记；目标和全论文仍未完成。')
    print(json.dumps(result,ensure_ascii=False))
