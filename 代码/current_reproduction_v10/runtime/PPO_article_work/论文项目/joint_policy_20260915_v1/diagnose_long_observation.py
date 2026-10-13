"""Four policy queries at a preserved pre-failure state; no command execution."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from study_io import BASE,register,receipt,write_json,sha
from joint_policy import JointPolicy
from joint_runtime_v5 import environment_factory


def main():
    origin=BASE/'joint_long_mixed_storage_recovery_v5'
    snapshot=BASE/'long_observation_capture_v1/state.pt'
    model=BASE/'joint_remaining_credit_training_v5/final.pt'
    paths=BASE/'learning_preparation_v4/paths.json'
    stage=register('long_observation_diagnosis_v5_binding_fix',[__file__,origin/'receipt.json',origin/'evaluation.json',
        origin/'trace.npz',snapshot,model,paths,BASE/'joint_runtime_v3.py',BASE/'joint_policy.py'],
        dict(snapshot_loads=1,model_loads=1,model_query_states=4,physics_ticks=0,training_decisions=0),
        'Check the effect of an out-of-training-range remaining-length observation at the preserved pre-failure state; do not infer trajectory success.')
    torch.set_num_threads(1)
    state=torch.load(snapshot,map_location='cpu',weights_only=False)
    with np.load(origin/'trace.npz') as data:
        ref,rows,actions=data['reference'],data['rows'],data['actions']
    assert state['trace_rows']==91740 and len(rows)==91750
    env=environment_factory(ref)
    env.restore(state['environment'])
    obs=env.observation()
    incoming=torch.load(model,map_location='cpu',weights_only=False)
    policy=JointPolicy(incoming['observation_dim'],mode='joint')
    policy.load_state_dict(incoming['policy'])
    prepared=json.loads(paths.read_text(encoding='utf-8'))
    lengths=[float(np.linalg.norm(np.diff(np.asarray(p),axis=0),axis=1).sum()) for p in prepared['training']]
    upper=math.log1p(max(lengths)/30.)/8.
    queries=[]
    for label,value in [('actual',float(obs[15])),('training_maximum',upper),('remaining_30mm',math.log(2)/8.),('zero_remaining',0.)]:
        altered=obs.copy();altered[15]=value
        latent,_,_=policy.act(altered,deterministic=True)
        action=env.decode_action(latent)
        if label=='actual':
            assert np.array_equal(action,actions[-1])
        queries.append(dict(label=label,remaining_feature=float(altered[15]),
            steering_degrees=float(np.rad2deg(action[0])),feed_mm_s=float(action[1]),D_mm=float(action[2])))
    i=env.tracker.frontier+1
    vectors=np.diff(ref,axis=0);segment_lengths=np.linalg.norm(vectors,axis=1)
    tangents=vectors/segment_lengths[:,None]
    turns=np.rad2deg(np.arccos(np.clip(np.sum(tangents[:-1]*tangents[1:],axis=1),-1.,1.)))
    result=dict(training_length_max=max(lengths),training_feature_max=upper,actual_feature=float(obs[15]),
        actual_remaining_mm=float(env.corridor.arcs[-1]-env.progress()[0]),
        observed_next_vertex=i,local_reference=ref[i-3:i+4].tolist(),local_turns=turns[i-4:i+3].tolist(),
        queries=queries,original_action_reproduced_bitwise=True,physics_ticks=0,
        corrected_trajectory_or_causal_success_proved=False)
    write_json(stage/'results.json',result)
    lines=['# 长路径失败前观测诊断','',
        '使用原保存的91740物理步状态，复现最后一次模型动作，三个物理动作与原记录逐位相同。诊断只改变观测中的剩余长度分量并查询模型四次，不执行任何新命令，不修改当前训练。','',
        f'训练路径最大长度{max(lengths):.3f} mm，对应剩余长度特征最大{upper:.4f}；实际失败前剩余长度{result["actual_remaining_mm"]:.3f} mm，特征为{obs[15]:.4f}。','',
        '| 查询 | 剩余长度特征 | 转向/度 | 速度意图/(mm/s) | D/mm |','|---|---:|---:|---:|---:|']
    for q in queries:
        lines.append(f"| {q['label']} | {q['remaining_feature']:.4f} | {q['steering_degrees']:.3f} | {q['feed_mm_s']:.3f} | {q['D_mm']:.3f} |")
    lines += ['', '该对照能证明这一观测分量对本状态动作输出的影响，不能证明简单裁剪后轨迹就会成功。最后状态位于150°角之后的75°转角内側边界附近，失败时速度约33.284 mm/s；不能仅以速度偏高解释。下一步需结合新训练的长路径表现，考虑训练中的长度和共线分段覆盖，保持双方训练与评价条件一致。']
    (stage/'长路径观测诊断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='four_pre_failure_policy_queries_completed',model_query_states=4,
        original_action_reproduced_bitwise=True,physics_ticks=0,training_decisions=0,goal_complete=False))
    print('\n'.join(lines))


if __name__=='__main__':main()
