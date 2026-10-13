"""Bounded feed-head sensitivity on new parameter variants; no training."""
import json
import time
import traceback
import numpy as np
import torch
from study_io import BASE, register, receipt, write_json, record_progress
from joint_policy import JointPolicy
from conditioned_curriculum_v3 import environment_factory
from train_pilot import evaluate, save_torch


if __name__ == '__main__':
    source = BASE/'joint_mixed_curriculum_v2/final.pt'
    prepared = BASE/'mixed_curriculum_preparation_v2'
    sources = [__file__, BASE/'feed_initialization_probe_v4.md', source,
               prepared/'paths.json', prepared/'receipt.json']
    sources += [BASE/name for name in ['joint_policy.py', 'joint_runtime.py',
        'joint_runtime_v2.py', 'joint_runtime_v3.py', 'geometry_spec.py',
        'conditioned_curriculum_v3.py', 'train_pilot.py', 'validate_actions.py', 'study_io.py']]
    stage = register('feed_initialization_probe_v4', sources,
        dict(shifts=[.3,.6,1.,1.5], validation_runs=48, training_decisions=0,
             physics_ticks_per_case=25000, wall_s=900, baseline_replays=0),
        'Diagnose low feed output using four preregistered parameter shifts on development paths.')
    torch.set_num_threads(1)
    started = time.perf_counter()
    costs = dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    results = []
    try:
        data = json.loads((prepared/'paths.json').read_text(encoding='utf-8'))
        paths = [np.array(ref,float) for ref in data['validation']]
        incoming = torch.load(source,map_location='cpu',weights_only=False)
        for label,shift in [('shift_0p3',.3),('shift_0p6',.6),('shift_1p0',1.),('shift_1p5',1.5)]:
            policy = JointPolicy(incoming['observation_dim'])
            policy.load_state_dict(incoming['policy'])
            with torch.no_grad():
                policy.actor[-1].bias[1] += shift
            save_torch(stage/(label+'.pt'),dict(policy=policy.state_dict(),
                observation_dim=incoming['observation_dim'],mode='joint',feed_bias_shift=shift))
            rows = evaluate(policy,paths,stage,label,None,costs,started+900,environment_factory)
            results.append(dict(shift=shift,tag=label,results=rows,
                successes=sum(r['success'] for r in rows),
                complete_time_sum_s=sum(r['elapsed_s'] for r in rows) if all(r['success'] for r in rows) else None))
            write_json(stage/'results.json',dict(candidates=results,costs=costs,training_decisions=0))
        receipt(stage,dict(status='feed_parameter_probe_completed',candidates=4,
            evaluation_runs=48,training_decisions=0,costs=costs,
            wall_s=time.perf_counter()-started,goal_complete=False))
        record_progress('feed_initialization_probe_v4',[stage/'receipt.json'],
            'v3三组训练按各自原登记推进；速度参数探测只作为学习初始化诊断，非训练或最终性能证据。完成后先比较全量成功率与完整时间，检查直线速度和急弯形状，再决定共同初始化/课程的有界后继。S2优势、强固定D、封存测试、多种子、长路径和最终论文验收仍待完成。')
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,
            completed_candidates=len(results),automatic_retry=False))
        raise
