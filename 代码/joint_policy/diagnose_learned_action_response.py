"""Measure the three action responses at saved high-speed corner states."""
import math
import time
import numpy as np
from study_io import BASE, register, receipt, write_json
from joint_runtime_v5 import environment_factory
from joint_runtime import DT, FILTER_TAU, wrap


def response(matrix):
    norms=np.linalg.norm(matrix,axis=0)
    if norms[2]<1e-10:
        return dict(column_norms=norms.tolist(),D_residual_fraction=None,D_local_zero=True)
    projection=matrix[:,:2]@np.linalg.lstsq(matrix[:,:2],matrix[:,2],rcond=1e-10)[0]
    return dict(column_norms=norms.tolist(),D_residual_fraction=float(
        np.linalg.norm(matrix[:,2]-projection)/norms[2]),D_local_zero=False)


def main():
    source=BASE/'joint_refinement_training_v6/final_validation'
    sources=[__file__,BASE/'joint_refinement_training_v6/receipt.json',source/'summary.json',
             BASE/'refinement_final_analysis_v6/receipt.json']
    sources += [source/f'{i}.npz' for i in range(4)]
    sources += [BASE/name for name in ['joint_runtime.py','joint_runtime_v2.py','joint_runtime_v3.py',
        'joint_runtime_v5.py','terminal_candidate_runtime_v1.py','geometry_spec.py','study_io.py']]
    stage=register('learned_macro_action_response_v6',sources,
        dict(saved_isolated_corners=4,states_per_corner=3,conditions_per_state=7,
             local_physics_ticks_max=1680,policy_calls=0,training_decisions=0,
             old_prefix_geometry_reconstruction_ticks_max=20000,wall_s=600),
        'Compare local D response with steering/feed response over 20 ms at saved learned corner states; no full trajectory or training rerun.')
    started=time.perf_counter()
    records=[]
    physics=0
    geometry_reconstruction=0
    for path_index in range(4):
        with np.load(source/f'{path_index}.npz') as data:
            ref,rows,actions=data['reference'],data['rows'],data['actions']
        apex=int(np.argmin(np.linalg.norm(rows[:,:2]-ref[1],axis=1)))
        frame_ticks=sorted(set(max(0,((apex+offset)//20)*20) for offset in [-140,-40,60]))
        assert len(frame_ticks)==3
        env=environment_factory(ref)
        # Probe physical held actions. The baseline uses the exact saved held values.
        env.decode_action=lambda held,fixed_d=None:np.asarray(held,float).copy()
        previous=0
        for frame_tick in frame_ticks:
            assert frame_tick+20<len(rows) and not rows[frame_tick:frame_tick+20,12].any()
            for j in range(previous,frame_tick):
                if j%20==0:
                    env.held=actions[j//20].copy()
                    env.decisions+=1
                s,_=env.progress()
                target,_=env.sample_reference(s+env.held[2])
                difference=target-env.position
                beta=math.atan2(difference[1],difference[0]) if np.linalg.norm(difference)>1e-12 else env.heading
                increment=(1-math.exp(-DT/FILTER_TAU))*wrap(beta+env.held[0]-env.heading)
                env.heading=wrap(env.heading+increment)
                desired=env.held[1]*np.array([math.cos(env.heading),math.sin(env.heading)])
                assert np.array_equal(desired,rows[j,8:10])
                assert np.array_equal(env.position+DT*rows[j,2:4],rows[j,:2])
                env.position=rows[j,:2].copy();env.velocity=rows[j,2:4].copy()
                env.acceleration=rows[j,4:6].copy()
                env.tracker.update(env.position)
                env.ticks+=1
                rest=np.linalg.norm(env.velocity)<=.001 and np.linalg.norm(env.acceleration)<=.01
                env.rest_ticks=env.rest_ticks+1 if rest else 0
                geometry_reconstruction+=1
            previous=frame_tick
            initial=env.state()
            held=actions[frame_tick//20].copy()
            scales=np.array([math.radians(.5),1.,.05])
            conditions=[held]+[held+sign*scales[k]*np.eye(3)[k] for k in range(3) for sign in [-1,1]]
            outputs=[]
            statuses=[]
            for condition in conditions:
                if time.perf_counter()-started>600: raise TimeoutError('Registered wall limit')
                env.restore(initial)
                _,_,done,info=env.step(condition)
                outputs.append(info['rows'])
                physics+=len(info['rows'])
                statuses.append(dict(done=done,reason=env.reason,ticks=len(info['rows'])))
            baseline_equal=np.array_equal(outputs[0],rows[frame_tick:frame_tick+20])
            baseline_error=float(np.max(np.abs(outputs[0]-rows[frame_tick:frame_tick+20])))
            assert baseline_equal, f'Unmatched saved-state baseline at {path_index}/{frame_tick}: {baseline_error}'
            result=dict(path=path_index,angle_deg=[50,95,125,150][path_index],
                frame_tick=frame_tick,offset_from_nearest_sample_ms=frame_tick-apex,
                baseline_exact=True,held_action=held.tolist(),perturbation_scales=scales.tolist(),
                statuses=statuses)
            if all(len(out)==20 for out in outputs):
                # Columns are responses to the stated physical perturbations, not unscaled SI derivatives.
                target=np.stack([(outputs[2+2*k][:,8:10]-outputs[1+2*k][:,8:10]).ravel()/2 for k in range(3)],axis=1)
                executed=np.stack([(outputs[2+2*k][:,2:4]-outputs[1+2*k][:,2:4]).ravel()/2 for k in range(3)],axis=1)
                result.update(target=response(target),executed=response(executed))
                np.savez_compressed(stage/f'{path_index}_{frame_tick}.npz',outputs=np.array(outputs),
                    target_response=target,executed_response=executed)
            else:
                np.savez_compressed(stage/f'{path_index}_{frame_tick}.npz',
                    **{f'condition_{i}':out for i,out in enumerate(outputs)})
                result['response_unavailable']='A perturbation terminated before 20 ms'
            records.append(result)
            env.restore(initial)
    assert physics<=1680 and geometry_reconstruction<=20000
    write_json(stage/'results.json',dict(states=records,physical_ticks=physics,
        old_prefix_geometry_reconstruction_ticks=geometry_reconstruction,policy_calls=0,
        training_decisions=0,finite_difference_not_global_identifiability=True))
    lines=['# 学得策略的三动作局部响应','',
        '从v6联合final四条完整单角的保存状态选取接近、转弯及离开三个20 ms宏步。原前缀只重建引导和有向门状态，未重复物理；十二个未扰动宏步与原保存命令逐位一致。扰动只用于有限局部响应，不是新的完整轨迹评价。','',
        '| 转角/° | 相对最近尖顶时刻/ms | D目标响应残差比例 | D执行响应残差比例 |',
        '|---|---:|---:|---:|']
    for r in records:
        def display(key):
            if key not in r:return '提前终止'
            value=r[key]['D_residual_fraction']
            return '局部零响应' if value is None else f'{value:.5f}'
        lines.append(f"| {r['angle_deg']} | {r['offset_from_nearest_sample_ms']} | {display('target')} | {display('executed')} |")
    lines += ['', '比例为D响应去除转向、速度两列张成空间后的相对残差；扰动尺度分别0.5°、1 mm/s、0.05 mm。局部线性响应不能证明全局冗余、最优动作或更高效率；小响应可能降低局部信用的可分辨性，也可能仅反映当前位置和执行饱和。全局价值仍须由同条件固定D训练比较验证。']
    (stage/'学得策略动作响应.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='saved_learned_state_macro_responses_completed',states=12,
        local_physics_ticks=physics,baseline_replay_ticks=240,new_perturbed_ticks=physics-240,
        policy_calls=0,training_decisions=0,wall_s=time.perf_counter()-started,
        old_prefix_geometry_reconstruction_ticks=geometry_reconstruction,goal_complete=False))
    print('\n'.join(lines))


if __name__=='__main__':main()
