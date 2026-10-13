"""Finite persistent action interventions from saved approach states; no training."""
import itertools
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from joint_policy import JointPolicy
from joint_runtime_v5 import environment_factory
from joint_runtime import DT, FILTER_TAU, wrap
from validate_actions import difference_audit
from summarize_conditioned_training import diagnostics
from study_io import BASE, register, receipt, write_json, sha, record_progress


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def reconstruct(ref,rows,actions,frame,costs,observations=None,state_snapshots=None,env_factory=None):
    """Reuse recorded physical states, replay only guidance and ordered gate bookkeeping."""
    env=(env_factory or environment_factory)(ref)
    for j in range(frame):
        assert not rows[j,12]
        if j%20==0:
            if observations is not None:observations.append(env.observation())
            if state_snapshots is not None:state_snapshots.append(env.state())
            env.held=actions[j//20].copy();env.decisions+=1
        target=env.guidance_target()
        delta=target-env.position
        beta=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-12 else env.heading
        increment=(1-math.exp(-DT/FILTER_TAU))*wrap(beta+env.held[0]-env.heading)
        env.heading=wrap(env.heading+increment)
        desired=env.held[1]*np.array([math.cos(env.heading),math.sin(env.heading)])
        assert np.array_equal(desired,rows[j,8:10])
        assert np.array_equal(env.position+DT*rows[j,2:4],rows[j,:2])
        env.position=rows[j,:2].copy();env.velocity=rows[j,2:4].copy()
        env.acceleration=rows[j,4:6].copy();env.tracker.update(env.position);env.ticks+=1
        rest=np.linalg.norm(env.velocity)<=.001 and np.linalg.norm(env.acceleration)<=.01
        env.rest_ticks=env.rest_ticks+1 if rest else 0
        costs['saved_prefix_geometry_ticks']+=1
    return env


def encode(held):
    unit=np.array([held[0]/(math.pi/2),held[1]/50-1,(held[2]-6.4)/5.6])
    assert (np.abs(unit)<1).all(), 'Registered offsets must remain in the existing action range'
    return np.arctanh(unit)


def shape(ref,rows,actions):
    result=diagnostics(ref,rows,actions)
    xy=rows[:,:2]-ref[1]
    apex=int(np.argmin(np.linalg.norm(xy,axis=1)))
    # These are positive left-turn paths in the original XY frame.
    inbound=(np.arange(len(xy))<apex)&(xy[:,0]>=-10)&(xy[:,0]<=-2)
    result['approach_minus10_to_minus2_median_signed_y_mm']=float(np.median(xy[inbound,1])) if inbound.any() else None
    result['approach_minus10_to_minus2_max_signed_y_mm']=float(np.max(xy[inbound,1])) if inbound.any() else None
    return result


def main():
    source=BASE/'ar_conditional_joint_20260916';traces=source/'final_validation'
    assert read(source/'selection.json')['selected']['decisions']==60000
    assert read(source/'receipt.json')['final_sha256']==sha(source/'final.pt')
    sources=[Path(__file__),source/'final.pt',source/'receipt.json',source/'selection.json',traces/'summary.json',
             BASE/'conditional_exploration_results_20260916/receipt.json',
             BASE/'conditional_results_recorded_20260916/receipt.json']
    sources += [traces/f'{i}.npz' for i in (1,2,3)]
    sources += [BASE/n for n in ('joint_policy.py','joint_runtime.py','joint_runtime_v2.py','joint_runtime_v3.py',
        'joint_runtime_v5.py','terminal_candidate_runtime_v1.py','geometry_spec.py','validate_actions.py',
        'summarize_conditioned_training.py','study_io.py')]
    stage=register('coordinated_suffix_probe_20260916',sources,
        dict(saved_paths=[1,2,3],reused_baselines=3,new_suffixes=21,intervention_decisions=8,
             steering_offset_degrees=2,feed_offset_mm_s=5,preview_offset_mm=2,
             new_physics_ticks=105000,actor_calls=5253,saved_prefix_geometry_ticks=3000,wall_s=600,
             training_decisions=0,training_updates=0),
        'At the last saved decision boundary before x=-10 mm, test the seven nonzero 2^3 positive action combinations for eight decisions, then original deterministic actor to full finish. Reuse three original baselines. Finite offline reachability/credit probe only; no templates, no extra selection or training.')
    costs=dict(new_suffixes=0,new_physics_ticks=0,actor_calls=0,saved_prefix_geometry_ticks=0,model_loads=0,training_updates=0)
    started=time.perf_counter();deadline=started+600
    try:
        torch.set_num_threads(1)
        incoming=torch.load(source/'final.pt',map_location='cpu',weights_only=False);costs['model_loads']=1
        policy=JointPolicy(incoming['observation_dim'],mode='joint');policy.load_state_dict(incoming['policy'])
        original=read(traces/'summary.json')['results'];groups=[]
        offsets=np.array([math.radians(2.),5.,2.])
        for index in (1,2,3):
            with np.load(traces/f'{index}.npz') as data:
                ref,rows,actions=(data[k].copy() for k in ('reference','rows','actions'))
            x=rows[:,0]-ref[1,0]
            crossing=int(np.flatnonzero(x>=-10)[0]);frame=((crossing+1)//20)*20
            env=reconstruct(ref,rows,actions,frame,costs);initial=env.state();phi=env.potential()
            latent,_,_=policy.act(env.observation(),deterministic=True);costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(latent),actions[frame//20]), 'Saved actor action must match exactly'
            base=next(r for r in original if r['path']==index);assert base['success']
            baseline=dict(bits=[0,0,0],reused=True,success=True,completion_time_s=base['completion_time_s'],
                suffix_return=-(base['completion_time_s']-frame*DT)-phi,diagnostics=shape(ref,rows,actions))
            records=[baseline]
            for bits in itertools.product((0,1),repeat=3):
                if not any(bits):continue
                env.restore(initial);chunks=[];helds=[];rewards=[];count=0
                while not env.done:
                    assert time.perf_counter()<deadline
                    assert costs['actor_calls']<5253 and costs['new_physics_ticks']<102880
                    latent,_,_=policy.act(env.observation(),deterministic=True);costs['actor_calls']+=1
                    if count<8:
                        held=env.decode_action(latent)+offsets*np.array(bits)
                        latent=encode(held)
                    _,reward,_,info=env.step(latent)
                    chunks.append(info['rows']);helds.append(info['held_action']);rewards.append(reward);count+=1
                    costs['new_physics_ticks']+=len(info['rows'])
                costs['new_suffixes']+=1
                full=np.vstack([rows[:frame],*chunks]);allactions=np.vstack([actions[:frame//20],helds])
                audit=difference_audit(ref[0],full)
                success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                tag=f"path{index}_"+''.join(map(str,bits))
                np.savez_compressed(stage/f'{tag}.npz',reference=ref,rows=full,actions=allactions,
                                    intervention_start_tick=frame,intervention_decisions=min(count,8))
                result=dict(bits=list(bits),reused=False,success=success,reason=env.reason,
                    completion_time_s=env.ticks*DT if success else None,elapsed_s=env.ticks*DT,
                    suffix_return=sum(rewards),delta_return_from_base=sum(rewards)-baseline['suffix_return'],
                    geometry=env.tracker.snapshot(1.25),audit=audit,diagnostics=shape(ref,full,allactions),
                    trace=f'{tag}.npz',suffix_decisions=count)
                if success:
                    result['delta_time_s']=result['completion_time_s']-baseline['completion_time_s']
                    assert abs(result['delta_return_from_base']+result['delta_time_s'])<1e-10
                records.append(result)
                print(index,bits,success,result.get('delta_time_s'),result['reason'],flush=True)
            groups.append(dict(path=index,angle_deg={1:95,2:125,3:150}[index],anchor_tick=frame,
                               anchor_position=initial['position'].tolist(),source_trace_sha256=sha(traces/f'{index}.npz'),records=records))
            write_json(stage/'partial_results.json',dict(groups=groups,costs=costs))
        assert costs['new_suffixes']==21 and costs['new_physics_ticks']<=105000 and costs['saved_prefix_geometry_ticks']<=3000
        result=dict(groups=groups,costs=costs,
            boundary='One signed offset and one duration at three existing development states. Finite counterfactual probe, not learned behavior, independent test or optimum. Correlated interventions are diagnostic only and never deployed.',
            new_training=False,goal_complete=False)
        write_json(stage/'results.json',result)
        lines=['# 持续三动作组合后缀探针','','从原条件联合60k的三个入弯状态分支；000直接复用已完成基线。bits依次为转向+2°、进给+5 mm/s、D+2 mm，持续8个决策，然后恢复原actor。',
               '','| 角度 | bits | 完整通过 | 完整时间差/s | 入弯段中位有符号y/mm | 最近尖顶/mm |',
               '|---|---|---:|---:|---:|---:|']
        for g in groups:
            for r in g['records']:
                delta=r.get('delta_time_s',0. if r['reused'] else None)
                y=r['diagnostics']['approach_minus10_to_minus2_median_signed_y_mm']
                d=r['diagnostics']['corners'][0]['minimum_vertex_distance_mm']
                lines.append(f"| {g['angle_deg']} | {''.join(map(str,r['bits']))} | {r['success']} | {delta if delta is not None else '失败不排名'} | {y:.3f} | {d:.3f} |")
        lines += ['',result['boundary'],'','成功后缀回报差与负完整时间差数值一致；未添加形状奖励。失败保留真实前缀，不作快速完成。']
        (stage/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        receipt(stage,dict(status='completed_finite_persistent_action_probe',**costs,wall_s=time.perf_counter()-started))
        record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
            '完成21个持续三动作组合后缀探针，3条原基线复用，无训练更新。仅在公开95/125/150度入弯状态检验2°/5mm每秒/2mm的七个组合，各持续8决策后回原actor；完整审核和回报/时间一致性已核。下一步按实际结果解释协同与动作可达性，不直接部署诊断规则或自动扩训。条件探索候选已按门槛停止，强v7/S2保留，研究/最终论文/云端精简验收未完成。')
        print('\n'.join(lines))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,automatic_retry=False))
        raise


if __name__=='__main__':main()
