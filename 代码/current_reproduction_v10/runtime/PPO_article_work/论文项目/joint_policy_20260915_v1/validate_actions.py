"""Finite action-sensitivity and complete-command integration checks."""
import math
import time
import traceback
import numpy as np
import torch
from study_io import BASE, ROOT, register, receipt, write_json, record_progress


def latent(speed=30.,distance=3.,steering=0.):
    return np.arctanh([steering/(math.pi/2),speed/50.-1.,(distance-2.7)/1.9])


def corner(angle,length=20.):
    radians = math.radians(angle)
    return np.array([[-length,0.],[0.,0.],[length*math.cos(radians),length*math.sin(radians)]])


def difference_audit(initial, rows):
    positions = np.vstack([initial,rows[:,:2]])
    velocity = np.vstack([np.zeros(2),np.diff(positions,axis=0)/.001])
    acceleration = np.vstack([np.zeros(2),np.diff(velocity,axis=0)/.001])
    jerk = np.diff(acceleration,axis=0)/.001
    peaks = [float(np.linalg.norm(x,axis=1).max()) for x in (velocity,acceleration,jerk)]
    tail = len(rows)>=21 and np.linalg.norm(velocity[-21:],axis=1).max()<=.001 and np.linalg.norm(acceleration[-21:],axis=1).max()<=.01
    return dict(peaks=peaks,kinematic_pass=bool(np.all(np.array(peaks)<=np.array([100.,2000.,62500.])*1.00001)),rest_pass=bool(tail))


if __name__=='__main__':
    torch.set_num_threads(1)
    sources = [__file__,BASE/'study_io.py',BASE/'geometry_spec.py',BASE/'joint_runtime.py',BASE/'joint_policy.py',BASE/'task_contract.md']
    stage = register('action_validation_v1',sources,dict(sensitivity_states=4,conditions_per_state=7,
        max_probe_ticks=560,complete_synthetic_runs=4,max_ticks_per_run=4000,training_updates=0,wall_limit_s=900),
        'Validate each held action over the full 20 ms target/execution stack; separate complete synthetic integration diagnostics.')
    started = time.perf_counter()
    try:
        from joint_runtime import JointRuntime
        from joint_policy import JointPolicy,advantages
        sensitivities = []
        for angle in [0.,45.,90.,130.]:
            env = JointRuntime(corner(angle),max_time=4.)
            env.position = np.array([-2.8,.15])
            env.tracker.position = env.position.copy()
            env.velocity = np.array([20.,0.])
            env.acceleration = np.zeros(2)
            initial = env.state()
            outputs = []
            base = latent(speed=24.)
            conditions = [base]+[base+sign*.01*np.eye(3)[j] for j in range(3) for sign in [-1,1]]
            for condition in conditions:
                env.restore(initial)
                _,_,_,info = env.step(condition)
                outputs.append(info['rows'])
            if not all(len(rows)==20 for rows in outputs):
                raise AssertionError('A sensitivity probe terminated early')
            target = np.stack([(outputs[2+2*j][:,8:10]-outputs[1+2*j][:,8:10]).ravel()/.02 for j in range(3)],axis=1)
            executed = np.stack([(outputs[2+2*j][:,2:4]-outputs[1+2*j][:,2:4]).ravel()/.02 for j in range(3)],axis=1)
            target_sv = np.linalg.svd(target,compute_uv=False)
            executed_sv = np.linalg.svd(executed,compute_uv=False)
            sensitivities.append(dict(angle_deg=angle,target_singular_values=target_sv.tolist(),
                execution_singular_values=executed_sv.tolist(),
                execution_column_norms=np.linalg.norm(executed,axis=0).tolist()))
            np.savez_compressed(stage/('sensitivity_'+str(int(angle))+'.npz'),outputs=np.asarray(outputs),
                                target_jacobian=target,execution_jacobian=executed)
        assert all(min(row['execution_column_norms'])>1e-6 for row in sensitivities)
        assert any(row['execution_singular_values'][-1]>1e-4 for row in sensitivities if row['angle_deg']>0)

        env = JointRuntime(corner(90.),max_time=4.)
        observation = env.observation()
        torch.manual_seed(713)
        policy = JointPolicy(len(observation))
        sample,old_log,value = policy.act(observation)
        new_log,_,_ = policy.evaluate(torch.tensor(observation),torch.tensor(sample))
        assert abs(float(new_log)-old_log)<1e-6
        (-new_log).backward()
        gradients = policy.actor[-1].bias.grad.detach().numpy()
        assert np.all(np.abs(gradients)>1e-7)
        only_d = JointPolicy(len(observation),mode='d_only',reference_policy=policy)
        before = only_d.distribution(torch.tensor(observation)).mean.detach().numpy()
        with torch.no_grad():
            for parameter in only_d.actor.parameters(): parameter.add_(.01)
        after = only_d.distribution(torch.tensor(observation)).mean.detach().numpy()
        assert np.array_equal(before[:2],after[:2]) and before[2]!=after[2]
        adv,_ = advantages([1.,2.],[.5,.8],[.8,7.],[False,True],lam=1.)
        assert np.allclose(adv,[2.5,1.2])
        bootstrap,_ = advantages([1.],[.5],[7.],[False],lam=1.)
        assert np.allclose(bootstrap,[7.5])

        runs = []
        for angle in [0.,45.,90.,130.]:
            env = JointRuntime(corner(angle,10.),max_time=4.)
            chunks = []
            total_reward = 0.
            while not env.done:
                _,reward,_,info = env.step(latent(speed=15.))
                total_reward += reward
                chunks.append(info['rows'])
                if time.perf_counter()-started>900: raise TimeoutError('Wall limit')
            rows = np.vstack(chunks)
            audit = difference_audit(env.corridor.reference[0],rows)
            expected = -env.ticks*.001-(0 if env.success else 5.)
            assert abs(total_reward-expected)<1e-9
            assert audit['kinematic_pass']
            if env.success: assert audit['rest_pass']
            runs.append(dict(angle_deg=angle,elapsed_s=env.ticks*.001,success=env.success,
                reason=env.reason,geometry=env.tracker.snapshot(1.25),audit=audit,
                reward=total_reward,decisions=env.decisions))
            np.savez_compressed(stage/('commands_'+str(int(angle))+'.npz'),reference=env.corridor.reference,rows=rows)
            print(runs[-1],flush=True)
        write_json(stage/'results.json',dict(sensitivity=sensitivities,integration_runs=runs,
            observation_dim=len(observation),all_three_policy_head_gradients=gradients.tolist(),
            likelihood_replay_pass=True,only_d_full_function_freeze_pass=True,
            true_terminal_and_truncation_bootstrap_pass=True,wall_s=time.perf_counter()-started))
        receipt(stage,dict(status='action_interface_and_probability_checks_passed',
            synthetic_success_count=sum(row['success'] for row in runs),synthetic_count=len(runs),
            policy_samples=1,backward_checks=1,training_updates=0,
            physics_ticks=560+sum(round(row['elapsed_s']*1000) for row in runs),
            full_reference_evaluations=0,performance_superiority_established=False))
        record_progress('joint_action_stage_v1',[stage/'receipt.json'],
            '真实三动作接口、联合潜变量似然、仅D完整函数冻结和终止/bootstrap检查完成，数值与合成过弯结果见joint_policy_20260915_v1/action_validation_v1。下一步依据实际合成结果做有限吞吐标定和新训练；固定D将保留同权限转向/速度训练。新长程性能和优于对照尚未建立，保持原稿英文风格，20–25页和全PDF验收仍待完成，目标active。')
        print({'status':'interface_passed','sensitivities':sensitivities,'observation_dim':len(observation),'wall_s':time.perf_counter()-started},flush=True)
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),wall_s=time.perf_counter()-started))
        raise
