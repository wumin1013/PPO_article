"""Frozen-policy Gaussian/Student exploration comparison before deciding to train."""
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from conditional_exploration import ConditionalExplorationPolicy
from student_exploration import StudentExplorationPolicy
from joint_runtime_v5 import environment_factory
from validate_actions import difference_audit
from probe_coordinated_suffix import read
from study_io import BASE,register,receipt,write_json,sha,record_progress


def main():
    source=BASE/'ar_conditional_joint_20260916'
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    indices=np.linspace(0,len(paths)-1,16).astype(int).tolist();assert len(set(indices))==16
    stage=register('student_exploration_physical_probe_20260916',
        [Path(__file__),BASE/'student_exploration.py',BASE/'conditional_exploration.py',
         BASE/'student_exploration_validation_20260916/receipt.json',BASE/'coordinated_exploration_coverage_20260916/receipt.json',
         BASE/'learning_preparation_v4/paths.json',source/'final.pt',source/'receipt.json',
         *[BASE/n for n in ('joint_policy.py','joint_runtime.py','joint_runtime_v2.py','joint_runtime_v3.py',
           'joint_runtime_v5.py','terminal_candidate_runtime_v1.py','geometry_spec.py','validate_actions.py')]],
        dict(samplers=['gaussian','student5'],paths=indices,frozen_episode_attempts=32,
             case_command_tick_stop=5000,physics_ticks_upper=227840,policy_decisions_upper=8000,
             wall_s=300,training_updates=0,seed_rule='2026091600+path index for each sampler'),
        'Matched-covariance frozen actor comparison, sixteen fixed training geometries each. Stop each case at 5000 ticks or terminal, retaining budget-censoring identity. Student candidate proceeds only if an eight-step demonstrated coordinated approach excursion appears and successful episode count is no more than one below Gaussian; no auto training or new reward.')
    costs=dict(episode_attempts=0,policy_decisions=0,extra_mean_actor_forwards=0,physics_ticks=0,training_updates=0)
    started=time.perf_counter();groups=[];torch.set_num_threads(1)
    try:
        incoming=torch.load(source/'final.pt',map_location='cpu',weights_only=False)
        for sampler,cls in (('gaussian',ConditionalExplorationPolicy),('student5',StudentExplorationPolicy)):
            policy=cls(incoming['observation_dim'],mode='joint');policy.load_state_dict(incoming['policy'])
            records=[];counts={str(d):dict(rows=0,longest=0,windows_eight=0) for d in (2,8)};eligible=0
            for index in indices:
                torch.manual_seed(2026091600+index);policy.reset_context();ref=np.asarray(paths[index],float);env=environment_factory(ref)
                chunks=[];actions=[];total_reward=0.;runs={k:0 for k in counts}
                costs['episode_attempts']+=1
                while not env.done and env.ticks<5000:
                    assert time.perf_counter()-started<300 and costs['policy_decisions']<8000
                    obs=env.observation()
                    with torch.no_grad():mean=policy.actor(torch.as_tensor(obs)).numpy()
                    costs['extra_mean_actor_forwards']+=1
                    latent,_,_=policy.act(obs);costs['policy_decisions']+=1
                    held=env.decode_action(latent);delta=held-env.decode_action(mean)
                    inside=False
                    if len(ref)==3:
                        a,b=ref[1]-ref[0],ref[2]-ref[1];sign=np.sign(a[0]*b[1]-a[1]*b[0])
                        distance=float((env.position-ref[1])@(a/np.linalg.norm(a)))
                        inside=bool(sign and -12<=distance<=-2)
                        if inside:
                            eligible+=1
                            for key in counts:
                                yes=sign*delta[0]>=math.radians(float(key)) and delta[1]>=5 and delta[2]>=2
                                runs[key]=runs[key]+1 if yes else 0
                                counts[key]['rows']+=int(yes);counts[key]['longest']=max(counts[key]['longest'],runs[key])
                                counts[key]['windows_eight']+=int(runs[key]>=8)
                    if not inside:
                        for key in runs:runs[key]=0
                    _,reward,_,info=env.step(latent);chunks.append(info['rows']);actions.append(info['held_action']);total_reward+=reward
                    costs['physics_ticks']+=len(info['rows']);assert costs['physics_ticks']<=227840
                rows=np.vstack(chunks);audit=difference_audit(ref[0],rows)
                success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                np.savez_compressed(stage/f'{sampler}_{index}.npz',reference=ref,rows=rows,actions=actions)
                records.append(dict(path=index,success=success,reason=env.reason if env.done else 'probe_budget_censored',
                    completion_time_s=env.ticks*.001 if success else None,elapsed_s=env.ticks*.001,
                    reward=total_reward,geometry=env.tracker.snapshot(1.25),audit=audit))
            group=dict(sampler=sampler,successes=sum(r['success'] for r in records),cases=records,
                       eligible_approach_rows=eligible,coordinated_counts=counts)
            groups.append(group);write_json(stage/'partial_results.json',dict(groups=groups,costs=costs))
            print(sampler,group['successes'],counts,flush=True)
        supported=(groups[1]['coordinated_counts']['2']['windows_eight']>0 and
                   groups[1]['successes']>=groups[0]['successes']-1)
        result=dict(groups=groups,costs=costs,physical_probe_support=supported,automatic_training=False,
            boundary='Sixteen preselected training geometries, one frozen actor and paired seed labels. Different samplers consume random numbers differently. No learned improvement, independent full seed or final performance comparison.')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_frozen_sampler_physical_comparison',**costs,
                          physical_probe_support=supported,wall_s=time.perf_counter()-started))
        record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
            f'联合重尾条件噪声先验验证及冻结策略32次物理探索对照已完成，未做训练更新；预注册推进支持={supported}，不自动训练。两种采样使用同模型、原奖励/72维/物理规则与16条训练几何。结合24个协同后缀及真实训练探索覆盖继续判断机制，有限探针不得冒称学会内侧过渡或直接部署模板。条件高斯60k候选已停止，强v7/S2和封存测试保持，最终研究/论文未完成。')
        print(costs,'physical_probe_support',supported)
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,automatic_retry=False));raise


if __name__=='__main__':main()
