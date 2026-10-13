"""A fixed 36-run exploration probe; no PPO updates or sealed paths."""
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from study_io import BASE,register,receipt,write_json,sha
from joint_policy import JointPolicy
from smooth_policy_candidate import SmoothExplorationPolicy
from joint_runtime_v5 import environment_factory
from train_pilot import save_torch
from validate_actions import difference_audit


def main():
    source=BASE/'fixed3_remaining_credit_training_v5/final.pt'
    paths_file=BASE/'learning_preparation_v4/paths.json'
    stage=register('smooth_exploration_physical_probe_v1',[__file__,source,paths_file,
        BASE/'smooth_policy_candidate.py',BASE/'smooth_exploration_candidate_validation_v1/receipt.json',
        BASE/'joint_policy.py',BASE/'joint_runtime_v5.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v2.py',
        BASE/'joint_runtime.py',BASE/'terminal_candidate_runtime_v1.py',BASE/'geometry_spec.py',
        BASE/'validate_actions.py',BASE/'train_pilot.py'],
        dict(runs=36,training_path_indices=[10,11,14,15,18,19],seeds=[7,19,31],
            samplers=['independent','gsde8'],latent_std=[.03,.10,.30],D_initial=4.6,
            actor_calls_max=36000,physics_ticks_max=900000,wall_s=1800,ppo_updates=0),
        'Compare only the temporal structure of exploration at the same learned mean and marginal std on six existing training paths; no trained-performance claim.')
    external=Path('D:/PPO_article_joint_runs/rolling_states/smooth_exploration_physical_probe_v1')
    external.mkdir(parents=True,exist_ok=False)
    write_json(stage/'storage.json',dict(directory=str(external),automatic_resume=False))
    torch.set_num_threads(1)
    incoming=torch.load(source,map_location='cpu',weights_only=False)
    prepared=json.loads(paths_file.read_text(encoding='utf-8'))
    costs=dict(source_model_loads=1,policy_instances=0,actor_calls=0,physics_ticks=0,ppo_updates=0)
    results=[]
    started=time.perf_counter()
    current=None
    try:
        for sampler in ['independent','gsde8']:
            for index in [10,11,14,15,18,19]:
                for seed in [7,19,31]:
                    current=f'{sampler}_path{index}_seed{seed}'
                    torch.manual_seed(seed)
                    policy=SmoothExplorationPolicy(72) if sampler=='gsde8' else JointPolicy(72)
                    costs['policy_instances']+=1
                    if sampler=='gsde8':policy.load_base_policy(incoming['policy'])
                    else:policy.load_state_dict(incoming['policy'])
                    with torch.no_grad():
                        policy.active_mask.fill_(1.)
                        policy.actor[-1].weight[2].zero_()
                        policy.actor[-1].bias[2]=math.atanh((4.6-6.4)/5.6)
                        policy.log_std.copy_(torch.log(torch.tensor([.03,.10,.30])))
                    # Reset after construction so the sampling seed is documented.
                    torch.manual_seed(seed)
                    ref=np.asarray(prepared['training'][index],float)
                    env=environment_factory(ref)
                    chunks=[];actions=[]
                    def checkpoint():
                        save_torch(external/'resume.pt',dict(case=current,policy=policy.state_dict(),
                            environment=env.state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                            rows=chunks,actions=actions,completed_results=results,costs=costs,
                            model_sha256=sha(source),phase='running'))
                    checkpoint()
                    while not env.done:
                        if time.perf_counter()-started>1800 or costs['actor_calls']>=36000 or costs['physics_ticks']>=900000:
                            raise TimeoutError('Registered exploration-probe budget exhausted')
                        action,_,_=policy.act(env.observation())
                        costs['actor_calls']+=1
                        _,_,_,info=env.step(action)
                        costs['physics_ticks']+=len(info['rows'])
                        chunks.append(info['rows']);actions.append(info['held_action'])
                        checkpoint()
                    rows=np.vstack(chunks);actions=np.asarray(actions)
                    audit=difference_audit(ref[0],rows)
                    success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                    value=dict(case=current,sampler=sampler,path_index=index,seed=seed,success=success,
                        reason=env.reason,elapsed_s=env.ticks*.001,complete_time_s=env.ticks*.001 if success else None,
                        geometry=env.tracker.snapshot(1.25),audit=audit,
                        mean_absolute_D_step=float(np.mean(np.abs(np.diff(actions[:,2])))),
                        D_quantiles=np.quantile(actions[:,2],[.1,.5,.9]).tolist())
                    results.append(value)
                    np.savez_compressed(stage/(current+'.npz'),reference=ref,rows=rows,actions=actions)
                    write_json(stage/(current+'.json'),value)
                    write_json(stage/'progress.json',dict(completed=len(results),costs=costs,current=current))
                    print(value,flush=True)
        write_json(stage/'results.json',dict(results=results,costs=costs,
            trained_performance=False,sealed_test=False,automatic_budget_extension=False))
        receipt(stage,dict(status='bounded_exploration_probe_completed',runs=36,costs=costs,
            wall_s=time.perf_counter()-started,goal_complete=False))
        write_json(external/'resume_pointer.json',dict(phase='complete',complete=True,automatic_resume_forbidden=True))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),current=current,costs=costs,
            completed_runs=len(results),automatic_retry=False))
        raise


if __name__=='__main__':main()
