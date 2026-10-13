"""Final bounded algorithm round: independently seeded matched continuation searches."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_bend_residual_policy import BendResidualPolicy
from paper_complete_evaluation import evaluate
from complete_corner_metrics import complete_corner_metrics
from study_io import BASE,register,receipt,sha,write_json

SEEDS=[921101,921102,921103]
ARMS=['joint','fixed_preview','no_steering','constant_feed']
GRID=[.8,1.5,3.,4.6,6.,8.,12.]
LOW=np.array([-.16,-2.,-2.]);HIGH=np.array([.16,1.,1.])


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1);review=BASE/'paper_evidence_review'
    frozen=read(review/'frozen_evaluation_candidate.json');model=BASE/frozen['model']
    assert sha(model)==frozen['sha256']
    parent=torch.load(model,map_location='cpu',weights_only=False)
    upper=read(review/'nonlocal_state_diagnostic/results.json')['training_max'][0]
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    refs=[np.asarray(paths[3],float),np.asarray(paths[11],float),
          np.array([[0.,0.],[90.,0.],[90.,40.]])]
    stage=register('paper_evidence_review/bend_residual_learning',
        [__file__,BASE/'paper_bend_residual_policy.py',BASE/'paper_state_support_policy.py',
         BASE/'paper_complete_evaluation.py',BASE/'complete_corner_metrics.py',
         review/'frozen_evaluation_candidate.json',review/'support_repair/receipt.json',
         BASE/'learning_preparation_v4/paths.json'],
        dict(algorithm_round=2,total_training_search_wall_s=1800,seeds=SEEDS,arms=ARMS,
            train_refs=[x.tolist() for x in refs],adaptive_candidates_per_seed_arm=10,
            fixed_preview_extra_distance_trials=7,fixed_grid_mm=GRID,
            parameter_lower=LOW.tolist(),parameter_upper=HIGH.tolist(),
            maximum_complete_episode_attempts=423,ppo_updates=0,offline_fit_updates=0,sealed_paths=0),
        'Final algorithm round. Train a three-coefficient visible-bend residual by complete-path '
        'parameter search, retaining all failures. Three independent seeded continuation searches '
        'per arm start from a shared immutable PPO/fitting/search-derived backbone; these are NOT '
        'independent from-scratch PPO pipelines. Each arm receives10 adaptive candidates on identical '
        'three full training paths. Fixed preview additionally receives all7 global distances before '
        'the10 refinement candidates. It has extra selection opportunity and cost, not a smaller budget. '
        'Select lexicographically by success count, failure progress, then complete time and small '
        'shape cost. No development or sealed outcomes used for model selection. No per-path D, '
        'trajectory template, action probe deployed, or further algorithm round. Evaluate all selected '
        'seeds/arms afterward with failures retained. Primary joint seed is921101, never best-of-seeds.')
    started=time.perf_counter();deadline=started+1800
    runs=[];stop=None
    for seed in SEEDS:
        for arm_index,arm in enumerate(ARMS):
            rng=np.random.default_rng(np.random.SeedSequence([seed,arm_index]))
            folder=stage/f'{arm}_{seed}';folder.mkdir()
            policy=BendResidualPolicy(load_credit_policy(parent),upper,np.zeros(3),arm)
            trials=[];best=None
            total=17 if arm=='fixed_preview' else 10
            for trial in range(total):
                if time.perf_counter()>=deadline:
                    stop='registered_1800s_training_search_limit';break
                is_grid=arm=='fixed_preview' and trial<7
                adaptive=trial-7 if arm=='fixed_preview' else trial
                if is_grid:
                    theta=np.zeros(3);fixed=GRID[trial]
                elif adaptive==0:
                    theta=np.zeros(3) if best is None else np.asarray(best['coefficients']);fixed=best['fixed_d'] if arm=='fixed_preview' else None
                elif adaptive==1:
                    theta=np.array([-.025,-.6,-.65]);fixed=best['fixed_d'] if arm=='fixed_preview' else None
                else:
                    center=np.asarray(best['coefficients'])
                    scale=np.array([.065,.65,.65])*(1.-.045*adaptive)
                    theta=np.clip(center+rng.normal(size=3)*scale,LOW,HIGH)
                    fixed=best['fixed_d'] if arm=='fixed_preview' else None
                if arm=='no_steering':theta[0]=0.
                if arm=='constant_feed':theta[1]=0.
                if arm=='fixed_preview':theta[2]=0.
                # Exact repeated candidates reuse their complete training receipts, never re-simulate.
                duplicate=next((r for r in trials if r['fixed_d']==fixed and np.array_equal(r['coefficients'],theta)),None)
                if duplicate is not None:
                    record=dict(duplicate,trial=trial,reused_trial=duplicate['trial'])
                    trials.append(record);continue
                policy.coefficients=theta
                records=[]
                for k,ref in enumerate(refs):
                    env=environment_factory(ref);env.fixed_d=fixed
                    path=folder/f'trial{trial}_path{k}.npz'
                    result=evaluate(policy,env,path,deadline,max_calls=1000)
                    result['trace']=str(path.relative_to(BASE))
                    if result['success']:
                        z=np.load(path)
                        shape=complete_corner_metrics(ref,z['rows'])
                        result['shape']=shape
                    records.append(result)
                    if result['error']:
                        stop=result['error'];break
                if len(records)!=3 or stop:break
                failures=sum(not r['success'] for r in records)
                progress=sum(1-r['geometry']['gates_passed']/max(r['geometry']['gate_count'],1)
                             for r in records if not r['success'])
                complete=sum(r['complete_time_s'] for r in records if r['success'])
                soft=sum(.002*r['shape']['full_physical_reverse_deg']+
                         .1*r['shape']['out_edge_outward_mm'] for r in records if r['success'])
                key=[failures,progress,complete+soft]
                record=dict(trial=trial,coefficients=theta.tolist(),fixed_d=fixed,selection_key=key,records=records)
                trials.append(record)
                if best is None or key<best['selection_key']:best=record
                write_json(folder/'search.json',dict(trials=trials,best_trial=best['trial']))
                print('SEARCH',seed,arm,trial,'failures',failures,'best',best['selection_key'],flush=True)
            complete_run=len(trials)==total and stop is None
            run=dict(seed=seed,arm=arm,complete_search=complete_run,planned_candidates=total,
                     completed_candidates=len(trials),best=best)
            if best is not None:
                spec=dict(architecture='visible_bend_residual_over_frozen_backbone',
                    backbone=frozen['model'],backbone_sha256=frozen['sha256'],upper=upper,
                    coefficients=best['coefficients'],arm=arm,fixed_d=best['fixed_d'],seed=seed,
                    parameter_search_complete=complete_run,selection_used_development=False,
                    selection_used_sealed=False,from_scratch_ppo_seed=False)
                write_json(folder/'selected_policy.json',spec)
                run['selected_policy']=str((folder/'selected_policy.json').relative_to(BASE))
                run['selected_policy_sha256']=sha(folder/'selected_policy.json')
            runs.append(run)
            write_json(stage/'results.json',dict(runs=runs,stop_reason=stop,wall_s=time.perf_counter()-started))
            if stop:break
        if stop:break
    calls=ticks=0
    for file in stage.glob('*/trial*_path*.json'):
        r=read(file);calls+=r['actor_calls'];ticks+=r['physics_ticks']
    receipt(stage,dict(status='completed_final_bounded_parameter_learning' if not stop else 'stopped_final_bounded_parameter_learning',
        algorithm_round=2,algorithm_rounds_used=2,algorithm_rounds_cap=2,
        completed_searches=sum(r['complete_search'] for r in runs),planned_searches=12,
        independent_continuation_seeds=3,independent_from_scratch_ppo_seeds=0,
        actor_calls=calls,physics_ticks=ticks,wall_s=time.perf_counter()-started,
        parameter_search=True,ppo_updates=0,offline_fitting_updates=0,
        results_sha256=sha(stage/'results.json'),stop_reason=stop,
        further_algorithm_optimization_allowed=False,sealed_test_opened=False,formal_adoption=False))
    print('LEARNING_FINISHED',len(runs),stop,flush=True)


if __name__=='__main__':main()
