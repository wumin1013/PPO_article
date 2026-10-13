"""One adaptive development continuation on the registered final-function teacher.

This is stage two of the same seed101 random-initialized pipeline, not another
independent training seed. Original fit and PPO branches are never overwritten.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import json
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from strong_policy_v4 import load_strong_policy, tensor_hash, STATIC_KEYS
from train_strong_fit_v4 import equivalent_loss

OUT=ROOT/'results/final_function_fit_v4/stage2_seed101/adaptive'
PARENT=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'
PARENT_HASH='188b02080069d7381a8ce9573ee936c89886e5b600e35f8e394e5f38f8256f94'
DATA=ROOT/'results/final_function_teacher_v4/preparation_002/adaptive_teacher_records.npz'
DATA_HASH='ff33674ba43a78a7367134f36436eb3c4d0752052738dfb516a40830f59dfa09'


def main():
    torch.set_num_threads(1)
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    assert sha(PARENT)==PARENT_HASH and sha(DATA)==DATA_HASH
    OUT.mkdir(parents=True,exist_ok=False)
    started,cpu=time.perf_counter(),time.process_time()
    policy,parent_metadata=load_strong_policy(PARENT)
    assert policy.seed==101 and policy.arm=='adaptive'
    policy.train()
    policy.actor.requires_grad_(True)
    with np.load(DATA) as d:
        x=torch.tensor(d['observations'],dtype=torch.float32)
        y=torch.tensor(d['targets_three'],dtype=torch.float32)
        groups=[np.flatnonzero((d['case']==case)&(d['neighborhood_sign']==0 if kind=='original' else d['neighborhood_sign']!=0))
                for case in ('original','dense') for kind in ('original','neighbor')]
    with torch.no_grad():
        values=[policy.actor.features(part) for part in x.split(512)]
        preview,joint=[torch.cat([item[i] for item in values]) for i in (0,1)]
    source_names=['train_final_function_fit_v4.py','strong_policy_v4.py','train_strong_fit_v4.py','prepare_final_function_teacher_v4b.py']
    registration=dict(created_utc=utc(),version='v4_final_function_fit_stage2_1',scope='ADAPTIVE DEVELOPMENT PILOT ONLY',
        seed=101,arm='adaptive',parent=str(PARENT.relative_to(ROOT)),parent_sha256=PARENT_HASH,
        parent_initialization='Same seed101 original12000 fit from independently random actor; this is continuation, not an additional independent model seed',
        prior_offline_iterations=12000,stage_iterations=12000,total_pipeline_offline_iterations=24000,new_ppo_updates=0,
        teacher_data=str(DATA.relative_to(ROOT)),teacher_data_sha256=DATA_HASH,
        teacher_source='Final strong c16d45 function on 6849 untouched historical observations plus 13698 fixed antithetic numerical neighbors',
        source_sample_count=6849,neighbor_sample_count=13698,total_sample_count=len(x),
        previous_teacher_preparation_sha256=sha(DATA.with_name('registration.json')),
        all_cases=['original','dense'],each_case_per_step=256,per_case_original_observations=128,per_case_neighbors=128,
        steps=12000,batch_size=512,optimizer='Fresh Adam state for the new supervised objective',learning_rate=6e-5,
        lr_multiplier_after_iteration8000=.3,equivalent_loss_scales=[.015,.15,.3],boundary_margin=.02,
        gradient_norm_limit=1.,all_eight_actor_parameter_tensors_trainable=True,all_static_feature_buffers_fixed=True,
        rng_seed=500101,checkpoints=[4000,8000,12000],selection='Fixed12000 stage-final; no success-conditioned checkpoint selection',
        observation_scope='Numerical neighbors are not asserted physically reachable; fixed guide-equivalent labels not prepared and no fixed-arm claim',
        no_failure_observations_added=True,no_failure_specific_weighting=True,no_new_test_data=True,
        source_sha256={name:sha(ROOT/'src'/name) for name in source_names},
        initial_trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),
        initial_costs='Parent fit and historical teacher preparation costs remain in their separate receipts; this receipt measures only the additional stage',
        wall_limit_s=1680.,stopping_rule='Exactly12000 iterations or nonfinite/error/wall limit; preserve interrupted model and all losses')
    write_json(OUT/'registration.json',registration)
    reg_hash=sha(OUT/'registration.json')
    rng=np.random.default_rng(500101)
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=6e-5)
    scales=torch.tensor([.015,.15,.3])
    summaries=[]

    def predict(ids):
        return policy.actor.from_features(x[ids],preview[ids],joint[ids])

    def save(path,step):
        torch.save(dict(format='strong_policy_v4',policy=policy.state_dict(),optimizer=optimizer.state_dict(),
            seed=101,arm='adaptive',fixed_preview_mm=3.,version=registration['version'],
            offline_iterations=12000+step,stage_iterations=step,new_ppo_updates=0,
            parent_checkpoint_sha256=PARENT_HASH,dataset_sha256=DATA_HASH,registration_sha256=reg_hash,
            trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),
            sampling_rng_state=rng.bit_generator.state,shared_static_assets_sha256=parent_metadata['shared_static_assets_sha256']),path)

    def diagnose(step,loss=None):
        with torch.no_grad():
            predictions=torch.cat([predict(ids) for ids in torch.arange(len(x)).split(512)])
            delta=predictions.clamp(-1,1)-y.clamp(-1,1)
            entry=dict(iteration=step,sampled_loss=loss,full_teacher_equivalent_loss=float(equivalent_loss(predictions,y,scales)),
                original_rmse=delta[:6849].square().mean(0).sqrt().tolist(),
                neighbor_rmse=delta[6849:].square().mean(0).sqrt().tolist(),
                original_maximum_error=delta[:6849].abs().max(0).values.tolist(),
                neighbor_maximum_error=delta[6849:].abs().max(0).values.tolist(),
                wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu)
        summaries.append(entry);write_json(OUT/'fit_log.json',summaries)
        return entry

    save(OUT/'initial.pt',0)
    baseline=diagnose(0)
    step=0
    error=None
    try:
        with (OUT/'losses.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,12001):
                if time.perf_counter()-started>1680.:
                    raise TimeoutError('Registered second-fit wall limit')
                ids=torch.from_numpy(np.concatenate([g[rng.integers(len(g),size=128)] for g in groups]))
                prediction=predict(ids);loss=equivalent_loss(prediction,y[ids],scales)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                rate=optimizer.param_groups[0]['lr']
                optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
                if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                if step==8000:
                    for group in optimizer.param_groups:group['lr']*=.3
                log.write(json.dumps(dict(iteration=step,loss=float(loss.detach()),lr=rate,
                    gradient_norm_before_clip=float(gradient),wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu))+'\n')
                if step%500==0:
                    log.flush();summary=diagnose(step,float(loss.detach()))
                    print('FINAL_FUNCTION_STAGE2',step,summary['original_rmse'],summary['neighbor_rmse'],flush=True)
                if step in (4000,8000,12000):save(OUT/f'fit_{step:05d}.pt',step)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    final=OUT/'final.pt'
    save(final,step)
    loaded,_=load_strong_policy(final)
    with torch.no_grad():reload_error=float(abs(loaded.mean(x[:512])-policy.mean(x[:512])).max())
    parent=torch.load(PARENT,map_location='cpu',weights_only=False)['policy']
    buffers_unchanged=all(torch.equal(parent['actor.'+key],policy.state_dict()['actor.'+key]) for key in STATIC_KEYS)
    result=dict(status='completed' if error is None and step==12000 else 'failed',error=error,
        seed=101,arm='adaptive',stage_iterations=step,total_pipeline_offline_iterations=12000+step,new_ppo_updates=0,
        sampled_records_this_stage=512*step,model_sha256=sha(final),initial_sha256=sha(OUT/'initial.pt'),
        parent_sha256=PARENT_HASH,parent_unchanged=sha(PARENT)==PARENT_HASH,dataset_unchanged=sha(DATA)==DATA_HASH,
        source_unchanged=all(sha(ROOT/'src'/n)==h for n,h in registration['source_sha256'].items()),
        static_buffers_unchanged=buffers_unchanged,reload_max_error=reload_error,
        registration_sha256=reg_hash,losses_sha256=sha(OUT/'losses.jsonl'),baseline=baseline,final_fit_summary=summaries[-1],
        total_wall_s=time.perf_counter()-started,total_cpu_s=time.process_time()-cpu,
        new_trajectory_executions=0,trajectory_completion_or_efficiency_not_yet_evaluated=True,
        independence_scope='Same seed101 pipeline continuation; not an independent seed or old88PPO reconstruction')
    write_json(OUT/'receipt.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
