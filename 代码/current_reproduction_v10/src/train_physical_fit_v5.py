"""Fresh seed101 actor, fixed24000 supervised updates on witnessed states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import shutil
import time
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from strong_policy_v4 import StrongPolicyV4,STATIC_KEYS,extract_static_assets,tensor_hash,load_strong_policy
from train_strong_fit_v4 import equivalent_loss

OUT=ROOT/'results/physical_fit_v5/pilot_seed101/adaptive'
DATA=ROOT/'results/physical_teacher_v5/collection_001'
MODEL=ROOT/'baseline_records/supplementary_reproduction/models/main.pt'
MODEL_HASH='c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
DATA_HASHES={'base_teacher_records.npz':'c9403371f0b0ee8e158477e42f5b321a0925065f0427fb60c6e3bcd02cf193a2',
    'branch_teacher_records.npz':'9dae60b3e2b141bd5172aad3f80bbdb3b09d93cd5c29d9849c7a10a52cd54328'}


def main():
    started,cpu=time.perf_counter(),time.process_time()
    torch.set_num_threads(1);torch.manual_seed(101);torch.use_deterministic_algorithms(True)
    assert sha(MODEL)==MODEL_HASH
    assert all(sha(DATA/name)==digest for name,digest in DATA_HASHES.items())
    validation=json.loads((DATA/'completion_validation.json').read_text('utf-8'))
    assert validation['status']=='passed' and validation['every_command_interval_and_ordered_branch_gate_recomputed']
    OUT.mkdir(parents=True,exist_ok=False)
    with np.load(DATA/'base_teacher_records.npz') as b,np.load(DATA/'branch_teacher_records.npz') as n:
        eligible=n['eligible_nonterminal_state'].astype(bool)
        observations=np.vstack([b['observations'],n['observations'][eligible]])
        targets=np.vstack([b['targets_three'],n['targets_three'][eligible]])
        cases=np.r_[b['case'],n['case'][eligible]]
        row_map=dict(source_kind=np.r_[np.zeros(len(b['observations']),int),np.ones(int(eligible.sum()),int)],
            source_row=np.r_[np.arange(len(b['observations'])),np.flatnonzero(eligible)],case=cases)
    assert len(observations)==14495 and int(eligible.sum())==9662
    assert np.isfinite(observations).all() and np.isfinite(targets).all()
    np.savez_compressed(OUT/'training_row_map.npz',**row_map)
    x=torch.tensor(observations,dtype=torch.float32);y=torch.tensor(targets,dtype=torch.float32)
    groups=[np.flatnonzero(cases==case) for case in ('original','dense')]
    assets=extract_static_assets(MODEL)
    torch.save(assets,OUT/'shared_static_assets.pt')
    policy=StrongPolicyV4(101,assets,'adaptive')
    initial={name:value.detach().clone() for name,value in policy.named_parameters()}
    with torch.no_grad():
        cached=[policy.actor.features(part) for part in x.split(512)]
        preview,joint=[torch.cat([pair[i] for pair in cached]) for i in (0,1)]
    sources={name:sha(ROOT/'src'/name) for name in ('train_physical_fit_v5.py','strong_policy_v4.py','train_strong_fit_v4.py')}
    registration=dict(created_utc=utc(),version='v5_physical_state_fresh_fit_1',scope='Single adaptive development pilot',
        seed=101,arm='adaptive',architecture='StrongPolicyV4:14-64-64-3 tanh;96 preview/192 joint fixed Wendland features',
        fresh_trainable_parameters=True,initialization='All base weights/biases fresh seed101; correction weights zero; no historical fitted actor parameters copied',
        static_assets_sha256=sha(OUT/'shared_static_assets.pt'),static_asset_tensor_sha256=assets['tensor_sha256'],
        historical_static_and_teacher_checkpoint_sha256=MODEL_HASH,static_asset_keys=STATIC_KEYS,
        teacher_data_sha256=DATA_HASHES,teacher_registration_sha256=sha(DATA/'registration.json'),
        teacher_receipt_sha256=sha(DATA/'receipt.json'),teacher_validation_sha256=sha(DATA/'completion_validation.json'),
        dataset_rows=14495,base_rows=4833,active_branch_rows=9662,terminal_rows_retained_not_used=4,
        terminal_rule='Registered terminal states need no next policy action; retained in source dataset and all branch outcomes',
        case_samples=[len(g) for g in groups],training_row_map_sha256=sha(OUT/'training_row_map.npz'),
        minibatch=512,per_case_per_update=256,sampling='Uniform with replacement within each program; no outcome/failure/region weighting',
        sampler_seed=101,offline_iterations=24000,optimizer='Adam',learning_rate=2e-4,
        multipliers_after_iterations={'8000':.3,'16000':.3},gradient_norm_limit=1.,
        equivalent_action_loss_scales=[.015,.15,.3],boundary_margin=.02,
        checkpoints=[0,8000,16000,24000],selection='Fixed24000 final only; no trajectory-dependent checkpoint selection',
        trainable_tensor_count=len(initial),trainable_parameters=sum(p.numel() for p in policy.parameters()),
        initial_trainable_tensor_sha256=tensor_hash(initial),all_static_buffers_fixed=True,
        no_fixed_arm_training=True,new_ppo_updates=0,new_environment_interactions_during_fit=0,
        source_sha256=sources,wall_limit_s=900.,upstream_cost_scope='Physical teacher collection receipt is separate; historical static map/teacher discovery is shared and not fully reconstructed')
    write_json(OUT/'registration.json',registration);reg_hash=sha(OUT/'registration.json')
    rng=np.random.default_rng(101)
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=2e-4)
    scales=torch.tensor([.015,.15,.3]);summaries=[]
    def prediction(indices):return policy.actor.from_features(x[indices],preview[indices],joint[indices])
    def save(path,step):
        torch.save(dict(format='strong_policy_v4',policy=policy.state_dict(),optimizer=optimizer.state_dict(),seed=101,arm='adaptive',
            fixed_preview_mm=3.,version=registration['version'],offline_iterations=step,new_ppo_updates=0,
            registration_sha256=reg_hash,shared_static_assets_sha256=registration['static_assets_sha256'],
            dataset_sha256=DATA_HASHES,pretrained_trainable_parameters_copied=False,
            trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),sampling_rng_state=rng.bit_generator.state),path)
    def diagnose(step):
        with torch.no_grad():
            output=torch.cat([prediction(ids) for ids in torch.arange(len(x)).split(512)])
            delta=output.clamp(-1,1)-y
            summary=dict(iteration=step,full_equivalent_loss=float(equivalent_loss(output,y,scales)),
                clipped_rmse=delta.square().mean(0).sqrt().tolist(),maximum_clipped_error=delta.abs().max(0).values.tolist(),
                per_case_rmse={case:delta[indices].square().mean(0).sqrt().tolist() for case,indices in zip(('original','dense'),groups)},
                wall_s=time.perf_counter()-started)
        summaries.append(summary);write_json(OUT/'fit_log.json',summaries)
        return summary
    save(OUT/'initial.pt',0);diagnose(0)
    prep_wall=time.perf_counter()-started;fit_start,fit_cpu=time.perf_counter(),time.process_time()
    step=0;error=None
    try:
        with (OUT/'losses.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,24001):
                if time.perf_counter()-fit_start>900.:raise TimeoutError('Registered fitting wall ceiling')
                ids=torch.from_numpy(np.concatenate([g[rng.integers(len(g),size=256)] for g in groups]))
                loss=equivalent_loss(prediction(ids),y[ids],scales)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                rate=optimizer.param_groups[0]['lr']
                optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
                if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                if step in (8000,16000):
                    for group in optimizer.param_groups:group['lr']*=.3
                log.write(json.dumps(dict(iteration=step,loss=float(loss.detach()),learning_rate=rate,
                    gradient_norm_before_clip=float(gradient),fit_wall_s=time.perf_counter()-fit_start))+'\n')
                if step%2000==0:
                    log.flush();summary=diagnose(step)
                    print('PHYSICAL_FIT_V5',step,summary['clipped_rmse'],flush=True)
                if step in (8000,16000,24000):save(OUT/f'fit_{step:05d}.pt',step)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    if error is None and step==24000:shutil.copy2(OUT/'fit_24000.pt',OUT/'final.pt')
    else:save(OUT/'interrupted.pt',step)
    final=OUT/('final.pt' if error is None else 'interrupted.pt')
    loaded,_=load_strong_policy(final)
    with torch.no_grad():reload_error=max(float((loaded.mean(part)-policy.mean(part)).abs().max()) for part in x.split(512))
    changed={name:bool(not torch.equal(initial[name],value)) for name,value in policy.named_parameters()}
    assert all(changed.values()) and len(changed)==8 and reload_error==0.
    static_unchanged=all(torch.equal(getattr(policy.actor,name),assets['tensors'][name]) for name in STATIC_KEYS)
    assert static_unchanged
    unchanged=sha(MODEL)==MODEL_HASH and all(sha(ROOT/'src'/name)==value for name,value in sources.items()) and all(sha(DATA/name)==value for name,value in DATA_HASHES.items())
    assert unchanged
    receipt=dict(status='completed' if error is None and step==24000 else 'failed',error=error,seed=101,arm='adaptive',
        offline_iterations=step,supervised_sample_presentations=512*step,model_sha256=sha(final),
        initial_sha256=sha(OUT/'initial.pt'),registration_sha256=reg_hash,losses_sha256=sha(OUT/'losses.jsonl'),
        static_buffers_unchanged=static_unchanged,updated_trainable_tensors=changed,reload_max_error=reload_error,
        registered_sources_data_teacher_unchanged=unchanged,final_fit_summary=summaries[-1],
        preparation_wall_s=prep_wall,fit_wall_s=time.perf_counter()-fit_start,fit_cpu_s=time.process_time()-fit_cpu,
        total_wall_s=time.perf_counter()-started,total_cpu_s=time.process_time()-cpu,
        new_ppo_updates=0,new_environment_interactions=0,complete_path_performance_not_yet_evaluated=True,
        identity='Fresh seed101 physical-state supervised adaptive pilot; not c16d45 historical main or prior seed101 continuation')
    write_json(OUT/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
