"""Analyze registered development runs without loading any held-out program."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from common import ROOT, sha, write_json
from model_io import load_model

OUT=ROOT/'results'/'development_v2'/'learning_001'


def read(path):return json.loads(path.read_text('utf-8'))


def main(extension=False):
    selection=read(OUT/'teacher_selection.json')
    teacher=[]
    for item in selection['candidates']:
        teacher.append(dict(id=item['candidate']['id'],successes=item['successes'],runs=item['runs'],
            valid_for_selection=item['all_development_paths_pass'],validation_sum_s=item['validation_sum_s'],
            failed_paths=[dict(path=r['path'],reason=r['reason']) for r in item['records'] if not r['success']]))
    rows=[];budgets=[];paired=[];features=[]
    with np.load(OUT/'shared_teacher_records.npz') as data:
        training_x=data['observations'].astype(np.float32)
        expected_scale=np.maximum(training_x.std(0),.05)
    roots=[OUT]+([OUT.with_name('learning_002')] if extension else [])
    for folder in [folder for root in roots for folder in sorted((root/'paired').glob('seed_*'))]:
        initial={}
        for arm in ('fixed','adaptive'):
            stage=folder/arm
            if not (stage/'receipt.json').exists():continue
            receipt=read(stage/'receipt.json');episodes=read(stage/'episodes.json')
            budgets.append(dict(seed=int(folder.name.split('_')[1]),arm=arm,**receipt,
                episode_counts=dict(finished=sum(e['complete'] for e in episodes),
                    executor_successes=sum(e['executor_success'] is True for e in episodes),
                    failed=sum(e['executor_success'] is False for e in episodes),unfinished=sum(not e['complete'] for e in episodes)),
                training_failure_reasons=dict(Counter(e['reason'] for e in episodes if e['executor_success'] is False))))
            for file in sorted(stage.glob('validation_*.json')):
                item=read(file)
                for result in item['records']:
                    rows.append(dict(seed=int(folder.name.split('_')[1]),arm=arm,step=item['step'],**result))
            policy,saved=load_model(stage/'final.pt')
            centers=policy.actor.centers.numpy();scale=policy.actor.scale.numpy();radii=policy.actor.radii.numpy()
            feature_check=dict(seed=int(folder.name.split('_')[1]),arm=arm,
                centers_count=len(centers),every_center_is_training_observation=all(np.any(np.all(training_x==c,axis=1)) for c in centers),
                scale_matches_training_only_std_floor=bool(np.array_equal(scale,expected_scale)),
                radii_are_prespecified_two=bool(np.all(radii==2.)),
                allocated_trainable_parameters=sum(p.numel() for p in policy.parameters()),
                actor_trainable_parameters=sum(p.numel() for p in policy.actor.parameters()),
                active_action_count=2 if arm=='fixed' else 3)
            if not all(feature_check[key] for key in ('every_center_is_training_observation','scale_matches_training_only_std_floor','radii_are_prespecified_two')):
                raise AssertionError('Training-only feature provenance check failed')
            features.append(feature_check)
            probe=torch.load(stage/'final.pt',map_location='cpu',weights_only=False)['policy']
            if not all(torch.equal(value,probe[key]) for key,value in policy.state_dict().items()):raise AssertionError('Loader changed saved tensors')
            initial[arm]=torch.load(stage/'checkpoint_00000.pt',map_location='cpu',weights_only=False)
        if set(initial)=={'fixed','adaptive'}:
            a,b=initial['fixed'],initial['adaptive']
            paired.append(dict(seed=int(folder.name.split('_')[1]),
                all_initial_tensors_identical=all(torch.equal(value,b['policy'][key]) for key,value in a['policy'].items()),
                torch_rng_states_identical=torch.equal(a['torch_rng_state'],b['torch_rng_state']),
                minibatch_rng_states_identical=a['batch_rng_state']==b['batch_rng_state'],
                path_rng_states_identical=a['path_rng_state']==b['path_rng_state']))
    baselines=[]
    for method in ('s2_v3','scmo_v2'):
        for name in ('validation_trapezoid','validation_zigzag'):
            file=ROOT/'results'/'deterministic'/f'{method}_t1.25_v100.0_a2000.0_j62500.0_{name}_0.001.json'
            r=read(file)
            baselines.append(dict(method=method,path=name,success=r['success'],complete_time_s=r['complete_time_s'],
                result_file=str(file.relative_to(ROOT)),result_sha256=sha(file)))
    old=[]
    for arm in ('fixed','adaptive'):
        for name in ('validation_trapezoid','validation_zigzag'):
            files=[ROOT/'results'/'validation'/f'fullrange_{arm}_101_{name}_0.001.json']
            for file in files:
                r=read(file)
                if 'complete_time_s' in r:old.append(dict(arm=arm,path=name,complete_time_s=r['complete_time_s'],success=r['success'],result_file=str(file.relative_to(ROOT))))
    summaries=[];differences=[]
    seed_ids=[101,211,307,401,503] if extension else [101]
    for name in ('validation_trapezoid','validation_zigzag'):
        arms={}
        for arm in ('fixed','adaptive'):
            records=[r for r in rows if r['path']==name and r['arm']==arm and r['step']==8192]
            values=[r['complete_time_s'] for r in records if r['success']]
            arms[arm]={r['seed']:r for r in records}
            summaries.append(dict(path=name,arm=arm,expected=len(seed_ids),attempts=len(records),successes=len(values),
                missing_registered_seeds=[seed for seed in seed_ids if seed not in arms[arm]],
                successful_time_mean_s=float(np.mean(values)) if values else None,
                successful_time_sd_s=float(np.std(values,ddof=1)) if len(values)>1 else None,
                failures=[dict(seed=r['seed'],reason=r['reason']) for r in records if not r['success']]))
        joint=[seed for seed in seed_ids if all(seed in arms[arm] and arms[arm][seed]['success'] for arm in arms)]
        delta=np.array([arms['fixed'][seed]['complete_time_s']-arms['adaptive'][seed]['complete_time_s'] for seed in joint])
        half=float(student_t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
        differences.append(dict(path=name,jointly_successful_seeds=joint,paired_difference_s=delta.tolist(),
            mean_fixed_minus_adaptive_s=float(delta.mean()) if len(delta) else None,
            conditional_student_95_ci_s=[float(delta.mean()-half),float(delta.mean()+half)] if half is not None else None))
    result=dict(scope='DEVELOPMENT ONLY; no prospective or old final-test programs were loaded',
        teacher=teacher,selected=selection.get('selected'),training_records=selection.get('training_records'),
        teacher_wall_s=selection.get('teacher_wall_s'),teacher_actor_calls=selection.get('recorded_teacher_actor_calls'),
        validation=rows,final_validation_summary=summaries,conditional_development_differences=differences,
        training=budgets,paired_initialization_checks=paired,training_only_feature_checks=features,deterministic_validation=baselines,
        aggregate_budget=dict(complete_training_runs=sum(r['complete'] for r in budgets),registered_training_runs=10 if extension else 2,
            teacher_records_shared=selection.get('training_records'),teacher_search_wall_s=selection.get('teacher_wall_s'),
            teacher_search_calls=selection.get('recorded_teacher_actor_calls'),
            sum_internal_training_wall_s=sum(r['cost']['total_wall_s'] for r in budgets),
            training_interactions=sum(r['cost']['training_interactions'] for r in budgets),
            physics_ticks=sum(r['cost']['physics_ticks'] for r in budgets),
            supervised_optimizer_steps=sum(r['cost']['supervised_optimizer_steps'] for r in budgets),
            supervised_samples=sum(r['cost']['supervised_samples'] for r in budgets),
            ppo_optimizer_steps=sum(r['cost']['ppo_optimizer_steps'] for r in budgets),
            ppo_samples=sum(r['cost']['ppo_samples'] for r in budgets),
            validation_wall_s=sum(r['cost']['validation_wall_s'] for r in budgets),
            wall_interpretation='Internal wall intervals include fitting, training and checkpoint validation; extension runs used at most two concurrent processes, so summed wall is not elapsed project time or CPU time.'),
        old_development_validation=old,
        metadata_correction='Per-arm registration labels action_sampling with the nominal experiment seed. Policy.__init__ then calls torch.manual_seed(seed+100000) before critic initialization; stochastic PPO actions continue that stream after critic construction. Checkpoint torch_rng_state preserves the exact stream. Initialization, fitting batches, and paired starting RNG states are checked here.',
        interpretation='The teacher and PPO budget both changed from v1; this protocol cannot attribute their combined improvement exclusively to longer PPO. Compare 0/4096/8192 checkpoints within the same protocol for development evidence only. Validation outcomes and conditional paired intervals are development diagnostics, not unbiased final-test estimates. A single seed does not establish independent training stability; completed registered seed counts are stated explicitly.')
    destination=roots[-1]
    write_json(destination/'analysis.json',result)
    lines=['# Registered development learning pilot','',result['scope'],'',
        'The six teacher candidates were registered before execution. All successful and failed runs are retained. The shared teacher is selected only if all four training and both validation programs pass the unchanged independent audit.','',
        '| Teacher | Complete successes | Validation total (s) | Eligible |',
        '|---|---:|---:|---|']
    for r in teacher:lines.append(f"| {r['id']} | {r['successes']}/{r['runs']} | {r['validation_sum_s']:.3f} | {r['valid_for_selection']} |")
    lines+=['','## Validation checkpoints','','| Seed | Arm | Decisions | Program | Complete | Time (s) / reason |','|---:|---|---:|---|---|---|']
    for r in rows:
        outcome=f"{r['complete_time_s']:.3f}" if r['success'] else r['reason']
        lines.append(f"| {r['seed']} | {r['arm']} | {r['step']} | {r['path']} | {r['success']} | {outcome} |")
    lines+=['','## Final development summary','','| Arm | Program | Complete successes | Successful time mean (s) | SD (s) |','|---|---|---:|---:|---:|']
    for r in summaries:
        mean='NA' if r['successful_time_mean_s'] is None else f"{r['successful_time_mean_s']:.3f}"
        sd='NA' if r['successful_time_sd_s'] is None else f"{r['successful_time_sd_s']:.3f}"
        lines.append(f"| {r['arm']} | {r['path']} | {r['successes']}/{r['expected']} | {mean} | {sd} |")
    lines+=['','## Deterministic validation context','','| Method | Program | Time (s) |','|---|---|---:|']
    for r in baselines:lines.append(f"| {r['method']} | {r['path']} | {r['complete_time_s']:.3f} |")
    lines+=['','## Reproducibility and limits','',result['metadata_correction'],'',result['interpretation'],'',
        'Both arms use the same selected teacher records, the same 192 training-derived Gaussian features, the same fitted starting tensors, 2,000 fitting updates and 8,192 PPO decisions. The fixed arm masks the third action. Every completed, failed and budget-truncated training episode is saved; training executor labels are distinct from independently audited validation success.','',
        'All model loading checks use the unchanged version-aware model_io.load_model. Source, model and dataset hashes are retained. The initial default AFC Python launch failed on duplicate OpenMP runtimes before execution; the original PPO environment is used without unsafe duplicate-library overrides.','',
        'See analysis.json for all individual outcome, failure, timing, budget and paired initialization checks.']
    (destination/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(teacher=teacher,paired=paired,validation=[dict(seed=r['seed'],arm=r['arm'],step=r['step'],path=r['path'],success=r['success'],time=r['complete_time_s']) for r in rows]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--extension',action='store_true')
    main(parser.parse_args().extension)
