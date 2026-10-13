"""Verify persisted paired pilot outputs and generate an internal result report."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from recovery_training_v6 import OUT,REG,read,arrays,verify_registration
from strong_policy_v4 import load_strong_policy,tensor_hash
from metrics import metrics


def main():
    started,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1)
    reg=verify_registration();receipts={};results=[];cost={};registrations={}
    for arm in reg['arms']:
        folder=OUT/('seed101_'+arm);receipt=read(folder/'receipt.json');receipts[arm]=receipt
        registrations[arm]=read(folder/'registration.json')
        assert receipt['status']=='completed' and receipt['iterations']==24000
        assert len((folder/'losses.jsonl').read_text('utf-8').splitlines())==24000
        assert sha(folder/'final.pt')==receipt['model_sha256']==sha(folder/'fit_24000.pt')
        policy,saved=load_strong_policy(folder/'final.pt')
        initial,_=load_strong_policy(folder/'initial.pt')
        assert tensor_hash(dict(initial.named_parameters()))==receipt['initial_trainable_tensor_sha256']
        assert saved['registration_sha256']==sha(REG) and saved['new_ppo_updates']==0
        evaluation=read(folder/'evaluation/registration.json')
        assert evaluation['model_sha256']==receipt['model_sha256'] and evaluation['runs']==reg['evaluation_runs']
        for run in reg['evaluation_runs']:
            stem=run['case']+f"_dt{round(run['dt_s']*1e6)}us"
            path=folder/'evaluation'/(stem+'.npz');result=read(path.with_suffix('.json'));trace=arrays(path)
            assert result['trace_sha256']==sha(path) and result['model_sha256']==receipt['model_sha256']
            assert result['registration_sha256']==sha(REG) and result['dt_s']==run['dt_s']
            assert result['partial_decision_position_samples_retained']==0
            assert np.array_equal(trace['rows'][:,:2],trace['positions'][1:])
            case=next(x for x in reg['cases'] if x['id']==run['case'])
            assert sha(ROOT/case['reference'])==result['reference_sha256']==case['reference_sha256']
            recomputed=metrics(trace['reference'],trace['positions'],run['dt_s'],case['evaluation_tolerance_mm'],
                declared_success=result['executor_reason']=='success',reason=result['executor_reason'])
            for key in ('success','complete_time_s','elapsed_prefix_s','vector_peaks','geometry','rest_pass'):
                assert recomputed[key]==result[key], (arm,stem,key)
            obs=torch.tensor(trace['observations'],dtype=torch.float32);support={}
            with torch.no_grad():
                features=[policy.actor.features(part) for part in obs.split(256)]
                for j,name in enumerate(('preview','joint')):
                    values=torch.cat([pair[j] for pair in features]).max(1).values
                    support[name]=dict(observations=len(obs),zero_support=int((values==0).sum()),
                        terminal_six_zero_support=int((values[-6:]==0).sum()),minimum_maximum_activation=float(values.min()))
            results.append(dict(arm=arm,path=run['case'],dt_s=run['dt_s'],success=result['success'],
                complete_time_s=result['complete_time_s'],elapsed_prefix_s=result['elapsed_prefix_s'],reason=result['reason'],
                vector_limit_ratios=result['vector_limit_ratios'],max_polyline_deviation_mm=result['max_polyline_deviation_mm'],
                ordered_gate_frontier=result['geometry']['gates_passed'],terminal_rest_s=result['terminal_rest_s'],
                feature_support=support,trace_sha256=sha(path),result_sha256=sha(path.with_suffix('.json')),
                observed_wall_s=result['wall_s'],online_components=result['cost']))
        cost[arm]=dict(preparation_wall_s=receipt['preparation_wall_s'],fitting_wall_s=receipt['fitting_wall_s'],
            total_fit_process_wall_s=receipt['total_wall_s'],fitting_cpu_s=receipt['fitting_cpu_s'],
            total_evaluation_wall_s=sum(x['observed_wall_s'] for x in results if x['arm']==arm))
    for key in ('initial_trainable_tensor_sha256','sampled_row_stream_sha256','feature_tensor_sha256'):
        assert receipts['adaptive'][key]==receipts['fixed'][key]
    for key in ('training_data_sha256','static_asset_sha256','allocation_parameters'):
        assert registrations['adaptive'][key]==registrations['fixed'][key]
    collection=read(OUT/'collection/receipt.json');dataset=read(OUT/'training_data/receipt.json')
    validation=read(OUT/'collection/completion_validation.json')
    inherited_rows=dataset['rows']-dataset['recovery_active_rows']
    adaptive_dense_fine=next(r for r in results if r['arm']=='adaptive' and r['path']=='dense' and r['dt_s']==.0005)
    result=dict(created_utc=utc(),status='passed',source_sha256=sha(Path(__file__)),registration_sha256=sha(REG),
        both_arms_complete_fixed_24000=True,all_registered_runs_retained=len(results)==16,
        identical_initial_trainable_tensors=True,identical_sampled_row_stream=True,identical_shared_maps_and_data=True,
        initial_trainable_tensor_sha256=receipts['adaptive']['initial_trainable_tensor_sha256'],
        sampled_row_stream_sha256=receipts['adaptive']['sampled_row_stream_sha256'],
        independent_saved_trajectory_metric_recomputation=True,statistical_unit='One paired seed; no multi-seed interval estimated',
        completed_runs={arm:sum(r['success'] for r in results if r['arm']==arm) for arm in reg['arms']},
        results=results,collection_cost={key:value for key,value in collection.items() if key.endswith('_s')},
        feature_preparation_wall_s=dataset['preparation_wall_s'],cost=cost,
        inherited_cost_scope=f"Historical c16 teacher construction and prior {inherited_rows:,}-row preparation remain shared upstream costs. They are not repeated seed-wise and are not included in this incremental pilot total; full historical teacher discovery cost is unavailable.",
        fit_diagnostics={arm:receipts[arm]['final_fit'] for arm in reg['arms']},
        audit_wall_s=time.perf_counter()-started,audit_cpu_s=time.process_time()-cpu)
    write_json(ROOT/'audit/recovery_training_v6_validation.json',result)
    lines=['# Recovery training v6: bounded paired development pilot','',
        'This is fresh-student supervised fitting conditional on the shared c16 historical teacher. It is not a reconstructed historical PPO chain, a five-seed experiment, or a final-test result. No main model, Figure 8, manuscript, or sealed holdout was changed.','',
        '## Concrete repairs and registered design','',
        f"- Decode the teacher request before perturbing physical feed and steering. All {validation['physical_feed_changed_events']:,} command events changed feed; raw-output saturation can no longer cancel the registered feed perturbation.",
        f"- Rebuild feature centers, scales and radii using training observations only. Both heads cover all {dataset['rows']:,} training rows; no historical model parameter or feature map is copied into students.",
        f"- Use identical seed-{reg['seed']} initial tensors, shared {dataset['rows']:,}-row data, four equal-weight sampling groups, {reg['offline_iterations']:,} Adam updates, and the fixed final checkpoint for both arms. The fixed arm uses the same physical-state steering conversion at {reg['fixed_preview_mm']:g} mm.",
        f"- Retain all {collection['physical_branches']:,} physical branches, {collection['command_events']:,} command events and {collection['physics_ticks']:,} intervals. The {dataset['excluded_recovery_terminal_or_failed_rows']} excluded labels are successful terminal states; {dataset['recovery_active_rows']:,} eligible endpoint labels enter training.",
        '- Scope is at most two requested 20 ms decisions per branch; the common executor may append its actual terminal stop. Local branch validity is not a proof of future teacher rollout completion.','',
        '## Actual development executions','',
        '| Program | dt (ms) | Adaptive | Fixed 3 mm |','|---|---:|---:|---:|']
    def text(value):
        digits=4 if value['dt_s']==.0005 else 3
        return f"{value['complete_time_s']:.{digits}f} s" if value['success'] else f"Incomplete; {value['elapsed_prefix_s']:.{digits}f} s prefix"
    for run in reg['evaluation_runs']:
        pair=[next(r for r in results if r['arm']==arm and r['path']==run['case'] and r['dt_s']==run['dt_s']) for arm in reg['arms']]
        lines.append(f"| {run['case']} | {run['dt_s']*1000:g} | {text(pair[0])} | {text(pair[1])} |")
    lines.extend(['',f"Completed runs: adaptive {result['completed_runs']['adaptive']}/8; fixed {result['completed_runs']['fixed']}/8. Every incomplete prefix remains archived and contributes to these counts. No mean over only successful runs is used.",'',
        'The long original program was deliberately outside this bounded pilot. Its prior failure remains in the existing evidence; this pilot does not replace it or establish complete long-program reliability.','',
        '## Checks and costs','',
        f"Saved trajectories were independently reloaded and passed through the common metric and ordered-gate checks. Registered sources, data, endpoints, model hashes, initial tensors and the complete sampled-row stream were verified. Fixed-preview label conversion has maximum unclipped direction error {validation['maximum_unclipped_fixed_direction_error_rad']:.3g} rad; all {validation['fixed_steering_clipped']} clipped recovery labels remain recorded.",'',
        f"Shared collection: preparation {collection['preparation_wall_s']:.3f} s; collection {collection['collection_wall_s']:.3f} s wall / {collection['collection_cpu_s']:.3f} s CPU. Independent collection audit: {validation['audit_wall_s']:.3f} s wall. Feature/data preparation: {dataset['preparation_wall_s']:.3f} s wall.",'',
        '| Arm | Fit preparation wall (s) | Fitting wall (s) | Fitting CPU (s) | Evaluation wall (s) |','|---|---:|---:|---:|---:|'])
    for arm in reg['arms']:
        c=cost[arm];lines.append(f"| {arm} | {c['preparation_wall_s']:.3f} | {c['fitting_wall_s']:.3f} | {c['fitting_cpu_s']:.3f} | {c['total_evaluation_wall_s']:.3f} |")
    lines.extend(['',result['inherited_cost_scope'],'',
        'All costs are observed on a shared host. Actor and executor components are retained per run; they provide no hard-real-time guarantee.','',
        '## Evidence boundary and next decision','',
        'The pilot tests two diagnosed training defects under a fixed paired recipe. Training feature support and actual physical perturbations are established. Generalization, reliable long-horizon recovery, and a consistent adaptive-preview benefit require completed independent evidence and are not inferred from low action-fit error. Any next PPO stage or executor change requires a separate registration and must preserve these outcomes.','',
        f"The adaptive dense 0.5 ms failure occurs after {adaptive_dense_fine['ordered_gate_frontier']} ordered gates, with a maximum command deviation of {adaptive_dense_fine['max_polyline_deviation_mm']:.6f} mm against 1.25 mm. All {adaptive_dense_fine['feature_support']['joint']['observations']} executed observations have nonzero support in both rebuilt feature maps, including the final six. This remaining failure cannot be attributed to the old zero-support defect alone. All failed prefixes satisfy vector limits, but fail geometry and terminal completion.",'',
        '## Reproduction','',
        'Use `D:\\Anaconda\\envs\\PPO\\python.exe` from the v5 workspace. The registered stages are `src/recovery_training_v6.py register`, `collect`, `audit`, `prepare`, `fit --arm adaptive`, `fit --arm fixed`, `evaluate --arm adaptive`, and `evaluate --arm fixed`. Creation stages intentionally reject existing output directories; reproduce in a separate workspace with an empty `results/recovery_training_v6` and no copied v6 registration. Existing evidence can be checked in place with `src/audit_recovery_training_v6.py`.','',
        'Internal machine-readable result: `audit/recovery_training_v6_validation.json`. Full branch and fit/evaluation records: `results/recovery_training_v6`.'])
    (ROOT/'audit/recovery_training_v6.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','completed_runs','both_arms_complete_fixed_24000','identical_initial_trainable_tensors','identical_sampled_row_stream','audit_wall_s')}),flush=True)


if __name__=='__main__':main()
