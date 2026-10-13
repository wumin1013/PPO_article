"""Deterministic final-function labels and finite observation neighborhoods.

Preparation only: no fitted weights are changed and no optimizer is created.
Abstract observation neighborhoods are not claimed to be physical trajectories.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from frozen_policy import FrozenPolicy
from strong_policy_v4 import load_strong_policy

OUT=ROOT/'results/final_function_teacher_v4/preparation_002'
MODEL=ROOT/'baseline_records/supplementary_reproduction/models/main.pt'
DATA=ROOT/'baseline_records/supplementary_reproduction/training/recorded_actions.npz'
FIT=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'
NOISE=np.array([.006,.003,.003,.008,.008,.006,.006,.003,.003,.003,.003,.003,.003,.006],dtype=np.float32)
NOISE_SEED=2026101101


def constrain_neighborhood(x):
    value=x.copy()
    value[:,6]=np.clip(value[:,6],-.995,.995)
    for indices in ([1,2],[3,4],[5,13]):
        magnitude=np.linalg.norm(value[:,indices],axis=1)
        value[:,indices]/=np.maximum(magnitude,1.)[:,None]
    for start,distance in [(7,2.),(9,6.),(11,15.)]:
        magnitude=np.linalg.norm(value[:,start:start+2],axis=1)
        maximum=(distance+1.25)/15.
        value[:,start:start+2]/=np.maximum(magnitude/maximum,1.)[:,None]
    return value


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=False)
    model_hash=sha(MODEL);data_hash=sha(DATA);fit_hash=sha(FIT)
    assert model_hash=='c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
    assert data_hash=='469c8d397d3a9d24b87d6fd399b0c78aa0dd7154dba9edcd8d4b9c5f68b51971'
    original_noise_source=ROOT/'runtime/PPO_article_joint_runs/natural_closed_resume/compact_dense_research/offline_feedback_fit.py'
    inventory=json.loads((ROOT/'audit/original_strong_recipe_inventory.json').read_text('utf-8'))
    reg=dict(created_utc=utc(),scope='DEVELOPMENT teacher preparation only; zero training and zero new trajectory execution',
        model_sha256=model_hash,dataset_sha256=data_hash,source_sha256=sha(__file__),noise_seed=NOISE_SEED,
        noise_std=NOISE.tolist(),noise_source=str(original_noise_source.relative_to(ROOT)),noise_source_sha256=sha(original_noise_source),
        augmentation='One seeded normal vector, clipped at4 componentwise standard deviations, per original row and its antithetic negative; retain all original6849 observations exactly; total20547.',
        range_controls='Neighbors only: cross-track remains an unconstrained local-reference coordinate; reference-sine[-.995,.995]; velocity/acceleration/far30 two-component norm<=1; forward2/6/15 coordinate norm<=(sensor+1.25)/15. Original observations remain unchanged.',
        consistency_scope='Numerical neighborhoods of the14-dimensional encoder, not asserted physically reachable states; independent perturbations can break geometric cross-coordinate identities.',
        labels='Frozen final c16d45 mean; save raw means and clip[-1,1] equivalent-action targets. No learned teacher copied into student initialization.',
        fixed_labels_available=False,fixed_reason='Fourteen abstract observations do not uniquely reconstruct ordered geometry and the guide at arbitrary learned preview; direct slicing would be invalid.',
        fixed_future_conversion='Restore full witnessed physical state on uniform executor, recompute observation, query final teacher, decode steering/feed/preview, compute beta_old+steering-beta_fixed3 then clip and record irreducible direction error; retain every clipping case.',
        failed_long_observations='Diagnostic only; never appended or specially weighted in teacher dataset.',
        known_teacher_upstream_wall_s=inventory['historical_seed923202_receipt_total_wall_s'],
        teacher_cost_qualification='Shared historical asset.7150.225s receipt sum excludes earlier teacher search and final historical matched fitting;19 upstream files unavailable; no claim of complete teacher-discovery cost.',
        no_training=True,no_new_final_data=True)
    write_json(OUT/'registration.json',reg)
    started,cpu=time.perf_counter(),time.process_time()
    with np.load(DATA) as d:
        base=d['observations'].copy();old=d['targets_three'].copy();case=d['case'].copy()
    rng=np.random.default_rng(NOISE_SEED)
    noise=np.clip(rng.standard_normal(base.shape),-4.,4.).astype(np.float32)*NOISE
    plus=constrain_neighborhood(base+noise);minus=constrain_neighborhood(base-noise)
    x=np.vstack([base,plus,minus])
    teacher=FrozenPolicy(MODEL)
    labels=[];batch_calls=0
    inference_start,inference_cpu=time.perf_counter(),time.process_time()
    for begin in range(0,len(x),512):
        labels.append(teacher.mean(x[begin:begin+512]).numpy());batch_calls+=1
    raw=np.vstack(labels);targets=np.clip(raw,-1.,1.)
    inference_wall=time.perf_counter()-inference_start;inference_cpu_s=time.process_time()-inference_cpu
    assert x.shape==(20547,14) and targets.shape==(20547,3) and np.isfinite(raw).all()
    output=OUT/'adaptive_teacher_records.npz'
    np.savez_compressed(output,observations=x,targets_three=targets,targets_three_raw=raw,
        case=np.tile(case,3),source_row=np.tile(np.arange(len(base)),3),
        neighborhood_sign=np.repeat(np.array([0,1,-1],dtype=np.int8),len(base)),
        applied_observation_delta=x-np.tile(base,(3,1)),requested_positive_noise=noise)
    student,_=load_strong_policy(FIT)
    with torch.no_grad():
        predictions=torch.cat([student.mean(p) for p in np.array_split(x,41)]).numpy()
    def error(a,b):
        delta=np.clip(a,-1,1)-np.clip(b,-1,1)
        return dict(rmse=np.sqrt(np.mean(delta**2,axis=0)).tolist(),maximum=np.max(abs(delta),axis=0).tolist())
    failure=json.loads((ROOT/'audit/pilot_long_failure_v4.json').read_text('utf-8'))
    failure_x=np.array(failure['last_ten_decisions']['observations'],dtype=np.float32)
    with torch.no_grad():
        failure_y=teacher.mean(failure_x).numpy();failure_pred=student.mean(failure_x).numpy()
    closest=[]
    for point in failure_x:
        distance=np.linalg.norm((base-point)/NOISE,axis=1)
        index=int(distance.argmin());closest.append(dict(source_row=index,noise_scaled_l2=float(distance[index])))
    report=dict(status='prepared_no_training',registration_sha256=sha(OUT/'registration.json'),
        output_file=str(output.relative_to(ROOT)),output_sha256=sha(output),samples=len(x),source_samples=len(base),
        teacher_batch_calls=batch_calls,teacher_observation_evaluations=len(x),teacher_inference_wall_s=inference_wall,
        teacher_inference_cpu_s=inference_cpu_s,total_preparation_wall_s=time.perf_counter()-started,total_preparation_cpu_s=time.process_time()-cpu,
        old_labels_vs_final_function=error(old,targets[:len(base)]),student_vs_final_function_on_original=error(predictions[:len(base)],targets[:len(base)]),
        student_vs_final_function_on_neighbors=error(predictions[len(base):],targets[len(base):]),
        original_rows_unchanged=bool(np.array_equal(x[:len(base)],base)),
        perturbation_max_abs_by_feature=abs(x[len(base):]-np.tile(base,(2,1))).max(axis=0).tolist(),
        range_control_changed_plus_rows=int(np.any(plus!=base+noise,axis=1).sum()),
        range_control_changed_minus_rows=int(np.any(minus!=base-noise,axis=1).sum()),
        failed_long_diagnostic=dict(rows=len(failure_x),student_vs_final_function=error(failure_pred,failure_y),nearest_original_row=closest,added_to_training=False),
        model_unchanged=sha(MODEL)==model_hash,old_dataset_unchanged=sha(DATA)==data_hash,fit_model_unchanged=sha(FIT)==fit_hash,
        source_unchanged=sha(__file__)==reg['source_sha256'],fixed_labels_prepared=False,
        replaces_rejected_preparation_001='Removed invalid local-reference cross-track clamp; rejection and all original files retained',
        conclusion='Fresh fit approximation error dominates label-version difference; final-function relabeling alone does not guarantee improved rollout. Neighbor supervision and optimization budget require a separately registered training pilot.')
    write_json(OUT/'receipt.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
