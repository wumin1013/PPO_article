"""Matched learned-metric controls for action-induced loss of later feedback support."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from joint_teacher_state_policy import features,INDICES
from consolidate_joint_teacher_policy import collect_data
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions,WORK
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


def metric_weights(md,gamma):
    weights=np.ones((len(md['centers']),md['centers'].shape[1]))
    stable=set([11,12,15]+list(range(22,72)))
    dynamic=[i for i,k in enumerate(INDICES) if k not in stable]+list(range(len(INDICES),weights.shape[1]))
    weights[7,dynamic]=gamma
    return weights


def evaluate_metric(job):
    path,u,tag,gamma=job
    policy=WORK['policy'];policy.metadata['per_center_distance_weights']=metric_weights(policy.metadata,gamma)
    return worker_evaluate((path,u,tag))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--complete-preparation',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_bounded_midturn_joint_learning_20260920'
    parent_file=BASE/'canonical_gate_band_decoded_joint_step_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    old=read(source/'state_joint/results.json')['candidates'];data=read(source/'baseline.json')
    baseline=data['baseline']['records'];by_path={r['path']:r for r in baseline}
    u0=np.asarray(old[0]['coefficients']);delta=np.asarray(read(source/'proposal.json')['coefficient_delta'])
    failed_preparation=BASE/'canonical_gate_band_motion_robust_midturn_joint_learning_20260920/receipt.json'
    if args.complete_preparation:
        assert read(failed_preparation)['status']=='failed_motion_metric_joint'
        assert not (failed_preparation.parent/'state_joint/results.json').exists()
    stage=register('canonical_gate_band_motion_robust_midturn_completion_20260920' if args.complete_preparation else
                   'canonical_gate_band_motion_robust_midturn_joint_learning_20260920',
        [__file__,parent_file,source/'receipt.json',source/'state_joint/candidate.pt',source/'state_joint/results.json',
         source/'actual_support_audit.json',source/'proposal.json',BASE/'joint_teacher_state_policy.py',
         BASE/'observable_memory_joint_policy.py',BASE/'consolidate_joint_teacher_policy.py',
         BASE/'shape_event_credit.py',BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py']+
         ([failed_preparation] if args.complete_preparation else []),
        dict(wall_s=180,workers=4,new_metric_weights=[.1,0.],joint_parameter_fractions=[.25,1.],
             maximum_new_vectors=4,coefficient_bounds=[-1.,1.],all_current_training_paths=88),
        ('Complete the same registered metric comparison after a JSON int64 '
         'serialization error during preparation. No new physical runs were '
         'started there; retain its failed receipt and rerun only saved-state '
         'preparation. Cast the recorded path ID to native int. ' if args.complete_preparation else '')+
        'Actual bounded early/midturn joint controls attenuate the later320ms '
        'kernel from0.5 to0.291,0.067,and0 as the preceding action changes '
        'the observed state. All three outgoing excursions worsen despite '
        'predicted improvement. Match TWO diagonal-distance metric variants '
        'on the later kernel only: motion/filtered-heading/old-action/innovation '
        'weights0.1 and0, retaining all local reference-geometry coordinates. '
        'Compare identical learned three-action coefficient steps at quarter '
        'and full amplitude against the saved ordinary-metric controls. No '
        'time/path/angle input, nominal-trajectory clamp, threshold relaxation '
        'or new action interpolation. Determine all affected current training '
        'paths from exact replay plus unchanged81-state reuse; execute every '
        'affected complete route. This is joint parameter/representation '
        'selection using full returns, not PPO or generalization proof. If '
        'a metric restores support but still fails shape guards, learn new '
        'joint weights using that actual response instead of adopting this '
        'diagnostic extrapolation or attributing everything to support loss.')
    started=time.perf_counter();deadline=started+165;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);pool=None;history=[];blocks=[]
    try:
        replay=collect_data(md,md['parent'],[r for r in baseline if r['path'] in [14,104,105,204,205,304,305]],deadline=deadline)
        with np.load(BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920/replayed_states.npz') as z:
            earlier={k:z[k].copy() for k in ('observations','innovations','path_ids','decisions')}
        keep=~np.isin(earlier['path_ids'],[14,104,105,204,205,304,305])
        replay={k:np.concatenate([earlier[k][keep],replay[k]]) for k in earlier}
        plain=features(md,replay['observations'],replay['innovations'])
        md['per_center_distance_weights']=metric_weights(md,1.)
        assert np.array_equal(plain,features(md,replay['observations'],replay['innovations']))
        support=[]
        for gamma in (.1,0.):
            md['per_center_distance_weights']=metric_weights(md,gamma)
            f=features(md,replay['observations'],replay['innovations']);active=np.any(f[:,7]!=0.,axis=1)
            paths=sorted(set([104]+np.unique(replay['path_ids'][active]).tolist()))
            row=dict(gamma=gamma,paths=paths,decisions=int(active.sum()),all_corners=all('shape' in by_path[p] for p in paths),
                activations=[dict(path=int(p),decision_ms=int(d)*20,value=float(abs(a).max()))
                    for p,d,a in zip(replay['path_ids'][active],replay['decisions'][active],f[active,7])])
            support.append(row)
            print('MOTION_METRIC_SUPPORT',gamma,paths,row['decisions'],row['all_corners'],flush=True)
        eligible=[r for r in support if r['all_corners']]
        paths=sorted(set(p for r in eligible for p in r['paths']))
        assert paths and 104 in paths
        selected=[by_path[p] for p in paths];choice=dict(paths=paths,variable_kernels=[3,7])
        write_json(stage/'support_audit.json',dict(selected=choice,variants=support,
            old_ones_metric_features_bitwise=True,current_training_states=len(replay['path_ids']),
            reweighted_observation_indices=list(range(11)),reweighted_innovation_indices=[0,1,2],
            unchanged_reference_observation_indices=[11,12,15]+list(range(22,72)),
            other81_states_reused_from_bitwise_unchanged_routes=True))
        torch.save(md,stage/'initial_policy.pt')
        _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
        retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
        def entry(records,u,tag,gamma):
            original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
            extra=intrinsic_guards(records,selected)
            return dict(tag=tag,coefficients=np.asarray(u).reshape(-1).tolist(),records=records,gamma=gamma,
                score=float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained))),
                measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
                augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
                actual_rerun_paths=paths,exact_zero_support_paths=[])
        for c in old:
            changed={r['path']:r for r in c['records']}
            records=[copy.deepcopy(changed.get(p,by_path[p])) for p in paths]
            history.append(entry(records,c['coefficients'],'reused_'+c['tag'],1.))
        zero=history[0];md['reused_candidates']=len(history)
        def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
        def best():return min(filter(feasible,history),key=lambda c:c['score'])
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for variant in eligible:
            gamma=variant['gamma']
            for fraction in (.25,1.):
                u=u0+fraction*delta;tag=f'metric_{gamma:g}_joint_{fraction:g}'
                records=pool.map(evaluate_metric,[(p,u,tag,gamma) for p in paths])
                c=entry(records,u,tag,gamma);history.append(c)
                blocks.append(dict(gamma=gamma,fraction=fraction,tag=tag,feasible=feasible(c)))
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                target=next(r for r in records if r['path']==104)
                print('MOTION_METRIC_JOINT',gamma,fraction,c['score'],min(c['augmented_measured_inequalities']),
                    target['shape']['out_edge_outward_mm'],flush=True)
        pool.close();pool.join();pool=None
        md['per_center_distance_weights']=metric_weights(md,best()['gamma'])
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        if history:
            md['per_center_distance_weights']=metric_weights(md,best()['gamma'])
            finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
        else:receipt(stage,dict(status='budget_stopped_motion_metric_preparation',wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_motion_metric_joint',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
