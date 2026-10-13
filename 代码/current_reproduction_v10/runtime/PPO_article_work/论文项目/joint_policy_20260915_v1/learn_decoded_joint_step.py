"""Resolve one measured zero-action-update plateau with bounded parameter steps."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_preapex_joint_compensation_20260920'
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    old=read(source/'state_joint/results.json')['candidates'];index={c['tag']:c for c in old}
    zero=copy.deepcopy(old[0]);anchor=copy.deepcopy(index['preapex_joint_39']);history=[zero,anchor]
    step=np.asarray(index['preapex_joint_40']['coefficients'])-anchor['coefficients']
    assert np.any(step)
    plateau=[]
    for tag in ('preapex_joint_40','preapex_joint_41','preapex_joint_42','preapex_joint_43'):
        for r,b in zip(index[tag]['records'],anchor['records']):
            with np.load(r['trace']) as z, np.load(b['trace']) as before:
                assert np.array_equal(z['rows'],before['rows']) and np.array_equal(z['actions'],before['actions'])
            plateau.append(dict(tag=tag,path=r['path'],rows_and_actions_bitwise=True))
    data=read(source/'baseline.json');baseline=data['baseline']['records'];selected=zero['records']
    paths=[r['path'] for r in selected];support=read(source/'support_audit.json');choice=support['selected']
    stage=register('canonical_gate_band_decoded_joint_step_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'shape_event_credit.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         source/'receipt.json',source/'state_joint/results.json',source/'state_joint/candidate.pt',
         source/'optimization_blocks.json',parent_file],
        dict(wall_s=90,workers=4,maximum_vectors=3,parameter_step_factors=[4.,16.,64.]),
        'Preapex compensation leaves only path14 apex speed margin-1.1732e-8; '
        'acceptance remains-1e-8. Four actual joint repairs have bitwise identical '
        'complete rows and decoded actions on all seven affected paths. This is '
        'direct evidence of a zero-action-update plateau for THESE parameter '
        'points only, not a cause of other failures. Reuse the first learned '
        'restoration direction from39 to40 and test parameter increments4x,16x,64x '
        'until one passes every original/intrinsic guard. Keep float32 outputs, '
        'unchanged20ms holds and1ms physics, and bitwise0:300ms prefix. These are '
        'learned-parameter line steps, not action interpolation or a speed rule. '
        'At most3 new complete vectors/90s; no acceptance tolerance change.')
    started=time.perf_counter();deadline=started+75;folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'decoded_plateau_audit.json',dict(records=plateau,new_physical_runs=0))
    md['reused_candidates']=2;blocks=[];pool=None
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for factor in (4.,16.,64.):
            u=np.asarray(anchor['coefficients'])+factor*step;tag=f'decoded_joint_{int(factor)}'
            assert parameters_in_bounds(u,-1.,1.)
            records=pool.map(worker_evaluate,[(p,u,tag) for p in paths]);identity=[]
            for r,b in zip(records,anchor['records']):
                with np.load(r['trace']) as z,np.load(b['trace']) as before:
                    assert np.array_equal(z['rows'][:300],before['rows'][:300])
                    identity.append(dict(path=r['path'],rows_bitwise=np.array_equal(z['rows'],before['rows']),
                        actions_bitwise=np.array_equal(z['actions'],before['actions'])))
            original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
            extra=intrinsic_guards(records,selected)
            c=dict(tag=tag,coefficients=u.tolist(),records=records,
                score=float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained))),
                measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
                augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
                actual_rerun_paths=paths,exact_zero_support_paths=[],early_prefix_bitwise=True)
            history.append(c);blocks.append(dict(factor=factor,tag=tag,identity=identity,feasible=feasible(c)))
            write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
            print('DECODED_JOINT',factor,c['score'],min(c['augmented_measured_inequalities']),feasible(c),flush=True)
            if feasible(c):break
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_decoded_joint_step',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=len(history)-2,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
