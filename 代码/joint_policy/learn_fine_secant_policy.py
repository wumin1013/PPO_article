"""Validate a measured fine-step secant before further guarded policy learning."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_full_pool_secant_guard import calibrate
from learn_fine_full_policy_credit import validate
from learn_pathwise_full_credit import actual_selection
from learn_full_pool_compensation import solve_response,progress
from learn_full_pool_joint_credit import initialize,evaluate,candidate_metadata
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_fine_full_policy_credit_learning_20260919'
    space=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    data=torch.load(source/'response_model.pt',map_location='cpu',weights_only=False)
    original,baseline,retained=data['response'],data['baseline'],data['retained']
    candidates={c['tag']:c for c in read(source/'results.json')['candidates']}
    half,full=candidates['fine_policy_joint_0.5'],candidates['fine_policy_joint_1.0']
    metadata=torch.load(space/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(space/'parameter_space.npz') as z:directions,trust=z['directions'].copy(),z['trust'].copy()
    stage=register('canonical_gate_band_fine_secant_policy_learning_20260919',
        [__file__,BASE/'learn_full_pool_secant_guard.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_pathwise_full_credit.py',source/'response_model.pt',source/'results.json',source/'receipt.json',
         space/'zero_extension.pt',space/'parameter_space.npz'],
        dict(wall_s=180,workers=2,fit_actors=1,held_actors=1,new_full64_candidates=2,optimizer_s=90),
        'Quarter-amplitude current full-policy responses improved prediction38percent; both '
        'new joint actors completed64/64 and reduced all aggregate shape measures, but small '
        'per-corner peak/apex/exit violations remain. Calibrate the fine response on the measured '
        'HALF actor only; check the FULL actor without fitting it. Accept calibration only for '
        '20percent lower shape error and nonworse155deg peak prediction. If accepted, refit both '
        'fine actors and learn half/full guarded coefficient updates. Retain all physical and '
        'actual acceptance thresholds, geometry,time,cruise and full-route evidence; no new '
        'response collection, deployment rule, PPO claim or final promotion.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'用小步完整轨迹校准联合信用并先做未参与拟合的全步验证，再学习受约束更新。')
    try:
        model,fit=calibrate(original,baseline,retained,trust,[(source,half)])
        before=validate(original,baseline,retained,[full])[0];after=validate(model,baseline,retained,[full])[0]
        accepted=bool(after['shape_error']<.8*before['shape_error'] and abs(after['peak14_error'])<=abs(before['peak14_error']))
        write_json(stage/'validation.json',dict(fit=fit,before=before,after=after,accepted=accepted,fit_to_full=False))
        print('FINE_SECANT_VALIDATION',before,after,accepted,flush=True)
        jobs=[]
        if accepted:
            model,fit=calibrate(original,baseline,retained,trust,[(source,half),(source,full)])
            write_json(stage/'fit.json',fit)
            torch.save(dict(response=model,baseline=baseline,retained=retained),stage/'response_model.pt')
            step,report=solve_response(baseline,retained,model,trust,True,started+100,aggregate_guards=True)
            write_json(stage/'optimizer.json',report)
            print('FINE_SECANT_SOLVER',report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
            if abs(step).max()>1e-6:jobs=[(f'fine_secant_joint_{s}',s*step) for s in (.5,1.)]
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        for tag,u in jobs:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        if jobs:
            pool=mp.get_context('spawn').Pool(2,initialize,(stage,metadata,directions,refs,curved,retained,started+170))
            for c in pool.imap_unordered(evaluate,jobs):
                outcomes.append(c);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('FINE_SECANT_ACTUAL',c['tag'],c['summary'],flush=True)
            pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag'];torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_fine_secant_policy_learning',best=selected['tag'] if selected else None,
            response_validated=accepted,full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_fine_secant_policy_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'继续根据实际逐角形状与完整路径收益修正联合学习，正式模型和最终验收边界不变。')


if __name__=='__main__':mp.freeze_support();main()
