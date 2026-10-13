"""Target current bend-state coverage with learned joint compact features."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from learn_training_pool_constraint_repair import build_extension
from learn_pathwise_full_credit import pathwise_response, actual_selection
from learn_full_pool_compensation import solve_response, progress
from learn_full_pool_joint_credit import initialize, evaluate, candidate_metadata
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def bend_support(baseline):
    selected=[];audit=[];support=[]
    for r in baseline['records']:
        with np.load(r['trace']) as z:rows,obs=z['rows'].copy(),z['observations'].copy()
        support.append(obs)
        if r['path'] not in (8,12,13,14,15):continue
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        angular=np.abs(rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4])/np.maximum(speed,1e-9)**2
        ids=np.flatnonzero((angular>.5)&(speed>5.)&(rows[:,12]==0))//20
        # Sampling support, not a runtime condition or requested trajectory.
        ids=np.unique(np.clip(np.r_[ids-2,ids-1,ids,ids+1,ids+2],0,len(obs)-1))
        selected.append(obs[ids]);audit.append(dict(path=r['path'],decisions=ids.tolist()))
    return np.vstack(selected),np.vstack(support),audit


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_full_pool_joint_credit_learning_20260919'
    parent=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    assert sha(parent)=='5ab47111bfcec3a511903e20f1249c91bbfb65e5ab4cd1a3d2b4a493a979c26a'
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    metadata=torch.load(parent,map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_current_bend_capacity_learning_20260919',
        [__file__,BASE/'learn_training_pool_constraint_repair.py',BASE/'local_suffix_credit_policy.py',
         BASE/'learn_pathwise_full_credit.py',BASE/'learn_full_pool_compensation.py',
         BASE/'learn_full_pool_joint_credit.py',source/'baseline.json',parent,
         BASE/'canonical_gate_band_current_zero_identity_check_20260919/receipt.json',
         BASE/'canonical_gate_band_full_pool_step_refinement_20260919/current_shape_timing.json'],
        dict(wall_s=900,workers=4,added_centers=6,added_parameters=18,
             response_candidates=36,baseline_identity_candidate=1,joint_candidates=2,
             maximum_head_latent_per_axis=[.015,.10,.10],normalized_joint_trust=2.),
        'The13-direction constrained update remains locally limited; tiny steps do not remove '
        'all peak regressions. Exact zero-update replay rules out mismatched baseline traces. '
        'Current155deg guide/steering composition jumps53.36deg at301ms before344ms curvature '
        'peak. Extend current compact continuous-state actor by6 state-space coverage centers '
        'from actual high-turn-rate neighborhoods on paths8/12/13/14/15 including two neighboring '
        'decisions for anticipatory support. State-space distances choose centers and radii; '
        'features are exactly zero on all current straight and first-entry protected observations. '
        'No path,angle,time label or desired action appears in the deployed policy. Learn18 '
        'steering/feed/preview coefficients from actual +/- complete64-policy responses; no '
        'feed labels, temporal filtering or altered executor. Verify exact zero-extension actions '
        'on all4119 saved observations, and physically check zero-extension64 before learning. '
        'Use per-route complete consequences, freeze missing-both-sides directions, and constrain '
        'per-corner and aggregate shape,all margins,time,cruise and saved-state function trust. '
        'At most half/full new learned actors. This is a bounded capacity-and-credit experiment '
        'with real joint parameter optimization, not PPO or causal proof of unique root cause. '
        'If no guarded measured progress, change mechanism rather than repeat this batch.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'从当前弯曲状态扩充六个连续特征，采集并学习三动作完整64路径联合响应。')
    try:
        current,support,selection=bend_support(baseline)
        spec,_,_,audit=build_extension(metadata,current,baseline['records'])
        policy=load_credit_policy(spec);original=load_credit_policy(metadata)
        identity=np.array([policy.act(o,True)[0]-original.act(o,True)[0] for o in support])
        audit.update(selected_bend_states=selection,zero_extension_actions_bitwise=bool(np.all(identity==0.)),
                     maximum_saved_state_identity_error=float(abs(identity).max()))
        assert audit['zero_extension_actions_bitwise'],audit
        features=policy.features(torch.tensor(support)).detach().numpy()*policy.output_scale.numpy()
        new=features[:,-6:,:]
        scale=np.minimum(np.array([.015,.10,.10])/np.maximum(abs(new).max(0),1e-12),100.)
        directions=np.zeros((18,len(spec['centers']),3))
        for c in range(6):
            for h in range(3):directions[3*c+h,-6+c,h]=scale[c,h]
        trust=(np.einsum('sch,ich->shi',features,directions)/np.array([.02,.15,.15])[None,:,None]).reshape(-1,18)
        np.savez_compressed(stage/'parameter_space.npz',directions=directions,trust=trust)
        torch.save(spec,stage/'zero_extension.pt');write_json(stage/'direction_audit.json',audit)
        write_json(stage/'baseline.json',data)
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,spec,directions,refs,curved,retained,started+890))
        zero=pool.apply(evaluate,(('zero_extension',np.zeros(18)),))
        identities=[]
        for a,b in zip(zero['records'],baseline['records']):
            with np.load(stage/a['trace']) as x,np.load(b['trace']) as y:
                equal=bool(np.array_equal(x['rows'],y['rows']) and np.array_equal(x['actions'],y['actions']))
            identities.append(dict(path=a['path'],bitwise=equal))
        write_json(stage/'zero_physical_identity.json',identities)
        assert all(r['bitwise'] for r in identities),'Zero extension must reproduce full64 physically'
        print('BEND_ZERO_FULL64_IDENTITY',True,flush=True)
        jobs=[(f'axis{i}_{s:+d}',s*np.eye(18)[i]) for i in range(18) for s in (-1,1)]
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('BEND_CAPACITY_AXIS',len(outcomes),c['tag'],c['summary']['successes'],c['summary']['reverse_sum_deg'],flush=True)
        indexed={c['tag']:c for c in outcomes}
        response_data,eligibility=pathwise_response(stage,baseline,retained,[(indexed[f'axis{i}_-1'],indexed[f'axis{i}_+1']) for i in range(18)])
        write_json(stage/'response_eligibility.json',eligibility)
        step,report=solve_response(baseline,retained,response_data,trust,True,min(started+790,time.perf_counter()+120),aggregate_guards=True)
        write_json(stage/'optimizer.json',report)
        print('BEND_CAPACITY_SOLVER',report['message'],report['predicted_objective'],report['maximum_function_change'],flush=True)
        jobs=[(f'bend_joint_{s}',s*step) for s in (.5,1.)] if abs(step).max()>1e-6 else []
        for tag,u in jobs:torch.save(candidate_metadata(spec,directions,u),stage/(tag+'.pt'))
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('BEND_CAPACITY_JOINT',c['tag'],c['summary'],flush=True)
        pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag']
            torch.save(candidate_metadata(spec,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_current_bend_capacity_learning',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes)+1,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_current_bend_capacity_learning',error=str(exc),full64_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据弯曲状态容量实验的实际完整64路径结果继续联合学习；正式保留模型及最终验收不变。')


if __name__=='__main__':mp.freeze_support();main()
