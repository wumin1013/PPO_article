"""Bounded physical line refinement of an already measured joint update."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_pathwise_full_credit import actual_selection
from learn_full_pool_compensation import progress
from learn_full_pool_joint_credit import initialize,evaluate,candidate_metadata
from learn_prefix_preserving_suffix_credit import prefix_indices
from audit_current_command_coordinates import command_signals
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def audit_shape_trace(record):
    with np.load(record['trace']) as z:ref,rows,actions,obs=[z[k].copy() for k in ('reference','rows','actions','observations')]
    signals=command_signals(ref,rows,actions)
    protected=prefix_indices(ref,rows,obs)
    ordinary=rows[:,12]==0
    delta=np.diff(signals['combined_deg']);delta[~(ordinary[:-1]&ordinary[1:])]=0.
    tick=int(np.argmax(abs(delta)))+1
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    cross=rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4]
    kappa=np.abs(cross)/np.maximum(speed,1e-9)**3
    kappa[speed<=5.]=0.
    peak=int(kappa.argmax())
    def window(center):
        result=[]
        for i in range(max(0,center//20-2),min(len(actions),center//20+3)):
            t=i*20
            result.append(dict(decision=i,start_ms=t,action=actions[i].tolist(),
                speed=float(speed[t]),physical_heading=float(signals['physical_heading_deg'][t]),
                guide_heading=float(signals['los_deg'][t]),combined_heading=float(signals['combined_deg'][t])))
        return result
    return dict(path=record['path'],trace_sha256=sha(record['trace']),protected_through_ms=20*len(protected),
        target_jump_ms=tick+1,target_jump_deg=float(delta[tick-1]),physical_peak_ms=peak+1,
        physical_peak_curvature=float(kappa[peak]),target_jump_window=window(tick),peak_window=window(peak))


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_full_pool_secant_guard_learning_20260919'
    space=BASE/'canonical_gate_band_active_constraint_joint_learning_20260919'
    parent=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    data=read(space/'baseline.json');baseline,retained=data['baseline'],data['retained']
    metadata=torch.load(parent,map_location='cpu',weights_only=False)
    with np.load(space/'parameter_space.npz') as z:directions=z['directions'].copy()
    step=np.array(read(source/'raw_optimizer.json')['parameters'])
    stage=register('canonical_gate_band_full_pool_step_refinement_20260919',
        [__file__,BASE/'learn_full_pool_joint_credit.py',BASE/'learn_pathwise_full_credit.py',
         source/'results.json',source/'receipt.json',source/'raw_optimizer.json',space/'parameter_space.npz',space/'baseline.json',parent],
        dict(wall_s=120,workers=2,new_full64_candidates=2,scales=[.125,.25],diagnostic_new_rollouts=0),
        'The guarded half-step completed64/64, preserved time/cruise and reduced all3 aggregate '
        'shape metrics, but path8 peak exceeded its per-corner envelope by2.94e-6/mm. No threshold '
        'is relaxed. Test only quarter/eighth steps of the SAME learned joint direction, reusing '
        'all prior measurements. Keep all original physical and actual selection requirements. '
        'Also locate real155deg target jumps and peaks in the current saved trace, to decide '
        'whether the next mechanism change should address current temporal action coupling. '
        'No extra physics for that audit, no runtime interpolation or switching, no PPO claim.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'仅验证已学联合方向的四分之一步和八分之一步，同时定位155度真实目标跳变。')
    try:
        write_json(stage/'current_shape_timing.json',[audit_shape_trace(r) for r in baseline['records'] if r['path'] in (8,12,14)])
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(2,initialize,(stage,metadata,directions,refs,curved,retained,started+115))
        jobs=[(f'joint_step_{s}',s*step) for s in (.125,.25)]
        for tag,u in jobs:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        for c in pool.imap_unordered(evaluate,jobs):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('REFINEMENT_ACTUAL',c['tag'],c['summary'],flush=True)
        pool.close();pool.join();pool=None
        selected=actual_selection(outcomes,baseline,retained)
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag']
            torch.save(candidate_metadata(metadata,directions,selected['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_full_pool_step_refinement',best=selected['tag'] if selected else None,
            full64_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_full_pool_step_refinement',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据新检查点与155度跳变时序继续闭环三动作学习，保持所有验收条件。')


if __name__=='__main__':mp.freeze_support();main()
