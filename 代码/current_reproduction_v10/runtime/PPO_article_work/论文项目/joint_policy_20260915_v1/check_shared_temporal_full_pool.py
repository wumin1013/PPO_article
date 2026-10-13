"""Validate learned temporal parameters on the whole expanded training pool."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import argparse
import copy
import time
import numpy as np
import torch
from learned_action_pulse_policy import LearnedActionPulsePolicy,environment_factory
from analyze_nonlinear_joint_response import measure
from learn_full_pool_joint_credit import aggregate
from learn_retained_joint_growth import smooth_cruise
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}


def initialize(stage,source,baseline,deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,source=source,baseline=baseline,deadline=deadline)
    WORK['retained']={r['path']:r.get('baseline_shape') for r in read(source/'baseline.json')['retained']}
    WORK['cached']={a['mode']:{r['path']:r for r in a['best']['records']}
                    for a in read(source/'results.json')['arms'] if a['improved']}


def evaluate(mode):
    model_file=WORK['source']/mode/'candidate.pt'
    md=torch.load(model_file,map_location='cpu',weights_only=False)
    policy=LearnedActionPulsePolicy(md['parent'],md['coefficients'],md['mode'],md.get('expert_state'),md.get('feature_spec'))
    folder=WORK['stage']/mode;folder.mkdir()
    records=[];guards=[];trace_hashes=[];peaks=[]
    for b in WORK['baseline']['records']:
        cached=WORK['cached'][mode].get(b['path'])
        if cached:
            r=copy.deepcopy(cached);file=BASE/r['trace']
            with np.load(file) as z:rows=z['rows'].copy()
            r['reused_optimization_outcome']=True
        else:
            with np.load(b['trace']) as z:ref=z['reference'].copy()
            env=policy.bind(environment_factory(ref))
            chunks=[];actions=[];obs=[];executed=[]
            while not env.done:
                if time.perf_counter()>WORK['deadline']:raise TimeoutError('Complete training-pool wall budget')
                x=env.observation()
                _,_,_,info=env.step(policy.act(x,True)[0])
                chunks.append(info['rows']);actions.append(info['held_action']);obs.append(x)
                executed.append(info['executed_actions_1ms'])
            rows=np.vstack(chunks);r=measure(ref,rows,env)
            file=folder/f"path{b['path']}.npz"
            np.savez_compressed(file,reference=ref,rows=rows,actions=np.asarray(actions),
                               observations=np.asarray(obs),executed_actions_1ms=np.vstack(executed))
            r.update(path=b['path'],trace=file.relative_to(WORK['stage']).as_posix(),curved_path=b['curved_path'])
            if r['success']:r['smooth_cruise99_s']=smooth_cruise(rows)
        records.append(r);trace_hashes.append(dict(path=r['path'],sha256=sha(file)))
        peaks.append([np.linalg.norm(rows[:,i:i+2],axis=1).max() for i in (2,4,6)])
        good=r['success'] and r['geometry']['ordered_complete'] and r['geometry']['geometry_complete']
        if 'shape' in b:
            keys=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                  'full_normalized_energy','full_spatial_peak_curvature_mm_inv')
            bounds=dict(b['shape'])
            if b['path'] in md.get('reference_envelope_paths',[]):
                reference=WORK['retained'][b['path']]
                for key in keys:bounds[key]=max(bounds[key],reference[key])
                bounds['apex_speed_mm_s']=min(bounds['apex_speed_mm_s'],reference['apex_speed_mm_s'])
            if md.get('protect_current_lateral_energy_exit'):
                for key in ('out_edge_outward_mm','full_normalized_energy'):
                    bounds[key]=b['shape'][key]
                keys+=('reverse_lateral_total_mm',)
            good=good and r['target_motion']['first_turn_inside'] and all(
                r['shape'][k]<=bounds[k]+1e-7 for k in keys)
            good=good and r['shape']['apex_speed_mm_s']>=bounds['apex_speed_mm_s']-1e-4
        guards.append(dict(path=b['path'],pass_all=bool(good),
            comparison='research_reference_envelope' if b['path'] in md.get('reference_envelope_paths',[]) else 'current_parent'))
    summary=aggregate(records);s=WORK['baseline']['summary']
    acceptable=bool(all(g['pass_all'] for g in guards)
        and summary['complete_time_s']<=s['complete_time_s']+.02
        and summary['curved_cruise99_s']>=s['curved_cruise99_s']-.05
        and summary['reverse_sum_deg']<s['reverse_sum_deg']
        and summary['exit_sum_mm']<=s['exit_sum_mm'] and summary['energy_sum']<=s['energy_sum'])
    result=dict(tag=mode,records=records,summary=summary,per_path_guards=guards,
        original64_summary=aggregate([r for r in records if r['path']<64]),
        added_training_summary=aggregate([r for r in records if r['path']>=64]),
        research_acceptable=acceptable,adopted=False,model_sha256=sha(model_file),
        vector_peaks=np.max(peaks,axis=0).tolist(),raw_trace_sha256=trace_hashes)
    write_json(WORK['stage']/(mode+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-stage',default='canonical_gate_band_shared_temporal_joint_learning_20260919')
    parser.add_argument('--output-stage',default='canonical_gate_band_shared_temporal_full_pool_check_20260919')
    args=parser.parse_args()
    source=BASE/args.source_stage
    result=read(source/'results.json')
    arms=[a['mode'] for a in result['arms'] if a['improved']]
    assert arms,'No feasible improved shared temporal policy; do not rerun zero baselines'
    data=read(source/'baseline.json')
    stage=register(args.output_stage,
        [__file__,BASE/'learned_action_pulse_policy.py',BASE/'joint_runtime.py',
         source/'results.json',source/'receipt.json',source/'baseline.json']+
        [source/a/'candidate.pt' for a in arms],
        dict(wall_s=420,workers=len(arms),paths=len(data['baseline']['records']),arms=arms,
             reuse_exact_optimization_paths=True),
        'Check each feasible learned policy on the entire explicitly registered training pool. '
        'Reuse its EXACT already measured optimization paths; run only remaining paths. Subset success is insufficient. '
        'Store actual1ms actions,rawXY,V/A/J,coverage,braking,rest and all existing corner/complete '
        'motion metrics. Require the checkpoint-declared per-corner research guards '
        '(current-parent bounds except explicitly recorded unresolved-path reference envelopes) '
        'and aggregate time/cruise/shape '
        'gains before saving a research model. No unchanged zero policy is rerun. No final adoption, '
        'PPO,independent,fair-comparison or sealed-test claim.')
    write_json(stage/'baseline.json',data)
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        pool=mp.get_context('spawn').Pool(len(arms),initialize,(stage,source,data['baseline'],started+405))
        for c in pool.imap_unordered(evaluate,arms):
            outcomes.append(c)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False))
            print('SHARED_TEMPORAL_FULL_POOL',c['tag'],c['summary'],c['research_acceptable'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['research_acceptable']]
        selected=min(good,key=lambda c:c['summary']['reverse_sum_deg']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            md=torch.load(source/selected['tag']/'candidate.pt',map_location='cpu',weights_only=False)
            torch.save(md,stage/'research_candidate.pt');result['best']=selected['tag']
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_shared_temporal_full_pool_check',candidates=len(outcomes),
            best=selected['tag'] if selected else None,wall_s=time.perf_counter()-started,
            research_model_sha256=sha(stage/'research_candidate.pt') if selected else None,
            training_paths=len(data['baseline']['records']),adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_shared_temporal_full_pool_check',error=str(exc),
            completed_candidates=len(outcomes),wall_s=time.perf_counter()-started,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
