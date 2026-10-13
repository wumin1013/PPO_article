"""Learn compact20ms three-action credit blocks from complete feedback outcomes."""
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
from learn_complete_joint_teachers import initialize,optimize_path,teacher_score,parameters_in_bounds
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bounded-continuation',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    prior=BASE/'canonical_gate_band_spatial_shared_joint_credit_learning_20260920'
    source=BASE/'canonical_gate_band_lateral_aware_expert_weight_learning_20260920'
    parent_file=source/'width_1.2/candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)=='59e1bf592d2f2a5f519a46825f0d673a3bdd6978b492962f8289de00757c5771'
    assert sha(expert_file)=='86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
    data=read(prior/'baseline.json');paths=[206,304]
    old=BASE/'canonical_gate_band_causal_block_joint_teacher_learning_20260920'
    seeds={};selection_audit=[]
    if args.bounded_continuation:
        for t in read(old/'results.json')['teachers']:
            history=read(old/f"path{t['path']}/results.json")['candidates']
            valid=[c for c in history if parameters_in_bounds(c['parameters'],-.25,1.25)
                   and min(c.get('measured_inequalities',[0.]))>=-1e-8
                   and not np.any(np.asarray(c['parameters'])[12:])]
            best=copy.deepcopy(min(valid,key=lambda c:c['score']))
            best['trace']=str((old/best['trace']).resolve())
            violation=float(max(0.,np.max(np.asarray(t['best']['parameters'])-1.25),
                                np.max(-.25-np.asarray(t['best']['parameters']))))
            selection_audit.append(dict(path=t['path'],previous_best=t['best']['tag'],
                previous_score=t['best']['score'],maximum_box_violation=violation,
                valid_saved_exit_candidates=len(valid),reused_tag=best['tag'],reused_score=best['score']))
            seeds[t['path']]=best
        write_json(old/'selection_bounds_audit.json',dict(records=selection_audit,
            new_rollouts=0,previous_best_eligible=False,
            reason='Saved history selection omitted the declared coefficient box; physical outcomes remain measured.'))
        fit=BASE/'canonical_gate_band_causal_block_state_weight_learning_20260920'
        if fit.exists():write_json(fit/'selection_hold.json',dict(eligible=False,
            reason='Teachers fail their declared parameter-box selection; preserve actual results, do not select model.',
            source_audit=str(old/'selection_bounds_audit.json')))
    stage=register('canonical_gate_band_bounded_causal_block_joint_teacher_learning_20260920'
                   if args.bounded_continuation else 'canonical_gate_band_causal_block_joint_teacher_learning_20260920',
        [__file__,BASE/'learn_complete_joint_teachers.py',BASE/'localize_joint_shape_residual.py',
         BASE/'complete_corner_metrics.py',parent_file,expert_file,prior/'baseline.json',
         prior/'receipt.json',prior/'spatial_expert/results.json',source/'raw_result_audit_width_1.2.json']+
         ([old/'results.json',old/'receipt.json',old/'selection_bounds_audit.json',
           old/'path206/results.json',old/'path304/results.json'] if args.bounded_continuation else []),
        dict(wall_s=480 if args.bounded_continuation else 600,workers=2,paths=paths,
             maximum_new_candidates_per_path=158 if args.bounded_continuation else 237,
             coefficient_bounds=[-.25,1.25],selection_box_tolerance=1e-8,
             reused_exit_block=args.bounded_continuation,
             offline_free_weights=36,weights_per_block=12,maximum_evaluations_per_block=80,
             block_order=['exit_recovery','pre_apex','early_approach'],
             block_support='four distinct20ms decisions per block; zero at every other decision'),
        ('Correct history selection: COBYLA trial points may violate its coefficient box. '
         'Reuse the best already measured box-feasible exit-only teacher for each path. '
         'Do NOT repeat exit-block or zero physical rollouts. Continue only the previously '
         'invalidly anchored pre-apex and early blocks, enforcing both the declared box '
         'and all original measured trajectory guards at every incumbent selection. '
         'This repairs a selection defect; it does not relax physics or shape constraints. '
         if args.bounded_continuation else '')+
        'The shared18-parameter spatial policy can reduce large outgoing excursions but '
        'its measured candidates worsen original14 reversal or other energy/bend-speed '
        'guards. Do not repeat that unchanged search. Test temporal credit resolution '
        'on206 sharpness and304 outgoing crossing: replace globally supported cubic '
        'teacher coefficients with independent three-action expert weights at twelve '
        'actual20ms decisions. Optimize four exit-recovery decisions first, then four '
        'pre-apex decisions, then four early visible-approach decisions. Every block '
        'objective and constraint is the ACTUAL entire closed-feedback completion. '
        'Exit-only changes cannot alter the earlier apex, removing artificial cross-phase '
        'coupling of the former teacher basis. This increases total capacity36vs12 but '
        'keeps the per-block search12D and total evaluation cap240; it is not a unique '
        'causal attribution claim. Keep current lateral displacement,energy,exit and '
        'all original physics,coverage,braking,rest,time/cruise plus declared unresolved '
        'shape envelopes. All intervention schedules remain OFFLINE ONLY. Useful '
        'teachers must become observable-state weights and pass the complete88 pool; '
        'no scheduled block rule is deployed. Reuse zero full paths, verify next20ms '
        'identity, preserve every actual candidate. No PPO or final acceptance.')
    write_json(stage/'baseline.json',data)
    if args.bounded_continuation:write_json(stage/'selection_bounds_audit.json',dict(records=selection_audit,new_rollouts=0))
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        jobs=[]
        for p in paths:
            b=next(r for r in data['baseline']['records'] if r['path']==p)
            o=next(r for r in data['retained'] if r['path']==p)['baseline_shape']
            seed=dict(tag='reused_parent',path=p,parameters=[0.]*36,record=b,
                      score=teacher_score(b,b,o),trace=b['trace'])
            jobs.append((p,b,o,seeds[p],1) if args.bounded_continuation else (p,b,o,seed))
        pool=mp.get_context('spawn').Pool(2,initialize,
            (stage,parent,started+(465 if args.bounded_continuation else 585),'block_cobyla','causal_blocks',True,True,expert,True))
        for r in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(r)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,temporal_scope='causal_blocks'))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_causal_block_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(t['score_improved'] for t in outcomes),trials=sum(t['trials'] for t in outcomes),
            source_model_sha256=sha(parent_file),expert_model_sha256=sha(expert_file),
            wall_s=time.perf_counter()-started,offline_teacher_only=True,deployable_policy=False,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_causal_block_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,goal_complete=False))
        raise


if __name__=='__main__':mp.freeze_support();main()
