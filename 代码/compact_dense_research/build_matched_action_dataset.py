"""Shared witnessed inputs with initial-direction-equivalent action labels."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import math
import shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,write,sha,verify_bound_inputs
from replay_lowdim_training_stage import SUPPLEMENT,SEEDS
from compact_farvector_runtime import FarVectorRuntime
from reconstruct_farvector_observations import set_physical
from probe_joint_target_coordinates import guide_angle
from joint_runtime import wrap


def main():
    verify_bound_inputs()
    stage=SUPPLEMENT/'matched_direction_initialization'
    stage.mkdir(exist_ok=True)
    assert not (stage/'protocol.json').exists(), 'Do not overwrite a registered corpus'
    if (stage/'supervision.npz').exists():
        recovery=stage/'incomplete_preparation_history'
        recovery.mkdir(exist_ok=False)
        shutil.copy2(stage/'supervision.npz',recovery/'supervision.npz')
    source=ROOT/'direct14_dictionary02_resume_round73'
    observations=[];targets3=[];targets2=[];cases=[];records=[];bindings={}
    for case in ['original','dense']:
        for key,dt in [('1ms',.001),('half_ms',.0005)]:
            path=source/f'eval_0000_{case}_{key}.npz'
            result_path=path.with_suffix('.json')
            gates=path.with_name(path.stem+'_gates.json')
            result=json.loads(result_path.read_text(encoding='utf-8'))
            assert result['success'] and result['trace_sha256']==sha(path)
            for p in [path,result_path,gates]:bindings[str(p)]=sha(p)
            with np.load(path) as d:
                ref,rows,x,actions,latents=[d[k].copy() for k in ['reference','rows','observations','actions','latents']]
                times=d['decision_times_s'].copy()
            events=json.loads(gates.read_text(encoding='utf-8'))
            crossing=np.array([v['command_segment'] for v in events])
            ids=np.arange(len(x))
            if case=='original':
                turning=(abs(x[:,6])>.015)|(abs(x[:,13])>.003)|(np.linalg.norm(x[:,1:3],axis=1)<.995)
                ids=np.unique(np.r_[np.flatnonzero(turning),np.arange(0,len(x),32),np.arange(100),np.arange(len(x)-100,len(x))])
            env=FarVectorRuntime(ref)
            ys=[];errors=[];direction_errors=[];clipped=0
            for i in ids:
                tick=int(round(times[i]/dt))
                if tick:set_physical(env.env,rows[tick-1])
                env.env.tracker.frontier=int(np.searchsorted(crossing,tick,side='left'))
                errors.append(float(abs(env.observation()-x[i]).max()))
                held=actions[i].copy()
                desired=wrap(guide_angle(env.env,held)+held[0])
                held[2]=3.
                steer=wrap(desired-guide_angle(env.env,held))
                clipped+=int(abs(steer)>math.pi/2)
                held[0]=np.clip(steer,-math.pi/2,math.pi/2)
                direction_errors.append(abs(wrap(guide_angle(env.env,held)+held[0]-desired)))
                ys.append([held[0]/(math.pi/2),held[1]/50.-1.])
            assert max(errors)<2e-6,(case,key,max(errors))
            observations.append(x[ids]);targets3.append(np.clip(latents[ids],-1,1));targets2.append(np.array(ys))
            cases.extend([case]*len(ids))
            records.append(dict(case=case,dt_s=dt,samples=len(ids),maximum_observation_reconstruction_error=max(errors),
                steering_targets_clipped=clipped,maximum_initial_direction_error_rad=max(direction_errors)))
    np.savez_compressed(stage/'supervision.npz',observations=np.vstack(observations).astype(np.float32),
        targets_three=np.vstack(targets3).astype(np.float32),targets_two=np.vstack(targets2).astype(np.float32),case=np.array(cases))
    shutil.copy2(__file__,stage/Path(__file__).name)
    write(stage/'dataset_audit.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),records=records,
        source_sha256=bindings,script_sha256=sha(__file__),dataset_sha256=sha(stage/'supervision.npz'),
        shared_observations=True,physical_preview_two_mm=3.,new_physics_steps=0,new_ppo_updates=0,
        scope='Offline labels on the same witnessed three-action states. Two-action labels preserve the initial requested direction when in range, not the entire20ms executed motion. No claim that these are successful two-action trajectories. Conversion does not execute inside either learned controller.'))
    write(stage/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),seeds=list(SEEDS),
        paired_initializers=str(SUPPLEMENT/'paired_action_ablation/protocol.json'),
        dataset_sha256=sha(stage/'supervision.npz'),
        offline_iterations=12000,fixed_checkpoint=12000,minibatch=512,case_fractions=[.5,.5],
        optimizer='Adam',learning_rate=2e-4,lr_multipliers_at_iterations={'4000':.3,'8000':.3},
        loss='Same clipped-action-equivalence loss, coordinate scales0.015/0.15/0.3, omit preview coordinate in two-output arm',
        feature_maps='Own identical192 steering/feed maps; three-output arm retains its96 preview-only features',
        teacher_scope='Both arms use the same fixed successful three-action trajectories; action coordinates differ by decoder. Conditional teacher-assisted training comparison, not a from-scratch action-space-only trial.',
        planned_ppo='Equal24 fresh PPO updates per arm followed by equal24 if needed under a recorded common rule; auxiliaries treated identically. Strict full evaluation after each fixed endpoint.',
        wall_limit_per_training_or_evaluation_round_s=1800,
        seed_replacement=False,selected_model_unchanged=True,
        adoption='All three main seeds need full success, physical constraints and unchanged corner/time gates. Two-action failures cannot establish a speedup. Preserve stronger historical baseline results.',
        stop='12000 offline iterations or wall limit; use fixed endpoint, preserve every failure.'))
    print('MATCHED_DATA',records,flush=True)


if __name__=='__main__':main()
