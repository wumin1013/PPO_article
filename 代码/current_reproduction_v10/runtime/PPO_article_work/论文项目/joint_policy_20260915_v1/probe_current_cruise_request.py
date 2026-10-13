"""Finite feed-request mechanism probes; no trained or deployed controller."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import score_complete
from complete_motion_metrics import complete_motion_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parent=BASE/'canonical_gate_band_bend_interpolated_learning_20260917'
    saved=read(parent/'results.json');selected=saved['full_candidate'];model=parent/saved['chosen']['model']
    support=BASE/'canonical_gate_band_current_cruise_learning_support_20260917'
    ids=[0,14,57];anchor=10;target=float(np.arctanh(.99))
    stage=register('canonical_gate_band_current_cruise_request_probe_20260917',
        [__file__,model,parent/'results.json',parent/'receipt.json',support/'results.json',
         support/'receipt.json',BASE/'joint_runtime.py',BASE/'probe_coordinated_suffix.py',BASE/'validate_actions.py'],
        dict(paths=ids,decision_anchor=anchor,feed_target_mm_s=99.5,
             feed_latent_increment_caps=[.05,None],actor_calls=2000,new_physics_ticks=60000,
             saved_prefix_geometry_ticks=1000,wall_s=60,maximum_complete_suffixes=6),
        'The saved-state audit finds requests near98.4mm/s on the long straight '
        'and current local-step limits too small to reach99 in most saved '
        'states. At the exact saved200ms state on a shallow turn, a155-degree '
        'turn, and one142.5mm straight, compare two sustained feed-only '
        'diagnostic interventions in the same target law: raise feed toward '
        'at least99.5mm/s by min(latent_cap,max(0,target_latent-current_latent)). '
        'The cap is either0.05 (the existing saved-credit-state allowance) or '
        'unrestricted by the learning proposal; runtime physical V/A/J, '
        'corridor, gates and original brake triggering stay unchanged. '
        'Steering/preview remain the learned outputs at each actual new '
        'state. This isolates the sustained request magnitude within that '
        'target law, not policy learning. Report full completion, raw shape, '
        'speed/cruise and actual brake/time changes. Failed prefixes have no '
        'comparable full timing. Do not deploy this clamp, label it learned, '
        'use it as a trajectory teacher or infer it is adequate for corners. '
        'Stop after six suffixes and use the causal outcome to choose the '
        'next learning-coordinate/credit mechanism.')
    start=time.perf_counter();torch.set_num_threads(1)
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture'])
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    baselines=[];records=[]
    for path in ids:
        j=next(j for j,r in enumerate(selected['results']) if r['original_path']==path)
        with np.load(BASE/selected['traces'][j]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,anchor*20,costs,env_factory=factory);state=env.state()
        latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
        assert np.array_equal(env.decode_action(latent),actions[anchor])
        baselines.append(dict(path=path,success=True,trace=selected['traces'][j],
            complete_time_s=len(rows)*.001,brake_tick=int(np.flatnonzero(rows[:,12])[0]),
            anchor_speed_mm_s=float(np.linalg.norm(env.velocity)),anchor_feed_mm_s=float(actions[anchor,1]),
            motion=complete_motion_metrics(ref,rows),score=score_complete(ref,rows,actions)))
        for tag,cap in [('small',.05),('target',None)]:
            env.restore(state);chunks=[];held=[]
            while not env.done:
                assert time.perf_counter()-start<55 and costs['actor_calls']<2000 and costs['new_physics_ticks']<57880
                action=policy.act(env.observation(),True)[0].astype(float);costs['actor_calls']+=1
                increment=max(0.,target-action[1])
                action[1]+=increment if cap is None else min(cap,increment)
                _,_,_,info=env.step(action);chunks.append(info['rows']);held.append(info['held_action'])
                costs['new_physics_ticks']+=len(info['rows'])
            full=np.vstack([rows[:anchor*20],*chunks]);full_actions=np.vstack([actions[:anchor],held])
            audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            name=f'path{path}_{tag}.npz';np.savez_compressed(stage/name,reference=ref,rows=full,actions=full_actions)
            braking=np.flatnonzero(full[:,12])
            r=dict(path=path,arm=tag,feed_latent_increment_cap=cap,success=success,reason=env.reason,
                trace=name,brake_tick=int(braking[0]) if len(braking) else None,
                complete_time_s=len(full)*.001 if success else None,
                complete_time_change_s=(len(full)-len(rows))*.001 if success else None,
                audit=audit,geometry=env.tracker.snapshot(1.25),
                motion=complete_motion_metrics(ref,full) if success else None,
                score=score_complete(ref,full,full_actions,env) if success else None)
            records.append(r)
            write_json(stage/'results.json',dict(parent_sha256=sha(model),baselines=baselines,
                records=records,costs=costs,diagnostic_only=True,goal_complete=False))
            print('CRUISE_REQUEST_PROBE',path,tag,success,r['complete_time_change_s'],
                  None if not success else r['motion']['cruise_ge99_s'],flush=True)
    receipt(stage,dict(status='completed_current_sustained_cruise_request_probe',suffixes=len(records),
        complete_suffixes=sum(r['success'] for r in records),costs=costs,
        wall_s=time.perf_counter()-start,diagnostic_only=True,goal_complete=False))


if __name__=='__main__':main()
