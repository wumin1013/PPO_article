"""Matched perturbation duration and complete-time sensitivity; diagnostic only."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,environment_factory
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import score_complete
from complete_motion_metrics import complete_motion_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_waveform_learned_update_20260917'
    selected=read(parent/'training_selection.json');model=parent/selected['model']
    ids=[0,7,15,48,57];anchor=10;amplitude=.1
    stage=register('canonical_feed_credit_duration_probe_20260917',
        [__file__,model,parent/'training_selection.json',parent/'receipt.json',
         BASE/'canonical_gate_learned_trajectory_review_20260917/time_credit_observation.json',
         BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py',
         BASE/'probe_coordinated_suffix.py',BASE/'validate_actions.py'],
        dict(paths=ids,decision_anchor=anchor,feed_latent_offset=amplitude,
             signs=[-1,1],durations=['one_decision','complete_suffix'],
             new_suffixes=20,actor_calls=3500,new_physics_ticks=90000,wall_s=120),
        'The prior canonical parent has5795/5820 single20ms perturbations with '
        'unchanged complete time. Test the duration hypothesis on the newer '
        'validated64/64 research model. At the exact saved200ms state on three '
        'isolated turns and two long straight TRAINING routes, apply the SAME '
        '+/-0.1 feed latent offset for one20ms action or every subsequent feedback '
        'action until actual completion/failure. Check speed>80 and acceleration '
        '<200 at the anchor to avoid an initial acceleration-saturation contrast. '
        'Initial held actions and their first20ms physical response must match '
        'exactly across durations for each sign. Other feedback outputs remain '
        'learned and respond to each branch\'s true state. This changes temporal '
        'coherence only within each paired sign, not reward, guide or executor. '
        'Record full time, shape, cruise and failures; failed prefixes are never '
        'fast completion. These are unlearned diagnostic interventions, not a '
        'new candidate or deployed feed rule. No development or sealed data.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    p=CanonicalGatePolicy(256)
    p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    baselines=[];records=[];pairs=[]
    for path in ids:
        j=next(i for i,r in enumerate(selected['results']) if r['original_path']==path)
        with np.load(BASE/selected['traces'][j]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,anchor*20,costs,env_factory=environment_factory)
        state=env.state();speed=float(np.linalg.norm(env.velocity));acceleration=float(np.linalg.norm(env.acceleration))
        assert speed>80 and acceleration<200,(path,speed,acceleration)
        latent=p.act(env.observation(),True)[0];costs['actor_calls']+=1
        assert np.array_equal(env.decode_action(latent),actions[anchor])
        baselines.append(dict(path=path,complete_time_s=len(rows)*.001,anchor_speed_mm_s=speed,
            anchor_acceleration_mm_s2=acceleration,**score_complete(ref,rows,actions),
            motion=complete_motion_metrics(ref,rows)))
        for sign in (-1,1):
            first_response=None;pair=[]
            for mode in ('one_decision','complete_suffix'):
                env.restore(state);chunks=[];held=[];step=0
                while not env.done:
                    assert time.perf_counter()-start<120
                    assert costs['actor_calls']<3500 and costs['new_physics_ticks']<87880
                    action=p.act(env.observation(),True)[0].astype(float);costs['actor_calls']+=1
                    if mode=='complete_suffix' or step==0:action[1]+=sign*amplitude
                    _,_,_,info=env.step(action);chunks.append(info['rows']);held.append(info['held_action']);step+=1
                    costs['new_physics_ticks']+=len(info['rows'])
                if first_response is None:first_response=chunks[0].copy()
                else:assert np.array_equal(first_response,chunks[0])
                full=np.vstack([rows[:anchor*20],*chunks]);full_actions=np.vstack([actions[:anchor],held])
                audit=difference_audit(ref[0],full)
                success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                name=f'p{path}_{sign:+d}_{mode}.npz'
                np.savez_compressed(stage/name,reference=ref,rows=full,actions=full_actions)
                r=dict(path=path,sign=sign,duration=mode,success=success,reason=env.reason,
                    complete_time_s=len(full)*.001 if success else None,
                    time_change_s=(len(full)-len(rows))*.001 if success else None,
                    trace=name,audit=audit,geometry=env.tracker.snapshot(1.25),
                    **score_complete(ref,full,full_actions,env),motion=complete_motion_metrics(ref,full))
                records.append(r);pair.append(r)
                print('FEED_DURATION',path,sign,mode,success,r['time_change_s'],flush=True)
            pairs.append(dict(path=path,sign=sign,identical_first20ms=True,
                complete_pair=all(r['success'] for r in pair),
                time_changes_s=[r['time_change_s'] for r in pair]))
        write_json(stage/'results.json',dict(baselines=baselines,records=records,pairs=pairs,
            parent_sha256=sha(model),costs=costs,diagnostic_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_matched_feed_credit_duration_probe',costs=costs,
        complete_suffixes=sum(r['success'] for r in records),suffixes=len(records),
        wall_s=time.perf_counter()-start,new_learned_models=0,goal_complete=False))


if __name__=='__main__':main()
