"""Complete an existing one-sided factorial with only48 missing axial branches."""
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_entry_action_interactions import score
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916';model=parent/'candidate_1.pt'
    source=BASE/'entry_action_interaction_probe_20260916';existing=read(source/'results.json')
    stage=register('symmetric_entry_credit_probe_20260916',
        [__file__,model,parent/'receipt.json',source/'receipt.json',source/'results.json',
         BASE/'probe_entry_action_interactions.py',BASE/'probe_coordinated_suffix.py'],
        dict(reused_axial_suffixes=48,new_suffixes=48,new_decisions=6000,new_physics_ticks=130000,wall_s=120),
        'Reuse all48 single-axis positive-factor branches and16 original states from the completed '
        'entry factorial. Add only the opposite offset for each axis, same60ms duration, frozen '
        'retained actor and original executor; finish every branch to full stop or genuine failure. '
        'Use complete-extent two-edge return for finite sensitivities: central when both sides '
        'complete, feasible one-sided when only one completes, zero when neither completes. '
        'Retain every failure as a local excluded-direction observation, not a good prefix. '
        'These samples support a matched constrained/unconstrained parameter-update study; '
        'finite sensitivities and local exclusions are not exact gradients or safety certificates.')
    started=time.monotonic();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True);policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    records=[]
    for row in existing['records']:
        if sum(row['bits'])!=1:continue
        axis=int(np.argmax(row['bits']))
        records.append(dict(row,axis=axis,factor_sign=1,reused=True,trace=(source/row['trace']).relative_to(BASE).as_posix()))
    for base in existing['baselines']:
        path=base['path']
        with np.load(parent/'training_1'/f'{path}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        for anchor in base['anchors']:
            frame=anchor*20;env=reconstruct(ref,rows,actions,frame,costs);initial=env.state()
            assert np.array_equal(env.decode_action(policy.act(env.observation(),True)[0]),actions[anchor]);costs['actor_calls']+=1
            for axis in range(3):
                positive=next(r for r in records if r['path']==path and r['anchor']==anchor and r['axis']==axis and r['factor_sign']==1)
                delta=-np.asarray(positive['delta']);env.restore(initial);chunks=[];held=[];observations=[];count=0
                while not env.done:
                    assert time.monotonic()-started<120 and costs['actor_calls']<6000 and costs['new_physics_ticks']<127880
                    obs=env.observation();latent=policy.act(obs,True)[0];costs['actor_calls']+=1
                    if count<3:latent=latent+delta
                    _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);observations.append(obs);count+=1
                    costs['new_physics_ticks']+=len(info['rows'])
                full=np.vstack([rows[:frame],*chunks]);allactions=np.vstack([actions[:anchor],held])
                audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                value=score(ref,full,allactions,env);name=f'p{path}_a{anchor}_axis{axis}_opposite.npz'
                np.savez_compressed(stage/name,reference=ref,rows=full,actions=allactions,observation=np.asarray(observations),
                    anchor=anchor,delta=delta,intervention_decisions=min(count,3))
                records.append(dict(path=path,anchor=anchor,axis=axis,factor_sign=-1,reused=False,
                    delta=delta.tolist(),trace=(stage/name).relative_to(BASE).as_posix(),success=success,reason=env.reason,
                    full_time_s=len(full)*.001 if success else None,audit=audit,geometry=env.tracker.snapshot(1.25),**value))
            print('SYMMETRIC',path,anchor,'opposite_complete',sum(r['success'] for r in records[-3:]),flush=True)
            write_json(stage/'results.json',dict(baselines=existing['baselines'],records=records,costs=costs))
    sensitivities=[]
    for base in existing['baselines']:
        for anchor in base['anchors']:
            gradient=[];modes=[]
            for axis in range(3):
                pair=[r for r in records if r['path']==base['path'] and r['anchor']==anchor and r['axis']==axis]
                good=[r for r in pair if r['success']]
                if len(good)==2:
                    derivative=(pair[0]['return_full']-pair[1]['return_full'])/(pair[0]['delta'][axis]-pair[1]['delta'][axis]);mode='central'
                elif len(good)==1:
                    derivative=(good[0]['return_full']-base['return_full'])/good[0]['delta'][axis];mode='feasible_one_sided'
                else:derivative=0.;mode='both_sides_failed'
                gradient.append(float(derivative));modes.append(mode)
            sensitivities.append(dict(path=base['path'],anchor=anchor,gradient=gradient,modes=modes))
    write_json(stage/'sensitivities.json',dict(records=sensitivities,full_extent=True,failed_axes=sum(not r['success'] for r in records)))
    receipt(stage,dict(status='completed_missing_opposite_action_branches',costs=costs,new_suffixes=48,
        reused_suffixes=48,new_successes=sum(r['success'] for r in records if not r['reused']),wall_s=time.monotonic()-started))


if __name__=='__main__':main()
