"""Bounded nonlinear function-capacity contrast, with unchanged state information."""
import time
import numpy as np
import torch
from nonlinear_feed_adapter import NonlinearFeedAdapter,masked_observation
from canonical_gate_state import CanonicalGatePolicy
from reflection_policy import reflect_observation
from train_pilot import save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1);torch.manual_seed(9181417)
    parent=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918'
    geometry=BASE/'canonical_gate_band_retained86_parameter_geometry_20260918'
    old_capacity=BASE/'canonical_gate_band_selective_cruise_actor_capacity_20260918'
    adopted=read(parent/'adoption.json');model=BASE/adopted['model']
    assert sha(model)==read(geometry/'results.json')['parent_sha256']
    stage=register('canonical_gate_band_nonlinear_cruise_capacity_20260918',
        [__file__,BASE/'nonlinear_feed_adapter.py',model,parent/'adoption.json',parent/'results.json',
         geometry/'geometry.npz',geometry/'results.json',geometry/'receipt.json',
         old_capacity/'results.json',old_capacity/'receipt.json'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=180,optimizer_steps=1200,
             architecture=[72,64,64,1],seed=9181417,target_feed_interval_mm_s=[99.,99.9],
             other_state_feed_change_mm_s=2.5,steering_and_preview_unchanged=True),
        'The prior frozen kernel and extended linear output spaces could not '
        'selectively raise all saved cruise targets while preserving other '
        'feed outputs. Test learned NONLINEAR features using the current86cd '
        'saved states. Add a reflection-invariant scalar feed MLP with zero '
        'initial output to the unchanged learned policy. Use exactly the same '
        '72 available slots and the same action-history/endpoint-bearing masks; '
        'no path IDs, new geometry input, template, runtime clamp or speed table. '
        'Preserve steering/preview functions exactly. Training targets are '
        'offline cruise-goal inequalities99..99.9 on saved eligible interiors '
        'and nominal+-2.5mm/s elsewhere, identical to the protected capacity '
        'problem. The nonlinear optimizer differs from the convex linear '
        'solver; failure is NOT proof of insufficient nonlinear capacity. '
        'Use1200 fixed Adam steps or165s, choose by independently decoded '
        'maximum inequality error, and record every residual. A successful '
        'fit establishes only saved-state expressibility, never actual cruise '
        'or shape. Subsequent bounded closed-loop diagnostics are separate; '
        'no automatic adoption, sealed access, physical relaxation or repeated '
        'width/learning-rate/seed sweep.')
    start=time.perf_counter();deadline=start+165
    with np.load(geometry/'geometry.npz') as z:
        observations=z['observation'].copy();latent=z['latent'][:,1].copy()
        nominal=z['decoded_actions'][:,1].copy();ids=z['cruise_state_indices'].copy()
        lower=z['lower_physical'].reshape(-1,3)[:,1].copy()
        upper=z['upper_physical'].reshape(-1,3)[:,1].copy()
        state_path=z['state_path'].copy()
    target=np.zeros(len(observations),dtype=bool);target[ids]=True
    lower[target]=np.arctanh(.98)-latent[target]
    upper[target]=np.arctanh(.998)-latent[target]
    md=torch.load(model,map_location='cpu',weights_only=False)
    policy=NonlinearFeedAdapter();missing=policy.load_state_dict(md['policy'],strict=False)
    assert not missing.unexpected_keys
    assert all(k.startswith(('feed_adapter.','adapter_')) for k in missing.missing_keys)
    for p in policy.parameters():p.requires_grad_(False)
    for p in policy.feed_adapter.parameters():p.requires_grad_(True)
    x=torch.tensor(observations)
    with torch.no_grad():
        masked=masked_observation(x);pair=torch.cat([masked,reflect_observation(masked)])
        policy.adapter_mean.copy_(pair.mean(0));policy.adapter_std.copy_(pair.std(0,unbiased=False).clamp(min=.1))
    # Neutral extension must exactly preserve individual deployed actions.
    original=CanonicalGatePolicy(256);original.load_state_dict(md['policy'])
    for obs in observations[::61]:
        assert np.array_equal(policy.act(obs,True)[0],original.act(obs,True)[0])
    lo=torch.tensor(lower,dtype=torch.float32);hi=torch.tensor(upper,dtype=torch.float32)
    optimizer=torch.optim.Adam(policy.feed_adapter.parameters(),lr=.003)
    best=None;history=[]
    def verify(step):
        nonlocal best
        with torch.no_grad():delta=policy.feed_correction(x).numpy().astype(float)
        feed=50*(1+np.tanh(latent+delta))
        lows=np.where(target,99.,np.maximum(nominal-2.5,0.))
        highs=np.where(target,99.9,np.minimum(nominal+2.5,100.))
        error=np.maximum(np.maximum(lows-feed,feed-highs),0.)
        score=(float(error.max()),float(np.mean(error**2)))
        if best is None or score<best['score']:
            best=dict(score=score,step=step,weights={k:v.detach().clone() for k,v in policy.feed_adapter.state_dict().items()})
        row=dict(step=step,maximum_feed_constraint_violation_mm_s=score[0],mean_squared_violation=score[1],
            target_min_feed_mm_s=float(feed[target].min()),
            target_requests_ge99=int((feed[target]>=99.-1e-8).sum()),
            nontarget_max_absolute_feed_change_mm_s=float(abs(feed[~target]-nominal[~target]).max()))
        history.append(row);print('NONLINEAR_CAPACITY',row,flush=True)
    verify(0);steps=0
    for step in range(1,1201):
        if time.perf_counter()>=deadline:break
        delta=policy.feed_correction(x)
        violation=torch.maximum(torch.relu(lo-delta),torch.relu(delta-hi))
        squared=violation.square()
        loss=squared.mean()+.1*torch.topk(squared,max(32,len(x)//20)).values.mean()+1e-7*delta.square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step();steps=step
        if step%100==0:verify(step)
    if not history or history[-1]['step']!=steps:verify(steps)
    policy.feed_adapter.load_state_dict(best['weights'])
    # Final constraints use the exact individual policy calls used at deployment.
    with torch.no_grad():
        actual=np.array([policy.act(obs,True)[0] for obs in observations])
        previous=np.array([original.act(obs,True)[0] for obs in observations])
    assert np.array_equal(actual[:,[0,2]],previous[:,[0,2]])
    feed=50*(1+np.tanh(actual[:,1].astype(float)))
    lows=np.where(target,99.,np.maximum(nominal-2.5,0.));highs=np.where(target,99.9,np.minimum(nominal+2.5,100.))
    errors=np.maximum(np.maximum(lows-feed,feed-highs),0.)
    np.savez_compressed(stage/'saved_state_fit.npz',feed=feed,nominal=nominal,errors=errors,
        target=target,state_path=state_path,latent=actual)
    save_torch(stage/'diagnostic_policy.pt',dict(policy=policy.state_dict(),architecture='nonlinear_feed_adapter_gate_band',
        parent_sha256=sha(model),environment_architecture=md['architecture'],diagnostic_only=True,
        learning_method='saved_state_nonlinear_feed_inequality_fit'))
    result=dict(parent_sha256=sha(model),history=history,selected_step=best['step'],optimizer_steps=steps,
        target_states=int(target.sum()),maximum_exact_feed_constraint_violation_mm_s=float(errors.max()),
        constraints_verified_with_1e6_mm_s_slack=bool(errors.max()<=1e-6),
        exact_target_requests_ge99=int((feed[target]>=99.-1e-8).sum()),
        other_action_functions_unchanged=True,physics_verified=False,learned_candidate_adopted=False,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_nonlinear_saved_cruise_capacity_fit',optimizer_steps=steps,
        selected_step=best['step'],new_physics_ticks=0,new_actor_decisions=0,
        saved_policy_queries=2*len(observations)+2*len(observations[::61]),wall_s=time.perf_counter()-start,
        learned_candidate_adopted=False,goal_complete=False))
    print('NONLINEAR_CAPACITY_FINAL',{k:v for k,v in result.items() if k!='history'},flush=True)


if __name__=='__main__':main()
