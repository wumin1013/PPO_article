"""Verify transparent accepted stepping and exact failure-prefix handling."""
import argparse
from types import SimpleNamespace
import numpy as np
import torch
from evaluate_historical_dense import ROOT, CASE, write, sha
from train_mixed_resolution_ppo import policy_from
from direct_two_action_control import DirectTwoRuntime
from training_rejection_handling import sampled_step


def main():
    p=argparse.ArgumentParser();p.add_argument('--dt',type=float,required=True);a=p.parse_args()
    if a.dt!=.001:
        from run_refined_turn_candidate import configure_resolution
        configure_resolution(a.dt)
    ref=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    policy=policy_from(torch.load(ROOT/'direct14_two_action_initialization_round27/d_3/fit_12000.pt',map_location='cpu',weights_only=False),922302)
    raw=DirectTwoRuntime(ref);wrapped=DirectTwoRuntime(ref);failures=[];error=0.
    for _ in range(8):
        action=policy.act(raw.observation(),True)[0]
        r1=raw.step(action);r2=sampled_step(wrapped,action,a.dt,failures.append)
        assert np.array_equal(r1[3]['rows'],r2[3]['rows'])
        assert np.array_equal(r1[0],r2[0]) and r1[1:3]==r2[1:3]
        error=max(error,float(abs(r1[3]['rows']-r2[3]['rows']).max()))
    assert not failures and 'update' not in vars(wrapped.tracker)
    original=wrapped.env.experimental_projector.enforce_kinematic_constraints;calls=0
    def reject(v,acc,j,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==4:raise ValueError('Inductive backup failed numerical certification')
        return original(v,acc,j,*args,**kwargs)
    wrapped.env.experimental_projector.enforce_kinematic_constraints=reject
    before=wrapped.ticks
    result=sampled_step(wrapped,policy.act(wrapped.observation(),True)[0],a.dt,failures.append)
    assert result[2] and not wrapped.success and len(failures)==1
    assert wrapped.ticks-before==3 and result[3]['rows'].shape==(3,4)
    assert abs(result[3]['elapsed_s']-3*a.dt)<1e-15
    assert 'update' not in vars(wrapped.tracker)
    report=dict(dt_s=a.dt,accepted_decisions=8,maximum_accepted_row_difference=error,
        injected_failure_expected_ticks=3,recorded_ticks=failures[0]['actual_accepted_ticks'],
        failure_terminal=True,full_evaluation_modified=False,helper_sha256=sha(ROOT/'training_rejection_handling.py'))
    write(ROOT/f'training_rejection_handling_test_{a.dt}.json',report);print(report)


if __name__=='__main__':main()
