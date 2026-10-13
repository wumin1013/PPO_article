"""One bounded action-forecast diagnostic around the historical controller."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,BASE,HIST,CASE,sha,write,verify_bound_inputs

STAGE=ROOT/'historical_dense_predictive_action_diagnostic'


def worker():
    import numpy as np
    import torch
    from candidate_policy_factory import candidate_policy
    from turn_aware_projector import environment_factory
    from local_arc_support import install_local_arc_support,ClusteredPreviewFloorPolicy
    from predictive_action_diagnostic import PredictiveActionDiagnostic,encode
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from analyze_dense_trace import analyze
    verify_bound_inputs()
    torch.set_num_threads(1)
    start=time.perf_counter()
    selection=json.loads((HIST/'selected_policy.json').read_text(encoding='utf-8'))
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    base,_=install_local_arc_support(candidate_policy(metadata,selection))
    base=ClusteredPreviewFloorPolicy(base,3.)
    policy=PredictiveActionDiagnostic(base,start+1050.)
    env=environment_factory(reference)
    policy.bind(env)
    # Two independent physical instances must give the same forecasted motion.
    # The nominal policy has not been advanced during this check.
    state=env.state()
    probe_action=np.array([.01,70.,3.5])
    latent=encode(probe_action)
    assert np.max(abs(env.decode_action(latent,None)-probe_action))<1e-10
    policy.prediction.restore(state)
    policy.prediction.pending_pulse=np.zeros(3)
    _,_,_,pred=policy.prediction.step(latent)
    _,_,_,real=env.step(latent)
    assert np.array_equal(pred['rows'],real['rows'])
    env.restore(state)
    env.pending_pulse=np.zeros(3)
    write(STAGE/'interface_validation.json',dict(action_encoding_error_below_1e_minus10=True,
        independent_physical_instances_next_step_identical=True,checked_physics_ticks=40))
    result=evaluate(policy,env,STAGE/'trace.npz',start+1050.,bind_policy=True)
    data=np.load(STAGE/'trace.npz')
    result['analytic']=analytic_command_audit(reference[0],data['rows'],geometry)
    if not result['analytic']['analytic_corridor_pass']:
        result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
    result.update(goal_complete=False,method_frozen=False,ppo_updates=0,
        study_kind='Online finite-horizon model-based action search; not PPO training',
        forecast_physics_ticks=policy.forecast_ticks,real_physics_ticks=len(data['rows']),
        changed_decisions=sum(r['selected']!='nominal' for r in policy.records),
        no_feasible_forecast_decisions=sum(r['feasible_candidates']==0 for r in policy.records),
        measured_wall_s=time.perf_counter()-start)
    write(STAGE/'result.json',result)
    write(STAGE/'action_search.json',policy.records)
    write(STAGE/'gate_events.json',env.tracker.events)
    if result['success']:write(STAGE/'all84_metrics.json',analyze(STAGE/'trace.npz',STAGE/'result.json'))
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    if '--worker' in sys.argv:
        worker();return
    manifest=verify_bound_inputs()
    STAGE.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'predictive_action_diagnostic.py',ROOT/'local_arc_support.py',ROOT/'evaluate_historical_dense.py']
    hashes={str(p):sha(p) for p in files}
    (STAGE/'source').mkdir()
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Single finite-horizon physical action-search diagnostic',
        wall_budget_s=1200,child_timeout_s=1150,simulation_deadline_s=1050,
        ppo_updates=0,physics_dt_s=.001,forecast_s=.12,horizon_decisions=6,
        source_sha256=hashes,inherited_manifest=manifest,reference_sha256=sha(CASE/'reference.csv'),
        evidence_gap='Scalar rule scans and a short residual PPO pilot did not resolve dynamic lag on clustered turns.',
        candidate_definition='Nominal then, only if its120ms held-action forecast is not feasible within1.24mm, eleven predefined one-channel or joint action changes. Choose minimum normalized intervention among feasible forecasts; otherwise preserve the longest feasible prefix diagnostic.',
        adoption='Diagnostic only. Full original geometry/dynamics/coverage/brake/rest must pass for time comparison. A successful search must not be described as PPO learning or as proof of final mechanism.',
        stop='One fixed algorithm; stop on real failure or deadline. No grid expansion in this round; preserve failed forecasts and real trace.',
        scope='No S2 or fixed-preview optimization. Online forecast cost is recorded. A120ms forecast is not a recursive feasibility guarantee.'))
    started=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1150)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-5000:])


if __name__=='__main__':main()
