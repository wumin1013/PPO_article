"""Offline14input fitting from valid local controls and complete dense records."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,sha,write,verify_bound_inputs

STAGE=ROOT/'direct14_local_target_fit_round15'
PREVIOUS=ROOT/'direct14_previewcue_ppo_seed922302_round13'
INITIAL=PREVIOUS/'checkpoint_0024.pt'
PROBES=ROOT/'direct14_joint_response_probe_round14'
TARGETS=PROBES/'offline_target_feasibility.json'
CONTEXTS=PREVIOUS/'all34_contexts.pt'


def collect_targets(checks,saved):
    import numpy as np
    import torch
    from compact_joint_response import JointResponsePolicy
    from compact_previewcue_runtime import PreviewCueRuntime
    contexts=torch.load(CONTEXTS,map_location='cpu',weights_only=False)
    selected=json.loads(TARGETS.read_text(encoding='utf-8'))['selected']
    observations=[];actions=[];groups=[];provenance=[]
    for context,target in zip(contexts,selected):
        assert context['id']==target['id']
        policy=JointResponsePolicy(feed_gain=target['feed_gain'],preview_gain=target['preview_gain'])
        policy.load_state_dict(saved['policy'])
        env=PreviewCueRuntime(checks.references['original'])
        env.restore(context['state'])
        chunks=[]
        for _ in range(48):
            x=env.observation();z=policy.act(x,True)[0]
            _,_,_,info=env.step(z)
            observations.append(x);actions.append(info['held_action']);groups.append(context['id'])
            chunks.append(info['rows'])
            assert not env.done,'Only complete valid local trajectories may supply targets'
        with np.load(PROBES/f"response_{target['response']:02d}"/(context['id']+'.npz')) as data:
            assert np.array_equal(np.vstack(chunks),data['rows']),'Local supervision replay changed'
        provenance.append(dict(**target,exact_probe_replay=True))
    local_x=np.asarray(observations,dtype=np.float32)
    local_actions=np.asarray(actions)
    with np.load(PREVIOUS/'eval_0024_dense.npz') as d:
        dense_x=d['observations'];dense_actions=d['actions']
    with np.load(PREVIOUS/'eval_0024_original.npz') as d:
        old_x=d['observations'];old_actions=d['actions']
    # Preserve startup, terminal braking inputs, and straight/arc cruise away
    # from registered corner neighborhoods. No route identity becomes an input.
    geometry=checks.geometry['original']
    metric=json.loads((PREVIOUS/'eval_0024_original_metrics.json').read_text(encoding='utf-8'))
    keep=np.ones(len(old_x),dtype=bool)
    for c in metric['corners']:
        lo=max(0,c['first_index']//20-18)
        hi=min(len(old_x),c['last_index']//20+32)
        keep[lo:hi]=False
    eligible=np.flatnonzero(keep)
    ids=np.unique(np.r_[np.arange(min(100,len(old_x))),np.arange(max(0,len(old_x)-100),len(old_x)),
        eligible[np.unique(np.linspace(0,len(eligible)-1,min(512,len(eligible))).astype(int))]])
    xs=[local_x,dense_x,old_x[ids]]
    physical=[local_actions,dense_actions,old_actions[ids]]
    fractions=[.45,.50,.05]
    x=torch.tensor(np.vstack(xs),dtype=torch.float32)
    y=torch.tensor(np.vstack(physical)/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6]),dtype=torch.float32)
    original=y.clone();y[y[:,1]>=1.-1e-6,1]=1.04
    assert torch.allclose(y.clamp(-1,1),original.clamp(-1,1),atol=1e-6,rtol=0)
    w=torch.tensor(np.concatenate([np.full(len(a),f/len(a),np.float32) for a,f in zip(xs,fractions)]))
    np.savez_compressed(STAGE/'supervision.npz',observation=x.numpy(),target=y.numpy(),original_target=original.numpy(),weight=w.numpy())
    write(STAGE/'dataset.json',dict(samples=[len(a) for a in xs],group_weights=fractions,local_targets=provenance,
        anchor_indices=ids.tolist(),policy_input_dimension=14,policy_memory_scalars=0,
        local_source_scope='34individually selected valid local probes; not one complete policy',
        complete_sources=[str(PREVIOUS/f'eval_0024_{c}.npz') for c in ['original','dense']],
        dense_source_sha256=sha(PREVIOUS/'eval_0024_dense.npz'),
        local_target_selection_sha256=sha(TARGETS)))
    return x,y,w


def worker():
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_research_eval import CompleteChecks
    torch.set_num_threads(1);torch.manual_seed(922302)
    started=time.perf_counter();deadline=started+1650.
    checks=CompleteChecks(STAGE,deadline)
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False)
    x,y,w=collect_targets(checks,saved)
    policy=FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(saved['policy'])
    center=(x*w[:,None]).sum(0)
    scale=torch.sqrt(((x-center).square()*w[:,None]).sum(0)).clamp_min(.02)
    normalized=(x-center)/scale
    actor=copy.deepcopy(policy.actor)
    with torch.no_grad():
        raw=actor[0].weight.clone()
        actor[0].weight.mul_(scale);actor[0].bias.add_(raw@center)
        initial_error=float((actor(normalized)-policy.actor(x)).abs().max())
    assert initial_error<3e-5
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    action_scale=torch.tensor([.015,.15,.3])
    fitting=[];evaluations=[]
    print('SUPERVISION_READY',len(x),'wall',time.perf_counter()-started,flush=True)
    for epoch in range(1,12001):
        if time.perf_counter()>deadline-250.:
            break
        loss=((((actor(normalized)-y)/action_scale).square()).mean(-1)*w).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        if epoch%100==0:
            fitting.append(dict(epoch=epoch,loss=float(loss.detach()),wall_s=time.perf_counter()-started))
        if epoch not in [3000,6000,12000]:
            continue
        exported=copy.deepcopy(actor)
        with torch.no_grad():
            exported[0].weight.div_(scale);exported[0].bias.sub_(exported[0].weight@center)
            export_error=float((exported(x)-actor(normalized)).abs().max())
        assert export_error<3e-5
        policy.actor.load_state_dict(exported.state_dict())
        checkpoint=STAGE/f'fit_{epoch:04d}.pt'
        torch.save(dict(policy=policy.state_dict(),fit_optimizer=optimizer.state_dict(),normalized_actor=actor.state_dict(),
            center=center,scale=scale,epoch=epoch,interface=[14,3,922302],policy_class='FineCompactPolicy',feed_gain=1.,
            observation_dimension=14,policy_memory_scalars=0,source_checkpoint=str(INITIAL),source_checkpoint_sha256=sha(INITIAL),
            new_ppo_updates=0,inherited_current_encoder_ppo_updates=24,inherited_old_encoder_ppo_updates=40,
            training_kind='Offline fit to valid local optimized controls plus complete dense controls; no online teacher',
            export_equivalence_error=export_error),checkpoint)
        record=checks.run(policy,checkpoint,epoch,new_ppo_updates=0)
        record['fit_epoch']=epoch
        evaluations.append(record)
        write(STAGE/'evaluations.json',evaluations);write(STAGE/'fit_log.json',fitting)
    complete=[r for r in evaluations if all(v['success'] for v in r['results'].values())]
    eligible=[r for r in complete if r['results']['dense']['complete_time_s']<=15.06985 and
        r['results']['original']['complete_time_s']<444.315 and
        r['results']['original']['corner_comparisons']['historical_main']['not_worse']>=19]
    best=min(eligible,key=lambda r:(-r['results']['original']['corner_comparisons']['historical_main']['not_worse'],
         r['results']['original']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(status='offline_fit_completed' if epoch==12000 else 'wall_budget_completed',
        offline_iterations=epoch,new_ppo_updates=0,inherited_current_encoder_ppo_updates=24,
        inherited_old_encoder_ppo_updates=40,initial_checkpoint=str(INITIAL),evaluations=evaluations,
        intermediate_candidate=best,adopted=False,retained_checkpoint=str(INITIAL),
        wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))


def main():
    if '--worker' in sys.argv:
        worker();return
    manifest=verify_bound_inputs()
    assert sha(INITIAL)=='c9066cbeade4ad93e9a6f1982844ac06cbb4be0559f9d394846d87b95b1281e0'
    assert json.loads((PROBES/'receipt.json').read_text(encoding='utf-8'))['exit_code']==0
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__),ROOT/'compact_research_eval.py',ROOT/'compact_joint_response.py',ROOT/'compact_fine_policy.py',
        ROOT/'compact_previewcue_runtime.py',ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',
        BASE/'paper_complete_evaluation.py',BASE/'natural_case_audit.py',BASE/'natural_case_metrics.py']
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Offline compact policy fit after local parameter search',wall_budget_s=1800,worker_deadline_s=1650,child_timeout_s=1740,
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),source_sha256=hashes,bound_historical_manifest=manifest,
        targets=str(TARGETS),targets_sha256=sha(TARGETS),contexts_sha256=sha(CONTEXTS),
        iterations=12000,evaluation_iterations=[3000,6000,12000],new_ppo_updates=0,
        evidence_gap='Uniform output responses cannot preserve dense success. Valid per-corner local targets achieve33of34and147ms summed window savings, but are not one policy. Test whether existing14local features can learn these coordinated actions while retaining the dense successful controller.',
        procedure='Replay34selected valid local probes with exact raw-trajectory equality, then fit same14input2x64network. Group weights45percent local,50percent dense,5percent startup/terminal/cruise anchors. No corner ID, case ID, time, teacher or policy memory at execution. All complete evaluations start at task origin.',
        adoption='Intermediate screen: both full successes,dense<=15.06985s,original<444.315s and>=19historical nonworse corners; final acceptance still requires all34,credible historical long-time gain and both resolutions,matched two-action and repeats. No automatic final adoption.',
        stop='12000offline iterations,3scheduled checks,or wall deadline; do not increase budget within this round',
        observation_dimension=14,policy_memory_scalars=0,goal_complete=False,method_frozen=False))
    started=time.perf_counter();code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            child=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1740)
            status='finished';code=child.returncode
        except subprocess.TimeoutExpired:
            status='timeout'
    verify_bound_inputs()
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:
        print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':
    main()
