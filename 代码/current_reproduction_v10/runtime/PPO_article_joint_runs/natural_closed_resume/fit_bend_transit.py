"""Fit existing coefficients to local crossing time from complete-route states.

Conditional parameter optimization only: no PPO updates or new model seeds.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,copy,json,pickle,shutil,sys,time
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from bend_conditioned_policy import BendConditionedPolicy


def crossing_time(file,context):
    """Linear command crossing of the fixed outgoing 50mm plane, no tick bias."""
    rows=np.load(file)['rows'];c=context['corner'];ref=context['reference']
    k=c['reference_vertex_index'];u=ref[k+1]-ref[k];u=u/np.linalg.norm(u)
    xy=np.vstack([context['state']['position'],rows[:,:2]])
    q=(xy-ref[k])@u
    hits=np.flatnonzero((q[:-1]<50.) & (q[1:]>=50.))
    if not len(hits):return None
    i=int(hits[-1]);fraction=(50.-q[i])/(q[i+1]-q[i])
    return float((i+fraction)*.001)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('parent');ap.add_argument('fixed')
    ap.add_argument('--budget',type=float,default=1200)
    ap.add_argument('--stage',default='joint_bend_transit_fit');args=ap.parse_args()
    parent=Path(args.parent);fixed=Path(args.fixed);root=OUT/'all34_dominance_research'
    stage=root/args.stage;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    selection=json.loads((parent/'selected_policy.json').read_text(encoding='utf-8'))
    cache=parent/'causal_contexts.pkl'
    with cache.open('rb') as f:contexts=pickle.load(f)
    assert len(contexts)==34
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    torch.set_num_threads(1)
    target_paths=[fixed/'full_metrics.json',fixed/'half_ms/corner_metrics.json',
                  BASE/'paper_evidence_review/natural_closed_case/s2_paper_metrics.json']
    maps=[{c['id']:c['speed_deficit_mm'] for c in json.loads(p.read_text(encoding='utf-8'))['corners']} for p in target_paths]
    targets={k:min(m[k] for m in maps) for k in maps[0]}
    ids=[4,14,6,28,20,31,2,10,16,18,21,26,29,30,23]
    training=[contexts[i] for i in ids]
    sources=[Path(__file__),OUT/'probe_all34_feed_boundary.py',OUT/'bend_conditioned_policy.py',
             OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py']
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='conditional_existing_coefficient_transit_optimization',
        parent=str(parent),wall_budget_s=args.budget,ppo_updates=0,conditional_search_seeds=[922221,922222,922223],
        source_sha256={str(p):sha(p) for p in sources},cache_sha256=sha(cache),
        target_sha256={str(p):sha(p) for p in target_paths},targets=targets,
        objective='Local outgoing50mm crossing time plus0.2s/mm positive gap to baseline envelope; final all34 dominance is mandatory. Crossing time interpolates the linear command crossing, removing only sampling overshoot. Registered deficit windows unchanged.',
        observations='Existing continuous bend and tracking error only; no new features, corner identifiers, clock schedules or route coordinates in added residual.',
        adopt='All34 actual contexts must pass physics and deficit envelope before a full-lap candidate is selected. Separate complete and half-ms runs mandatory.',
        stop='Failing context stops proposal; stop optimization at85 percent and checks before wall budget. Retain every outcome.'))
    def factory(ref):
        env=environment_factory(ref)
        if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
        return env
    local.environment_factory=factory;start=time.perf_counter();deadline=start+args.budget-10
    trials=[];retained=[]
    def run(spec,label,subset,until):
        records=[]
        for c in subset:
            if time.perf_counter()>until-3:break
            file=stage/(label+'_'+c['corner']['id']+'.npz')
            rec=local.rollout(metadata,c,spec,file,until,BendConditionedPolicy)
            rec['crossing_time_s']=crossing_time(file,c) if rec['context_pass'] else None
            if rec['complete_window']:
                rec['positive_deficit_gap_mm']=max(rec['speed_deficit_mm']-targets[c['corner']['id']]-1e-7,0.)
            records.append(rec)
            if not rec['context_pass'] or not rec['complete_window']:break
        passed=len(records)==len(subset) and all(r['context_pass'] and r['complete_window'] for r in records)
        transit=sum(r['crossing_time_s'] for r in records) if passed else None
        gap=sum(r['positive_deficit_gap_mm'] for r in records) if passed else None
        cost=transit+.2*gap if passed else None
        record=dict(label=label,spec=spec,records=records,passed=passed,transit_time_s=transit,
            positive_gap_sum_mm=gap,dominance=passed and gap<=1e-12,cost=cost)
        trials.append(record);write_json(stage/'results.json',trials)
        print('TRANSIT_FIT',label,passed,'cost',cost,'gap',gap,'n',len(records),'wall',round(time.perf_counter()-start,2),flush=True)
        return record
    def key(r):return (not r['passed'],r['cost'] if r['passed'] else 1000-len(r['records']))
    original=copy.deepcopy(selection['bend_spec'])
    prior=run(original,'identity',training,deadline)
    proposals=[]
    for row,amount in [(2,.15),(2,-.15),(0,.02),(0,-.02),(1,.3),(1,-.3)]:
        for col in [2,1]:
            spec=copy.deepcopy(original);spec['bend_weights'][row][col]+=amount
            proposals.append((f'grid_r{row}_c{col}_d{amount}',spec))
    for label,spec in proposals:
        if time.perf_counter()>start+args.budget*.4:break
        r=run(spec,label,training,min(deadline,start+args.budget*.45))
        if key(r)<key(prior):prior=r
    for seed in [922221,922222,922223]:
        rng=np.random.default_rng(seed);best=prior
        until=min(start+args.budget*.85,time.perf_counter()+args.budget*.15)
        for step in range(20):
            if time.perf_counter()>until-20:break
            spec=copy.deepcopy(best['spec']);w=np.asarray(spec['bend_weights'])
            scale=np.array([.012,.20,.12])[:,None]*(1. if step<10 else .55)
            delta=rng.normal(size=(3,3))*scale
            # Coordinate and joint proposals both stay in the existing nine coefficients.
            if step%2==0:
                mask=np.zeros((3,3));mask[rng.integers(0,3),rng.integers(0,3)]=1.;delta*=mask
            w+=delta;spec['bend_weights']=w.tolist()
            r=run(spec,f'seed{seed}_step{step}',training,until)
            if key(r)<key(best):best=r
        retained.append(dict(seed=seed,selected_label=best['label'],spec=best['spec'],cost=best['cost']))
        write_json(stage/'retained.json',retained)
    checked=[]
    eligible=sorted([r for r in trials if r['dominance']],key=lambda r:r['transit_time_s'])
    seen=set()
    for candidate in eligible:
        signature=json.dumps(candidate['spec'],sort_keys=True)
        if signature in seen:continue
        seen.add(signature)
        if time.perf_counter()>deadline-50:break
        r=run(candidate['spec'],'all34_'+candidate['label'],contexts,deadline)
        checked.append(dict(seed=selection['seed'],result=r));write_json(stage/'all34_checks.json',checked)
        if r['dominance']:
            write_json(stage/'selected_policy.json',dict(selection,bend_spec=r['spec'],goal_complete=False,
                full_lap_not_tested=True,selection_method='conditional_transit_objective',parent=str(parent)))
            break
    write_json(stage/'receipt.json',dict(goal_complete=False,wall_s=time.perf_counter()-start,
        ppo_updates=0,trials=len(trials),all34_checked=len(checked),selected=(stage/'selected_policy.json').exists()))


if __name__=='__main__':main()
