"""Same deterministic feed-coordination grid for Joint and Fixed candidates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,pickle,time,copy,shutil,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from bend_conditioned_policy import BendConditionedPolicy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('fit_folder');ap.add_argument('--arm',choices=['joint','fixed'],required=True);args=ap.parse_args()
    torch.set_num_threads(1);fit=Path(args.fit_folder);root=OUT/'all34_dominance_research';stage=fit.with_name(fit.name+'_coordination_margin');stage.mkdir(exist_ok=False)
    selection=json.loads((fit/'selected_policy.json').read_text(encoding='utf-8'))
    cache=root/(args.arm+'_fit_seed922202_full')/'causal_contexts.pkl'
    with cache.open('rb') as f:contexts=pickle.load(f)
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    ids=[4,29,18,21,10,16,26,2,6,28,20,30,31,14,23];training=[contexts[i] for i in ids]
    sources=[Path(__file__),OUT/'bend_conditioned_policy.py',OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py']
    (stage/'source').mkdir()
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='targeted_medium_bend_coordination_grid',arm=args.arm,wall_budget_s=600,
        source_sha256={str(p):sha(p) for p in sources},source_selection_sha256=sha(fit/'selected_policy.json'),source_cache_sha256=sha(cache),
        proposals=['zero','medium_feed03','medium_feed06','medium_feed09','balanced_feed','steering_feed','preview_plus_feed','preview_minus_feed','steering_plus_feed'],effect='Continuous bend-dependent three-action coefficient changes; no corner IDs in the policy.',
        objective='Feasibility then positive per-corner gap against strongest conditionally verified Fixed, with0.002 whole-local deficit regularizer; targets are diagnostics pending true full-lap evidence.',
        adopt='Only select for full-lap validation after34 contexts pass.',stop='Stop each proposal on first failed context and all work before590s.',ppo_updates=0))
    def factory(ref):
        env=environment_factory(ref)
        if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
        return env
    local.environment_factory=factory;start=time.perf_counter();trials=[]
    target_file=root/'fixed_bend_physical_fit_saturation_feed_grid/all34_checks.json'
    fixed=json.loads(target_file.read_text(encoding='utf-8'))[-1]['result']
    targets={r['corner']:r['speed_deficit_mm'] for r in fixed['records']}
    write_json(stage/'target_binding.json',dict(source=str(target_file),sha256=sha(target_file),targets=targets))
    def run(spec,label,subset):
        records=[]
        for c in subset:
            if time.perf_counter()>start+580:break
            r=local.rollout(metadata,c,spec,stage/(label+'_'+c['corner']['id']+'.npz'),start+590,BendConditionedPolicy);records.append(r)
            if not r['context_pass'] or not r['complete_window']:break
        passed=len(records)==len(subset) and all(r['context_pass'] and r['complete_window'] for r in records)
        cost=sum(max(r['speed_deficit_mm']-targets[r['corner']]-1e-7,0.)+.002*r['total_local_speed_deficit_mm'] for r in records) if passed else None
        rec=dict(label=label,passed=passed,spec=spec,fixed_d=selection['fixed_d'],cost=cost,records=records)
        trials.append(rec);write_json(stage/'results.json',trials)
        print('BEND_FEED_GRID',args.arm,label,passed,cost,'contexts',len(records),'wall',time.perf_counter()-start,flush=True)
        return rec
    proposals=[('zero',[0,0,0],[0,0,0],[0,0,0]),
        ('medium_feed03',[0,.3,0],[0,0,0],[0,0,0]),
        ('medium_feed06',[-.1,.6,0],[0,0,0],[0,0,0]),
        ('medium_feed09',[-.2,.9,0],[0,0,0],[0,0,0]),
        ('balanced_feed',[.15,.3,0],[0,0,0],[0,0,0]),
        ('steering_feed',[-.1,.6,0],[-.01,-.01,0],[0,0,0]),
        ('preview_plus_feed',[-.2,.9,0],[0,0,0],[0,.15,0]),
        ('preview_minus_feed',[-.2,.9,0],[0,0,0],[0,-.15,0]),
        ('steering_plus_feed',[-.2,.9,0],[0,.01,0],[0,0,0])]
    for label,feed,steer,preview in proposals:
        spec=copy.deepcopy(selection['bend_spec'])
        spec['bend_weights']=(np.asarray(spec['bend_weights'])+np.array([steer,feed,preview])).tolist()
        run(spec,label,training)
    eligible=sorted([r for r in trials if r['passed']],key=lambda r:r['cost']);checked=[]
    for candidate in eligible:
        r=run(candidate['spec'],'all34_'+candidate['label'],contexts)
        checked.append(dict(seed=selection['seed'],result=r));write_json(stage/'all34_checks.json',checked)
        if r['passed']:
            write_json(stage/'selected_policy.json',dict(selection,bend_spec=r['spec'],goal_complete=False,full_lap_not_tested=True));break
    write_json(stage/'receipt.json',dict(goal_complete=False,wall_s=time.perf_counter()-start,ppo_updates=0,all34_passed=any(r['result']['passed'] for r in checked)))

if __name__=='__main__':main()
