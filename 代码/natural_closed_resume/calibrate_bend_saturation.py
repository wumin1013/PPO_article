"""Matched saturation calibration for an already fitted continuous bend residual."""
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
    torch.set_num_threads(1);fit=Path(args.fit_folder);root=OUT/'all34_dominance_research';stage=fit.with_name(fit.name+'_saturation');stage.mkdir(exist_ok=False)
    selection=json.loads((fit/'selected_policy.json').read_text(encoding='utf-8'))
    with (root/(args.arm+'_fit_seed922202_full')/'causal_contexts.pkl').open('rb') as f:contexts=pickle.load(f)
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    registration=json.loads((fit/'registration.json').read_text(encoding='utf-8'));targets=registration['targets']
    train_ids=[23,21,26,6,28,30,29,4,20,16,10,18,31,14,2];training=[contexts[i] for i in train_ids]
    sources=[Path(__file__),OUT/'bend_conditioned_policy.py',OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py']
    (stage/'source').mkdir()
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='paired_cruise_saturation_grid',wall_budget_s=600,arm=args.arm,
        source_sha256={str(p):sha(p) for p in sources},source_selection_sha256=sha(fit/'selected_policy.json'),
        low_values=[.85,.90,.93,.945],objective='Feasibility, then positive deficit envelope gap plus0.05 times total local deficit.',
        adopt='Only select for complete-lap validation after all34 physical contexts pass. No full time or dominance claim.',
        stop='First failed context stops each proposal; all checks end before590s; ppo_updates=0.'))
    def factory(ref):
        env=environment_factory(ref)
        if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
        return env
    local.environment_factory=factory;start=time.perf_counter();trials=[]
    def run(spec,label,subset):
        records=[]
        for c in subset:
            if time.perf_counter()>start+580:break
            r=local.rollout(metadata,c,spec,stage/(label+'_'+c['corner']['id']+'.npz'),start+590,BendConditionedPolicy);records.append(r)
            if not r['context_pass'] or not r['complete_window']:break
        passed=len(records)==len(subset) and all(r['context_pass'] and r['complete_window'] for r in records)
        gap=sum(max(r['speed_deficit_mm']-targets[r['corner']],0) for r in records) if passed else None
        total=sum(r['total_local_speed_deficit_mm'] for r in records) if passed else None
        cost=gap+.05*total if passed else None
        rec=dict(label=label,passed=passed,spec=spec,fixed_d=selection['fixed_d'],cost=cost,positive_gap_sum_mm=gap,records=records)
        trials.append(rec);write_json(stage/'results.json',trials)
        print('BEND_SATURATION',args.arm,label,passed,cost,gap,'wall',time.perf_counter()-start,flush=True)
        return rec
    for low in [.85,.90,.93,.945]:
        spec=copy.deepcopy(selection['bend_spec']);spec['cruise_base']['low']=low
        run(spec,'low'+str(low),training)
    eligible=sorted([r for r in trials if r['passed']],key=lambda r:r['cost']);checked=[]
    for candidate in eligible:
        r=run(candidate['spec'],'all34_low'+str(candidate['spec']['cruise_base']['low']),contexts)
        checked.append(dict(seed=selection['seed'],result=r));write_json(stage/'all34_checks.json',checked)
        if r['passed']:
            selected=dict(selection,bend_spec=r['spec'],goal_complete=False,full_lap_not_tested=True)
            write_json(stage/'selected_policy.json',selected);break
    write_json(stage/'receipt.json',dict(goal_complete=False,wall_s=time.perf_counter()-start,ppo_updates=0,all34_passed=any(r['result']['passed'] for r in checked)))

if __name__=='__main__':main()
