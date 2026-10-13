"""Check existing retained coefficients from current causal full-route contexts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,json,pickle,shutil,sys,time
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import probe_all34_feed_boundary as local
from turn_aware_projector import environment_factory
from bend_conditioned_policy import BendConditionedPolicy
from fit_bend_transit import crossing_time

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('fit');ap.add_argument('parent');ap.add_argument('fixed')
    ap.add_argument('--budget',type=float,default=600)
    args=ap.parse_args();fit=Path(args.fit);parent=Path(args.parent);fixed=Path(args.fixed)
    start=time.perf_counter();deadline=start+args.budget-15
    selection=json.loads((parent/'selected_policy.json').read_text(encoding='utf-8'))
    with (parent/'causal_contexts.pkl').open('rb') as f:contexts=pickle.load(f)
    assert len(contexts)==34
    retained=json.loads((fit/'retained.json').read_text(encoding='utf-8'))
    retained=[r for r in retained if r['spec']!=selection['bend_spec']]
    maps=[]
    targets_files=[fixed/'full_metrics.json',fixed/'half_ms/corner_metrics.json',BASE/'paper_evidence_review/natural_closed_case/s2_paper_metrics.json']
    for p in targets_files:
        maps.append({r['id']:r['speed_deficit_mm'] for r in json.loads(p.read_text(encoding='utf-8'))['corners']})
    targets={k:min(m[k] for m in maps) for k in maps[0]}
    torch.set_num_threads(1)
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    def factory(reference):
        env=environment_factory(reference)
        if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
        return env
    local.environment_factory=factory
    for candidate in retained:
        if time.perf_counter()>deadline-60:break
        stage=fit.parent/('joint_transit_retained_seed'+str(candidate['seed']))
        stage.mkdir(exist_ok=False);(stage/'source').mkdir()
        files=[Path(__file__),OUT/'probe_all34_feed_boundary.py',OUT/'fit_bend_transit.py',OUT/'bend_conditioned_policy.py',OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py']
        for p in files:shutil.copy2(p,stage/'source'/p.name)
        write_json(stage/'registration.json',dict(kind='retained_parameter_candidate_check',wall_budget_s=args.budget,shared_round_deadline=True,ppo_updates=0,new_optimization_steps=0,
            conditional_search_seed=candidate['seed'],source_label=candidate['selected_label'],parent=str(parent),
            source_sha256={str(p):sha(p) for p in files},retained_sha256=sha(fit/'retained.json'),cache_sha256=sha(parent/'causal_contexts.pkl'),
            adopt='All34 same real states must pass physical and deficit envelope before full-lap validation; no adoption before paired resolution success.',
            stop='Stop candidate on failed context, envelope violation or deadline; preserve all results.'))
        records=[];label='retained_'+candidate['selected_label'];candidate_start=time.perf_counter()
        for c in contexts:
            if time.perf_counter()>deadline-5:break
            file=stage/(label+'_'+c['corner']['id']+'.npz')
            rec=local.rollout(metadata,c,candidate['spec'],file,deadline,BendConditionedPolicy)
            rec['crossing_time_s']=crossing_time(file,c) if rec['context_pass'] else None
            rec['positive_deficit_gap_mm']=max(rec['speed_deficit_mm']-targets[c['corner']['id']]-1e-7,0.) if rec['complete_window'] else None
            records.append(rec)
            if not rec['context_pass'] or not rec['complete_window'] or rec['positive_deficit_gap_mm']>1e-12:break
        passed=len(records)==34 and all(r['context_pass'] and r['complete_window'] for r in records)
        dominance=passed and all(r['positive_deficit_gap_mm']<=1e-12 for r in records)
        result=dict(label=label,spec=candidate['spec'],records=records,passed=passed,dominance=dominance,
            transit_time_s=sum(r['crossing_time_s'] for r in records) if passed else None)
        write_json(stage/'all34_checks.json',[dict(seed=selection['seed'],conditional_search_seed=candidate['seed'],result=result)])
        if dominance:
            write_json(stage/'selected_policy.json',dict(selection,bend_spec=candidate['spec'],parent=str(parent),goal_complete=False,
                full_lap_not_tested=True,conditional_search_seed=candidate['seed'],conditional_search_label=candidate['selected_label'],
                selection_method='retained_coefficients_all34_check_from_current_causal_states'))
        write_json(stage/'receipt.json',dict(goal_complete=False,wall_s=time.perf_counter()-candidate_start,round_elapsed_s=time.perf_counter()-start,ppo_updates=0,
            new_optimization_steps=0,all34_checked=passed,dominance=dominance,selected=dominance))
        print('RETAINED_CHECK',candidate['seed'],dominance,len(records),result['transit_time_s'],time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
