"""One bounded full-lap run of an already retained seed or declared action probe."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,sys,json,time,shutil
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit
from natural_case_metrics import metrics
from study_io import sha,write_json
from arc_cruise_policy import build_policy


class ActionProbe:
    def __init__(self,base,mode):self.base=base;self.mode=mode
    def bind(self,env):return self.base.bind(env)
    def act(self,observation,deterministic=True):
        latent=self.base.act(observation,deterministic)[0].copy()
        if self.mode=='zero_steering':latent[0]=0.
        elif self.mode=='maximum_feed':latent[1]=np.arctanh(1.-1e-12)
        return latent,0.,0.


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('--seed-label');
    ap.add_argument('--probe',choices=['zero_steering','maximum_feed']);args=ap.parse_args()
    assert not (args.seed_label and args.probe)
    candidate=Path(args.candidate);selected=json.loads((candidate/'selected_policy.json').read_text(encoding='utf-8'))
    parent_reg=json.loads((candidate/'registration.json').read_text(encoding='utf-8'))
    spec=selected['spec']
    if args.seed_label:spec=next(r['spec'] for r in selected['seed_best'] if r['label']==args.seed_label)
    label=('probe_'+args.probe) if args.probe else ('retained_'+args.seed_label if args.seed_label else 'selected_repeat')
    stage=candidate/label;stage.mkdir(exist_ok=False)
    model=Path(selected['model']);assert sha(model)==selected['model_sha256']
    write_json(stage/'selected_policy.json',dict(selected,spec=spec,selected_from=label,probe=args.probe))
    write_json(stage/'registration.json',dict(kind='unretrained_action_intervention_probe' if args.probe else 'retained_conditional_seed_full_evaluation',
        arm=parent_reg['arm'],spec=spec,seed_label=args.seed_label,probe=args.probe,ppo_updates=0,
        independent_full_pipeline_ppo=False,wall_budget_s=1800,full_deadline_s=1720,
        source_selection_sha256=sha(candidate/'selected_policy.json'),model_sha256=sha(model),
        interpretation='Full success requires unchanged original lap, corridor, vector bounds, complete coverage and physical rest; failure prefixes are not complete times. No parameters are selected in this round.',
        code_sha256={p.name:sha(p) for p in (Path(__file__),OUT/'arc_cruise_policy.py',OUT/'cruise_residual_policy.py',OUT/'turn_feedback_policy.py',BASE/'paper_complete_evaluation.py')}))
    for p in (Path(__file__),OUT/'arc_cruise_policy.py',OUT/'cruise_residual_policy.py',OUT/'turn_feedback_policy.py'):
        shutil.copy2(p,stage/p.name)
    torch.set_num_threads(1);case=BASE/'paper_evidence_review/natural_closed_case'
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1);geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    policy=build_policy(load_credit_policy(torch.load(model,map_location='cpu',weights_only=False)),spec)
    if args.probe:policy=ActionProbe(policy,args.probe)
    env=environment_factory(reference)
    if parent_reg['arm']=='fixed':env.fixed_d=spec['fixed_d']
    start=time.perf_counter();r=evaluate(policy,env,stage/'full_trace.npz',start+1720)
    if r['success']:
        rows=np.load(stage/'full_trace.npz')['rows'];r['analytic']=analytic_command_audit(reference[0],rows,geometry)
        m=metrics(reference,rows,geometry);write_json(stage/'full_metrics.json',m)
        r['success']=bool(r['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and m['projection_raw_backward_steps']==0)
        if not r['success']:r['complete_time_s']=None
    r.update(spec=spec,probe=args.probe,seed_label=args.seed_label);write_json(stage/'full_result.json',r)
    write_json(stage/'receipt.json',dict(status='complete_bounded_evaluation',wall_s=time.perf_counter()-start,
        success=r['success'],complete_time_s=r['complete_time_s'],goal_complete=False,
        ppo_updates=0,result_sha256=sha(stage/'full_result.json')))
    print('RETAINED_COMPLETE',r['success'],r['complete_time_s'],r['reason'],flush=True)


if __name__=='__main__':main()
