"""Two causal channel interventions on a verified candidate; no retraining claim."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,shutil,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
from candidate_policy_factory import candidate_policy
from turn_aware_projector import environment_factory
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit
from natural_case_metrics import metrics

class ChannelIntervention:
    def __init__(self,base,mode):self.base=base;self.mode=mode
    def bind(self,env):return self.base.bind(env)
    def act(self,x,deterministic=True):
        z=self.base.act(x,deterministic)[0].astype(float)
        if self.mode=='zero_steering':z[0]=0.
        elif self.mode=='maximum_feed':z[1]=20.
        else:raise ValueError(self.mode)
        return z,0.,0.

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');args=ap.parse_args();parent=Path(args.candidate)
    assert json.loads((parent/'full_result.json').read_text(encoding='utf-8'))['success']
    selection=json.loads((parent/'selected_policy.json').read_text(encoding='utf-8'));assert selection['fixed_d'] is None
    assert sha(Path(selection['model']))==selection['model_sha256'];torch.set_num_threads(1)
    stage=parent/'channel_interventions';stage.mkdir(exist_ok=False)
    sources=[Path(__file__),OUT/'candidate_policy_factory.py',OUT/'bend_conditioned_policy.py',OUT/'coordinated_cruise_policy.py',OUT/'turn_aware_projector.py']
    (stage/'source').mkdir()
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='two_full_task_causal_channel_interventions',wall_budget_s=1800,
        per_intervention_execution_deadline_s=850,ppo_updates=0,retrained_ablation=False,
        modes=['zero_steering','maximum_feed'],fixed_preview_comparator='Separately conditionally optimized fixed-D policy, not counted as a third channel intervention here.',
        source_sha256={str(p):sha(p) for p in sources},selection_sha256=sha(parent/'selected_policy.json'),
        scope='Same complete reference and physical task. Alter only the named output after the policy call; observations and memory evolve causally.',
        stop='Physical failure stops each probe. Budget exhaustion is inconclusive, not a successful or physically failed full run. No complete time for any incomplete prefix.'))
    case=BASE/'paper_evidence_review/natural_closed_case';ref=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1);g=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False);start=time.perf_counter();results=[]
    for mode in ['zero_steering','maximum_feed']:
        folder=stage/mode;folder.mkdir();policy=ChannelIntervention(candidate_policy(metadata,selection),mode)
        env=environment_factory(ref);result=evaluate(policy,env,folder/'trace.npz',min(start+1760,time.perf_counter()+850))
        result.update(ppo_updates=0,intervention=mode,retrained_ablation=False,goal_complete=False)
        rows=np.load(folder/'trace.npz')['rows']
        if len(rows):result['analytic']=analytic_command_audit(ref[0],rows,g)
        if result['success']:
            m=metrics(ref,rows,g);write_json(folder/'metrics.json',m)
            result['success']=bool(result['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'])
            if not result['success']:result['complete_time_s']=None
        write_json(folder/'result.json',result);write_json(folder/'ordered_gate_events.json',env.tracker.events)
        results.append(dict(mode=mode,success=result['success'],reason=result['reason'],error=result['error'],
            complete_time_s=result['complete_time_s'],prefix_or_complete_s=result['elapsed_prefix_or_complete_s'],gates=result['geometry']['gates_passed']))
        print('CHANNEL_INTERVENTION',results[-1],flush=True)
    write_json(stage/'receipt.json',dict(goal_complete=False,wall_s=time.perf_counter()-start,ppo_updates=0,results=results))

if __name__=='__main__':main()
