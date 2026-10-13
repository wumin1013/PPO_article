"""Shared complete evaluator for the explicit14input preview-cue policy."""
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,BASE,CASE,write,sha
from compact_previewcue_runtime import PreviewCueRuntime
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit
from natural_case_metrics import metrics
from analyze_dense_trace import analyze


class CompleteChecks:
    def __init__(self, stage, deadline, runtime_class=PreviewCueRuntime):
        self.stage,self.deadline=Path(stage),deadline
        self.runtime_class=runtime_class
        self.cases={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
        self.references={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in self.cases.items()}
        self.geometry={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in self.cases.items()}
        self.protocol=json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
        self.comparators={}
        for name,item in self.protocol['original']['comparators'].items():
            assert sha(item['path'])==item['sha256']
            self.comparators[name]=json.loads(Path(item['path']).read_text(encoding='utf-8'))

    def run(self, policy, checkpoint, step, *, new_ppo_updates=0):
        results={}
        for case in ['dense','original']:
            label=f'eval_{step:04d}_{case}'
            trace=self.stage/(label+'.npz')
            env=self.runtime_class(self.references[case])
            result=evaluate(policy,env,trace,self.deadline)
            with np.load(trace) as data:
                rows=data['rows']
            if len(rows):
                result['analytic']=analytic_command_audit(self.references[case][0],rows,self.geometry[case])
                if not result['analytic']['analytic_corridor_pass']:
                    result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
            result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,
                inherited_controller_at_execution=False,ppo_updates_this_stage=new_ppo_updates,
                observation_features=env.feature_names(),goal_complete=False,method_frozen=False)
            write(trace.with_suffix('.json'),result)
            write(self.stage/(label+'_gates.json'),env.tracker.events)
            if result['success']:
                metric=metrics(self.references[case],rows,self.geometry[case]) if case=='original' else analyze(trace,trace.with_suffix('.json'))
                write(self.stage/(label+'_metrics.json'),metric)
                if case=='original':
                    result['corner_comparisons']={}
                    for name,old in self.comparators.items():
                        delta=[a['speed_deficit_mm']-b['speed_deficit_mm'] for a,b in zip(metric['corners'],old['corners'])]
                        result['corner_comparisons'][name]=dict(not_worse=sum(d<=1e-7 for d in delta),
                            total_change_mm=sum(delta),maximum_increase_mm=max(delta))
                    write(trace.with_suffix('.json'),result)
            results[case]=result
            print('FULL_CHECK',step,case,result['success'],result['complete_time_s'],result['reason'],flush=True)
        return dict(update=step,checkpoint=str(checkpoint),results=results)
