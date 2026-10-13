"""Audit same-checkpoint joint requirements using saved full evaluations only."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,BASE,sha,write


def audit(stage):
    protocol_path=ROOT/'joint_lowdim_acceptance_20260922.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    sources={str(protocol_path):sha(protocol_path)}
    comparators={}
    for name,entry in protocol['original']['comparators'].items():
        path=Path(entry['path']);assert sha(path)==entry['sha256']
        comparators[name]=json.loads(path.read_text(encoding='utf-8'));sources[str(path)]=sha(path)
    evaluations_path=stage/'evaluations.json'
    if not evaluations_path.exists():return None
    records=json.loads(evaluations_path.read_text(encoding='utf-8'));sources[str(evaluations_path)]=sha(evaluations_path)
    results=[]
    for record in records:
        original=record['results']['original'];dense=record['results']['dense']
        row=dict(update=record['update'],checkpoint=record['checkpoint'],
            checkpoint_sha256=sha(record['checkpoint']),original_success=original['success'],dense_success=dense['success'],
            original_complete_time_s=original['complete_time_s'],dense_complete_time_s=dense['complete_time_s'],
            same_checkpoint=original['checkpoint_sha256']==dense['checkpoint_sha256'],
            observation_dimension=original['observation_dimension'],policy_memory_scalars=original['policy_memory_scalars'],
            original34_comparisons=None,nominal_necessary_conditions_passed=False)
        if original['success']:
            path=stage/f"eval_{record['update']:04d}_original_metrics.json"
            metric=json.loads(path.read_text(encoding='utf-8'));sources[str(path)]=sha(path)
            assert metric['all34_windows_complete']
            corners={c['id']:c for c in metric['corners']}
            row['original34_comparisons']={}
            joint=np.ones(34,dtype=bool)
            for name,b in comparators.items():
                old={c['id']:c for c in b['corners']};assert set(corners)==set(old)
                delta=np.array([corners[k]['speed_deficit_mm']-old[k]['speed_deficit_mm'] for k in corners])
                joint &= delta<=1e-7
                row['original34_comparisons'][name]=dict(corners_not_worse=int((delta<=1e-7).sum()),
                    maximum_deficit_increase_mm=float(delta.max()),total_deficit_change_mm=float(delta.sum()),
                    complete_time_gain_s=b['complete_recorded_time_s']-original['complete_time_s'],
                    corner_changes=[dict(id=k,deficit_change_mm=float(v)) for k,v in zip(corners,delta)])
            row['original34_joint_not_worse']=int(joint.sum())
            row['original_cruise99_s']=metric['cruise']['99']['time_s']
        if dense['success']:
            row['dense_gain_percent_vs_strong_two_action']=100*(1-dense['complete_time_s']/15.863)
            row['dense_gain_percent_vs_S2']=100*(1-dense['complete_time_s']/19.957)
            row['dense5percent_nominal_pass']=dense['complete_time_s']<=15.06985+1e-10
        if original['success'] and dense['success']:
            comparisons=row['original34_comparisons']
            row['nominal_necessary_conditions_passed']=bool(row['same_checkpoint'] and row['observation_dimension']==14
                and row['policy_memory_scalars']==0 and row['original34_joint_not_worse']==34
                and comparisons['historical_main']['total_deficit_change_mm']<-1e-7
                and all(c['complete_time_gain_s']>.1 for c in comparisons.values()) and row['dense5percent_nominal_pass'])
        results.append(row)
    report=dict(protocol=str(protocol_path),source_sha256=sources,candidates=results,goal_complete=False,method_frozen=False,
        remaining_for_final_acceptance='Same-model paired-resolution numerical margin, matched compact two-output training, independent repeats, and actual mechanism evidence must also pass. Nominal screening never freezes the method.')
    write(stage/'joint_acceptance_audit.json',report)
    for row in results:
        small={k:v for k,v in row.items() if k!='original34_comparisons'}
        small['original34_comparisons']={k:{a:b for a,b in v.items() if a!='corner_changes'} for k,v in (row['original34_comparisons'] or {}).items()}
        print(json.dumps(small,ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage');args=parser.parse_args()
    assert Path(args.stage).name==args.stage
    audit(ROOT/args.stage)
