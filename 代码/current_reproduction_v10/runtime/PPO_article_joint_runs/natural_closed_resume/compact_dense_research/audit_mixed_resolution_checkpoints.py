"""Read-only paired-resolution screening; never promotes a failed prefix."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,write,sha
from train_mixed_resolution_ppo import HALF_METRICS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(name,extra_comparators=None):
    stage=ROOT/name;protocol_path=ROOT/'joint_lowdim_acceptance_20260922.json';protocol=read(protocol_path)
    names=['historical_main','historical_strong_fixed','S2']
    original={};sources={str(protocol_path):sha(protocol_path)}
    matched_path=ROOT/'matched14_two_action_evidence_20260923.json'
    matched=read(matched_path) if matched_path.exists() else None
    dense_two_action=None
    if matched:
        sources[str(matched_path)]=sha(matched_path)
        pairs=[]
        for item in matched['records']:
            assert sha(item['checkpoint'])==item['checkpoint_sha256']
            results=[]
            for binding in item['cases']['dense']:
                path=Path(binding['result']);assert sha(path)==binding['result_sha256']
                result=read(path)
                assert result['checkpoint_sha256']==item['checkpoint_sha256']
                assert result['trace_sha256']==binding['trace_sha256']==sha(path.with_suffix('.npz'))
                assert (result['observation_dimension'],result['actor_outputs'],result['policy_memory_scalars'])==(14,2,0)
                results.append(result);sources[str(path)]=sha(path)
            if len(results)==2 and all(r['success'] for r in results):
                pairs.append([r['complete_time_s'] for r in results])
        assert pairs
        dense_two_action=np.min(pairs,axis=0).tolist()
    for key,path in zip(names,HALF_METRICS):
        entry=protocol['original']['comparators'][key]
        assert sha(entry['path'])==entry['sha256']
        first=read(entry['path']);second=read(path)
        second=second.get('metrics',second)
        result_path=path.parent/'full_result.json' if key!='S2' else ROOT.parent/'s2_resolution_half_ms/result.json'
        result=read(result_path);assert result['success']
        original[key]=dict(metrics=[first,second],times=[first['complete_recorded_time_s'],result['complete_time_s']])
        for p in [Path(entry['path']),path,result_path]:sources[str(p)]=sha(p)
    extra_names=[]
    if extra_comparators:
        extra=read(extra_comparators);sources[str(extra_comparators)]=sha(extra_comparators)
        for path,digest in extra['source_sha256'].items():
            assert sha(path)==digest,(path,'changed registered comparator')
            sources[path]=digest
        assert len(extra['1ms'])==len(extra['half_ms'])>0
        for paths in zip(extra['1ms'],extra['half_ms']):
            metrics=[];times=[];checkpoint_hashes=[]
            for key,path in zip(['1ms','half_ms'],paths):
                path=Path(path);m=read(path)
                result_path=path.with_name(path.name.replace('_metrics.json','.json'));result=read(result_path)
                assert result['success'] and m['all34_windows_complete'] and len(m['corners'])==34
                assert (result['observation_dimension'],result['actor_outputs'],result['policy_memory_scalars'])==(14,2,0)
                assert result['checkpoint_sha256']==sha(result['checkpoint'])
                assert result['trace_sha256']==sha(result_path.with_suffix('.npz'))
                assert result['dt_s']==(.001 if key=='1ms' else .0005)
                metrics.append(m);times.append(result['complete_time_s']);checkpoint_hashes.append(result['checkpoint_sha256'])
            assert checkpoint_hashes[0]==checkpoint_hashes[1]
            label='capacity_two_action_'+Path(paths[0]).name.split('_')[1]
            assert label not in original
            original[label]=dict(metrics=metrics,times=times);extra_names.append(label)
    records=[]
    evaluations=read(stage/'evaluations.json') if (stage/'evaluations.json').exists() else []
    for evaluation in evaluations:
        count=evaluation['update'];checkpoint=stage/f'checkpoint_{count:04d}.pt'
        record=dict(update=count,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),physical_four_success=False)
        data={}
        for case in ['dense','original']:
            pair=[]
            for key in ['1ms','half_ms']:
                path=stage/f'eval_{count:04d}_{case}_{key}.json'
                if not path.exists():break
                item=read(path);assert item['checkpoint_sha256']==sha(checkpoint)
                pair.append(item);sources[str(path)]=sha(path)
            data[case]=pair
        record['dense_success']=[r['success'] for r in data['dense']]
        record['dense_times_s']=[r['complete_time_s'] for r in data['dense']]
        record['original_success']=[r['success'] for r in data['original']]
        record['original_times_s']=[r['complete_time_s'] for r in data['original']]
        record['physical_four_success']=all(len(data[k])==2 and all(r['success'] for r in data[k]) for k in data)
        if len(data['original'])==2 and all(r['success'] for r in data['original']):
            candidate=[]
            for key in ['1ms','half_ms']:
                path=stage/f'eval_{count:04d}_original_{key}_metrics.json';sources[str(path)]=sha(path)
                m=read(path);assert m['all34_windows_complete'];candidate.append(m)
            record['original_comparisons']={}
            for key,baseline in original.items():
                times=record['original_times_s'];base_times=baseline['times']
                margin=max(.1,5.*(abs(times[1]-times[0])+abs(base_times[1]-base_times[0])))
                comparisons=[]
                for current,other in zip(candidate,baseline['metrics']):
                    old={r['id']:r['speed_deficit_mm'] for r in other['corners']}
                    deltas={r['id']:r['speed_deficit_mm']-old[r['id']] for r in current['corners']}
                    assert len(deltas)==34
                    comparisons.append(dict(not_worse=sum(v<=1e-7 for v in deltas.values()),
                        total_deficit_change_mm=sum(deltas.values()),losing={k:v for k,v in deltas.items() if v>1e-7}))
                gains=[b-a for a,b in zip(times,base_times)]
                record['original_comparisons'][key]=dict(corners=comparisons,time_gains_s=gains,
                    numerical_margin_s=margin,time_pass=all(g>margin for g in gains),
                    corners_pass=all(c['not_worse']==34 for c in comparisons))
        if len(data['dense'])==2 and all(r['success'] for r in data['dense']):
            times=record['dense_times_s']
            record['dense_vs_historical_two_action_percent']=[100.*(1.-t/b) for t,b in zip(times,protocol['dense']['strong_two_action_times_s'])]
            record['dense_vs_S2_percent']=[100.*(1.-t/b) for t,b in zip(times,protocol['dense']['S2_times_s'])]
            if dense_two_action:
                record['dense_vs_matched14_two_action_percent']=[100.*(1.-t/b) for t,b in zip(times,dense_two_action)]
                record['dense_five_percent_time_gate']=bool(all(v>=5. for v in record['dense_vs_matched14_two_action_percent']+record['dense_vs_S2_percent']))
        record['numerical_performance_gates_pass']=bool(record['physical_four_success']
            and all(key in record.get('original_comparisons',{}) for key in names)
            and all(record['original_comparisons'][key]['corners_pass'] and record['original_comparisons'][key]['time_pass'] for key in names)
            and record.get('dense_five_percent_time_gate',False))
        if extra_names:
            record['extended_two_action_all34_pass']=bool(all(key in record.get('original_comparisons',{})
                and record['original_comparisons'][key]['corners_pass'] for key in extra_names))
            record['extended_numerical_performance_gates_pass']=bool(record['numerical_performance_gates_pass']
                and record['extended_two_action_all34_pass']
                and all(record['original_comparisons'][key]['time_pass'] for key in extra_names))
        records.append(record)
    report=dict(stage=str(stage),records=records,source_sha256=sources,
        matched14_two_action_times_s=dense_two_action,
        same14input_two_action_comparator_still_required=matched is None,
        corresponding_capacity_full_comparison_available=bool(extra_names),
        capacity_and_training_opportunity_match_still_required=not bool(extra_names),
        independent_training_budget_equality_established=False,
        comparison_scope='Fastest valid paired dense results retained even if another case failed;14inputs alone do not establish a strict architectural ablation.',
        independent_repeats_and_mechanism_still_required=True,
        goal_complete=False,method_frozen=False)
    write(stage/('paired_resolution_extended_audit.json' if extra_comparators else 'paired_resolution_audit.json'),report)
    for r in records:
        print(json.dumps({k:v for k,v in r.items() if k not in ['checkpoint','checkpoint_sha256']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--extra-comparators',type=Path)
    a=p.parse_args();audit(a.name,a.extra_comparators)
