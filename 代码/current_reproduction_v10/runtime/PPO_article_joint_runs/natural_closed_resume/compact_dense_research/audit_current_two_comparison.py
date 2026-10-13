"""Read-only comparison against every complete paired supplemental two-output case."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from evaluate_historical_dense import ROOT,write,sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pair(directory,case,outputs,bindings):
    records=[]
    for key in ['1ms','half_ms']:
        found=list(directory.glob(f'eval_*_{case}_{key}.json'))
        if len(found)!=1:return None
        path=found[0]
        record=read(path)
        if not record['success']:return None
        assert (record['observation_dimension'],record['actor_outputs'],record['policy_memory_scalars'])==(14,outputs,0)
        checkpoint=Path(read(directory/'registration.json')['checkpoint'])
        assert record['checkpoint_sha256']==sha(checkpoint)
        assert record['trace_sha256']==sha(path.with_suffix('.npz'))
        for p in [path,path.with_suffix('.npz'),checkpoint]:bindings[str(p)]=sha(p)
        records.append((path,record))
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',type=Path,required=True)
    args=parser.parse_args()
    sup=ROOT/'experimental_supplement_20260923'
    directories=[]
    for seed in [923201,923202,923203]:
        directories += [sup/f'matched_direction_initialization/seed_{seed}/two/evaluation',
            sup/f'coherent_teacher_refinement/seed_{seed}/common_two/evaluation',
            sup/f'common_two_action_expansion/seed_{seed}/two/evaluation',
            sup/f'two_visited_feedback/seed_{seed}/evaluation',
            sup/f'two_accumulated_feedback/seed_{seed}/evaluation']
        directories += [ROOT/f'supplement_common_start_{seed}_two_ppo{n}/evaluation' for n in [24,48]]
    bindings={str(Path(__file__)):sha(__file__)}
    all_pass=True
    records=[]
    dense_times=[]
    for case in ['original','dense']:
        current=pair(args.evaluation,case,3,bindings)
        assert current is not None,'Candidate does not complete both integration intervals'
        times=np.array([r['complete_time_s'] for p,r in current])
        for directory in directories:
            other=pair(directory,case,2,bindings)
            if other is None:continue
            base=np.array([r['complete_time_s'] for p,r in other])
            margin=max(.1,5*(abs(times[1]-times[0])+abs(base[1]-base[0])))
            gains=base-times
            row=dict(case=case,baseline=str(directory),baseline_times_s=base.tolist(),candidate_times_s=times.tolist(),
                time_gains_s=gains.tolist(),numerical_margin_s=float(margin),time_pass=bool(np.all(gains>margin)))
            if case=='original':
                corners=[]
                for (cp,cr),(bp,br) in zip(current,other):
                    cm=cp.with_name(cp.stem+'_metrics.json');bm=bp.with_name(bp.stem+'_metrics.json')
                    for path in [cm,bm]:bindings[str(path)]=sha(path)
                    c,b=read(cm),read(bm)
                    assert c['all34_windows_complete'] and b['all34_windows_complete']
                    old={r['id']:r['speed_deficit_mm'] for r in b['corners']}
                    diff={r['id']:r['speed_deficit_mm']-old[r['id']] for r in c['corners']}
                    assert len(diff)==34
                    corners.append(dict(not_worse=sum(v<=1e-7 for v in diff.values()),losing={k:v for k,v in diff.items() if v>1e-7}))
                row['corners']=corners
                row['corners_pass']=all(r['not_worse']==34 for r in corners)
                all_pass=all_pass and row['corners_pass'] and row['time_pass']
            else:
                row['gains_percent']=(100*(1-times/base)).tolist()
                row['five_percent_pass']=bool(np.all(100*(1-times/base)>=5.))
                all_pass=all_pass and row['time_pass'] and row['five_percent_pass']
                dense_times.append(base)
            records.append(row)
    assert dense_times and any(r['case']=='original' for r in records)
    envelope=np.min(dense_times,axis=0)
    candidate=np.array([r['complete_time_s'] for p,r in pair(args.evaluation,'dense',3,bindings)])
    envelope_margin=max(.1,5*(abs(candidate[1]-candidate[0])+abs(envelope[1]-envelope[0])))
    envelope_pass=bool(np.all(envelope-candidate>envelope_margin) and np.all(100*(1-candidate/envelope)>=5.))
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate_evaluation=str(args.evaluation),source_sha256=bindings,records=records,
        baseline_pair_count=len(records),dense_envelope_s=envelope.tolist(),dense_envelope_gain_percent=(100*(1-candidate/envelope)).tolist(),
        dense_envelope_margin_s=float(envelope_margin),all_new_two_action_comparisons_pass=bool(all_pass and envelope_pass),
        scope='Per-case complete paired executions retained even if another case failed. This compares developed controllers; it does not establish equal training budgets or independent complete-pipeline repetitions.')
    output=args.evaluation/'current_two_action_comparison.json'
    assert not output.exists(),'Preserve existing audit; use an explicitly versioned follow-up when baseline evidence changes'
    write(output,report)
    print('CURRENT_TWO_ACTION_AUDIT',all_pass and envelope_pass,'pairs',len(records),'envelope',envelope.tolist(),'gains',report['dense_envelope_gain_percent'],flush=True)


if __name__=='__main__':main()
