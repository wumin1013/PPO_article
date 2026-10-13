"""Apply the registered endpoint rule to all three fixed seeds, retaining failures."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from evaluate_historical_dense import ROOT, sha, write
from audit_current_two_comparison import read


def main():
    sup=ROOT/'experimental_supplement_20260923'
    out=sup/'common_two_action_expansion/completed_endpoint_selection.json'
    assert not out.exists()
    protocol=sup/'common_two_action_expansion/protocol.json'
    gate=sup/'paired_seed2_initial_recovery/seed_923202/attempt_2/initial_execution_equivalence.json'
    assert read(gate)['initial_pair_gate_pass']
    bindings={str(p):sha(p) for p in [Path(__file__),protocol,gate]}
    denominators=np.array([445.185,445.185,19.957,19.9565])
    endpoint_records=[]; selected=[]; budgets=[]
    for seed in [923201,923202,923203]:
        for arm in ['two','three']:
            initial=(sup/f'paired_seed2_initial_recovery/seed_{seed}/attempt_2/{arm}/evaluation' if seed==923202 else
                sup/f'common_two_action_expansion/seed_{seed}/{arm}/evaluation')
            candidates=[]
            for endpoint in [0,24,48]:
                directory=initial if endpoint==0 else ROOT/f'supplement_common_start_{seed}_{arm}_ppo{endpoint}/evaluation'
                receipt=read(directory/'replay_receipt.json')
                assert receipt['all_four_evaluations_recorded'] and receipt['source_unchanged']
                registration=directory/'registration.json'
                checkpoint=Path(read(registration)['checkpoint'])
                for p in [directory/'replay_receipt.json',registration,checkpoint]:bindings[str(p)]=sha(p)
                times=[];success=[];reasons=[]
                for case in ['original','dense']:
                    for key in ['1ms','half_ms']:
                        paths=list(directory.glob(f'eval_*_{case}_{key}.json'));assert len(paths)==1
                        path=paths[0];record=read(path)
                        assert record['checkpoint_sha256']==sha(checkpoint) and record['trace_sha256']==sha(path.with_suffix('.npz'))
                        for p in [path,path.with_suffix('.npz')]:bindings[str(p)]=sha(p)
                        success.append(record['success']);times.append(record['complete_time_s']);reasons.append(record['reason'])
                row=dict(seed=seed,arm=arm,cumulative_updates=endpoint,success=success,times_s=times,
                    reasons=reasons,directory=str(directory),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
                endpoint_records.append(row)
                if all(success):
                    norm=np.array(times)/denominators
                    candidates.append((float(norm.max()),float(norm.mean()),endpoint,row))
                if endpoint:
                    stage=directory.parent
                    updates=read(stage/'updates.json');training=read(stage/'receipt.json')
                    assert len(updates)==24 and training.get('error') is None
                    registered=read(stage/'registration.json')
                    assert all(sha(p)==h for p,h in registered['source_sha256'].items())
                    for p in [stage/'registration.json',stage/'receipt.json',stage/'updates.json',stage/'paired_protocol_binding.json']:
                        bindings[str(p)]=sha(p)
                    budgets.append(dict(seed=seed,arm=arm,cumulative_endpoint=endpoint,updates_this_stage=24,decisions_this_stage=24*2048,wall_s=training['wall_s']))
            assert candidates,'No complete endpoint; do not silently replace the seed'
            selected.append(min(candidates,key=lambda r:r[:3])[3])
    comparisons=[]
    for seed in [923201,923202,923203]:
        two=next(r for r in selected if r['seed']==seed and r['arm']=='two')
        three=next(r for r in selected if r['seed']==seed and r['arm']=='three')
        a,b=np.array(two['times_s']),np.array(three['times_s'])
        margins=[max(.1,5*(abs(a[i+1]-a[i])+abs(b[i+1]-b[i]))) for i in [0,2]]
        comparisons.append(dict(seed=seed,two_endpoint=two['cumulative_updates'],three_endpoint=three['cumulative_updates'],
            two_times_s=a.tolist(),three_times_s=b.tolist(),gain_seconds=(a-b).tolist(),gain_percent=(100*(1-b/a)).tolist(),
            numerical_margin_s=margins,credible_time_gain=[bool(np.all((a-b)[i:i+2]>margins[i//2])) for i in [0,2]],
            dense_five_percent=bool(np.all(100*(1-b[2:]/a[2:])>=5.))))
    write(out,dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=bindings,
        S2_denominators_s=denominators.tolist(),selection_rule=read(protocol)['selection'],
        endpoint_records=endpoint_records,selected=selected,paired_comparisons=comparisons,budgets=budgets,
        all_three_seeds_complete_registered_paired_opportunities=True,updates_each_arm=48,decisions_each_arm=98304,
        uniform_full_training_histories=False,
        scope='Shared-start paired PPO refinement after seed-dependent initialization. Retain0/24/48failures and every stronger valid developed baseline. Do not equate this intervention with independent full-pipeline repetitions.',
        all_pairs_show_credible_dense_gain=all(r['credible_time_gain'][1] for r in comparisons),
        all_pairs_show_dense_five_percent=all(r['dense_five_percent'] for r in comparisons)))
    print('PAIRED_SELECTION',comparisons,flush=True)


if __name__=='__main__':main()
