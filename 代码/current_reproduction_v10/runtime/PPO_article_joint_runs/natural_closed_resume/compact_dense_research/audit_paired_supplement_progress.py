"""Track conditional repeats and paired training without promoting failures."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from evaluate_historical_dense import ROOT,BASE,write,sha
from replay_lowdim_training_stage import SUPPLEMENT,SEEDS


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    now=datetime.now(timezone.utc)
    endpoints=[]
    active=[]
    for seed in SEEDS:
        directory=SUPPLEMENT/'independent_training'/f'seed_{seed}/final_evaluation'
        results=[]
        for case in ['original','dense']:
            for key in ['1ms','half_ms']:
                path=directory/f'eval_0004_{case}_{key}.json'
                if path.exists():
                    value=read(path)
                    results.append(dict(case=case,dt_s=value['dt_s'],success=value['success'],
                        complete_time_s=value['complete_time_s'],prefix_s=value['elapsed_prefix_s'],
                        reason=value['reason'],error=value['error'],source=str(path),source_sha256=sha(path)))
        endpoints.append(dict(seed=seed,registered_evaluations_complete=len(results)==4,
            all_four_success=len(results)==4 and all(v['success'] for v in results),results=results))
    paired=[]
    for seed in SEEDS:
        for arm in ['three','two']:
            for total in [24,48]:
                folder=ROOT/f'supplement_paired_{seed}_{arm}_ppo{total}'
                if not (folder/'registration.json').exists():continue
                record=dict(seed=seed,arm=arm,total_endpoint=total,directory=str(folder))
                if (folder/'receipt.json').exists():record.update(read(folder/'receipt.json'))
                else:
                    record['status']='active_without_final_receipt'
                    active.append(str(folder))
                if (folder/'updates.json').exists():
                    updates=read(folder/'updates.json')
                    record['completed_updates']=len(updates)
                    record['all_auxiliary_weights_zero']=all(u['auxiliary_weight_each']==0. for u in updates)
                    assert record['all_auxiliary_weights_zero']
                paired.append(record)
    report=dict(updated_utc=now.isoformat(),fixed_shared_data_pipeline_endpoints=endpoints,
        conditional_pipeline_repetitions_evaluated=sum(r['registered_evaluations_complete'] for r in endpoints),
        all_four_success_count=sum(r['all_four_success'] for r in endpoints),denominator=len(SEEDS),
        independent_feature_discovery_repetitions=0,
        scope='Independent initialization/optimization conditional on shared historical teacher data and fitted feature maps; final physical failures are retained.',
        paired_training=paired,paired_active_stages=active,
        paired_protocol=str(SUPPLEMENT/'paired_action_ablation/protocol.json'),
        paired_ablation_complete=False,selected_frozen_model_changed=False)
    output=SUPPLEMENT/'paired_and_repeat_progress.json'
    if output.exists():
        history=SUPPLEMENT/'status_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
        history.mkdir(parents=True)
        shutil.copy2(output,history/output.name)
    write(output,report)
    progress_path=SUPPLEMENT/'progress.json'
    progress=read(progress_path)
    progress.update(paired_and_repeat_progress=str(output),
        conditional_pipeline_repetitions_evaluated=report['conditional_pipeline_repetitions_evaluated'],
        conditional_pipeline_all_four_success_count=report['all_four_success_count'],
        paired_active_stages=active,paired_ablation_complete=False)
    write(progress_path,progress)
    for p in [BASE/'当前执行状态.json',ROOT/'current_progress.json',ROOT.parent/'live_research_summary.json',
              ROOT.parent/'all34_dominance_research/active_research_checkpoint.json']:
        value=read(p)
        value.update(updated_utc=now.isoformat(),paired_and_repeat_progress=str(output),
            active_stages=active,goal_complete=False,final_manuscript_accepted=False,
            current='User-authorized paired supplement active. Consult paired_and_repeat_progress. Conditional fixed-data repetition outcomes are retained, including failures; new equal-budget PPO continuations are not final successes. Frozen73 paper model remains unchanged.')
        write(p,value)
    print('CONDITIONAL_REPEAT_EVALUATIONS',report['conditional_pipeline_repetitions_evaluated'],
          'ALL_FOUR_SUCCESS',report['all_four_success_count'],'PAIRED',
          [(r['seed'],r['arm'],r.get('completed_updates',0),r['status']) for r in paired])


if __name__=='__main__':main()
