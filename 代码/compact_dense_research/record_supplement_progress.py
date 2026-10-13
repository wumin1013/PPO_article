"""Record observed stage receipts without promoting partial runs to full repetitions."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from evaluate_historical_dense import ROOT, BASE, write

SUPPLEMENT=ROOT/'experimental_supplement_20260923'
ORDER=['compression','ppo_initial','geometry_fit','saturation_fit','ppo_deficit','preview_fit','ppo_preview','ppo_preview_evaluation_completion','contexts','ppo_guided','farvector_fit','local_feedback_fit','local_features','final_evaluation']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    path=SUPPLEMENT/'progress.json';progress=read(path) if path.exists() else {}
    now=datetime.now(timezone.utc);history=SUPPLEMENT/'status_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
    history.mkdir(parents=True)
    active=[];seeds=[]
    for seed in (923201,923202,923203):
        records=[]
        for name in ORDER:
            stage=SUPPLEMENT/'independent_training'/f'seed_{seed}'/name
            if not (stage/'registration.json').exists():continue
            receipt=stage/'replay_receipt.json'
            record={'stage':name,'directory':str(stage)}
            if receipt.exists():record.update(read(receipt))
            else:
                record['status']='started_without_final_receipt'
                active.append(str(stage))
            result=stage/'result.json'
            if result.exists():
                value=read(result)
                record['observed_result']={k:value[k] for k in ('epochs','offline_iterations','new_decisions','ppo_updates','status','wall_s') if k in value}
            records.append(record)
        seeds.append({'seed':seed,'stages':records})
    progress.update(updated_utc=now.isoformat(),seed_stages=seeds,active_stages=active,
        active_stage_names=sorted({Path(p).name for p in active}),
        active_stage=', '.join(sorted({Path(p).name for p in active})) or None,
        selected_model_frozen=True,full_pipeline_repetitions_complete=progress.get('full_pipeline_repetitions_complete',0),
        status='Supplementary experiments underway; final training and paired evaluation remain to be completed.')
    progress.pop('sessions',None)  # Old tool session IDs are not stage state.
    paper=BASE/'paper_evidence_review/verified_results/lowdim_original_revision/experimental_supplement_review/delivery_manifest.json'
    if paper.exists():
        delivery=read(paper)
        progress.update(working_manuscript_pages=delivery['pages'],working_pdf=delivery['pdf'],
            working_pdf_delivered=True,interim_delivery_manifest=str(paper),
            experimental_tables_in_main_text=delivery.get('experimental_tables_in_main_text',False),
            experimental_result_tables_in_main_text=delivery.get('experimental_result_tables_in_main_text',False),
            training_parameter_tables_in_appendix=delivery.get('training_parameter_tables_in_appendix',False),
            training_budget_table_in_appendix=delivery.get('training_budget_table_in_appendix',False))
    if path.exists():shutil.copy2(path,history/'progress.json')
    write(path,progress)
    statepaths=[BASE/'当前执行状态.json',ROOT.parent/'live_research_summary.json',ROOT.parent/'all34_dominance_research/active_research_checkpoint.json',ROOT/'current_progress.json']
    for i,p in enumerate(statepaths):
        shutil.copy2(p,history/f'state_{i}.json');s=read(p)
        s.update(updated_utc=now.isoformat(),supplement_progress=str(path),supplement_requested_by_user=True,
            active_stages=active,current='User-authorized experimental supplement is active. Existing73candidate0 remains frozen. Consult supplement_progress for actual independent-seed stage receipts. Partial seed stages are not full-pipeline repetitions; interim paper supplement is separate from final acceptance.',
            goal_complete=False,final_manuscript_accepted=False)
        if paper.exists():
            s.update(supplement_review_pdf=delivery['pdf'],supplement_review_pdf_sha256=delivery['pdf_sha256'],
                supplement_review_manifest=str(paper),experimental_results_tables_location='Main experimental section per author request')
        write(p,s)
    print('SUPPLEMENT_PROGRESS',[(s['seed'],[(r['stage'],r['status']) for r in s['stages']]) for s in seeds])


if __name__=='__main__':main()
