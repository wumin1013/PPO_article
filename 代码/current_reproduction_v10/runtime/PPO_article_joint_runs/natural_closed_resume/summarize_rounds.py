"""Evidence ledger from retained registrations and receipts, without status inflation."""
from pathlib import Path
import sys,json,re
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from study_io import write_json,sha


def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def main():
    rounds=[]
    stages=sorted(OUT.glob('*_training'))
    stages=[p for stage in stages for p in (stage,stage/'exact_continuation') if p.is_dir()]
    for path in stages:
        reg=read(path/'registration.json')
        if not reg:continue
        receipt=read(path/'receipt.json');train=read(path/'training.json');full=read(path/'full_result.json')
        trials=train.get('trials',[]) if train else []
        labels=[t.get('label','') for t in trials];seeds=sorted(set(re.findall(r'seed(\d+)',','.join(labels))))
        rounds.append(dict(stage=str(path.relative_to(OUT)),arm=reg['arm'],training_kind=reg['kind'],
            planned_seeds=reg.get('seeds'),observed_search_seeds=seeds,completed_candidate_trial_records=sum('seed' in x for x in labels),
            successful_training_trial_records=sum(bool(t.get('success')) for t in trials),
            wall_s=receipt.get('wall_s',receipt.get('total_wall_s')) if receipt else None,wall_budget_s=reg.get('wall_budget_s'),
            full_success=full.get('success') if full else None,full_time_s=full.get('complete_time_s') if full else None,
            incomplete_prefix_s=full.get('elapsed_prefix_or_complete_s') if full and not full.get('success') else None,
            incomplete_reason=full.get('error') or full.get('reason') if full and not full.get('success') else None,
            status=receipt.get('status') if receipt else 'running_or_no_receipt',ppo_updates=reg.get('ppo_updates',0),
            registration_sha256=sha(path/'registration.json'),receipt_sha256=sha(path/'receipt.json') if receipt else None))
    write_json(OUT/'new_research_round_ledger.json',dict(rounds=rounds,
        interpretation='Conditional parameter searches share one inherited composite backbone; no new PPO updates, independent end-to-end seeds, unseen tests, or method freeze.',
        missing_receipt_is_not_success=True,goal_complete=False))
    for r in rounds:print(r['stage'],r['status'],r['full_success'],r['full_time_s'],r['wall_s'])


if __name__=='__main__':main()
