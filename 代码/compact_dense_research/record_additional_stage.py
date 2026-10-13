"""Append an explicitly checked stage without replacing prior research records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
STATES = [BASE/'当前执行状态.json', OUT/'live_research_summary.json',
          OUT/'all34_dominance_research/active_research_checkpoint.json', ROOT/'current_progress.json']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--session',type=int)
    parser.add_argument('--current',required=True)
    args=parser.parse_args()
    assert Path(args.stage).name == args.stage
    stage=ROOT/args.stage
    record=dict(directory=str(stage))
    for key in ['registration','receipt','result','full_result','comparison']:
        path=stage/f'{key}.json'
        if path.is_file(): record[key]=json.loads(path.read_text(encoding='utf-8'))
    assert 'registration' in record
    active='receipt' not in record
    assert not active or args.session is not None
    record['verified_exec_session']=args.session if active else None
    now=datetime.now(timezone.utc)
    backup=ROOT/'state_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True,exist_ok=False)
    receipts=[]
    for path in STATES:
        raw=path.read_bytes()
        saved=backup/path.name
        saved.write_bytes(raw)
        receipts.append(dict(original=str(path),path=str(saved),sha256=hashlib.sha256(raw).hexdigest()))
        state=json.loads(raw)
        records=state.setdefault('additional_research_stages',{})
        records[args.stage]=record
        stages=[x for x in state.get('active_stages',[]) if x!=str(stage)]
        if active: stages.append(str(stage))
        state.update(updated_utc=now.isoformat(),current=args.current,active_stages=stages,
            goal_complete=False,method_frozen=False,research_acceptance_passed=False,
            final_manuscript_accepted=False,final_pdf=None,
            verified_active_exec_session=args.session if active else None,
            latest_state_backup_receipt=str(backup/'receipt.json'))
        state['verified_active_exec_sessions']={name:value['verified_exec_session']
            for name,value in records.items() if value.get('verified_exec_session') is not None and 'receipt' not in value}
        path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (backup/'receipt.json').write_text(json.dumps(dict(backups=receipts,stage=record,
        active=active,session=args.session if active else None),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(active=active,stage=str(stage),backup=str(backup)),ensure_ascii=False))


if __name__=='__main__': main()
