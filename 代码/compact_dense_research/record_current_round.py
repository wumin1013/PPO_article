"""Append a bounded round pointer while preserving earlier research records."""
import argparse
import json
import shutil
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,write,sha


def record(name,message,session=None,also_active=None):
    stage=ROOT/name
    assert stage.is_dir() and (stage/'registration.json').is_file()
    now=datetime.now(timezone.utc)
    backup=ROOT/'state_history'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+name)
    backup.mkdir(parents=True,exist_ok=False)
    paths=[BASE/'当前执行状态.json',ROOT.parent/'live_research_summary.json',
           ROOT.parent/'all34_dominance_research/active_research_checkpoint.json',ROOT/'current_progress.json']
    receipt=[]
    active=dict(also_active or {})
    if session is not None:active[name]=session
    for i,path in enumerate(paths):
        target=backup/f'{i}_{path.name}';shutil.copy2(path,target)
        data=json.loads(path.read_text(encoding='utf-8'))
        data.update(updated_utc=now.isoformat(),current=message,goal_complete=False,method_frozen=False,
            research_acceptance_passed=False,final_manuscript_accepted=False,
            active_stages=[str(ROOT/k) for k in active],
            verified_active_exec_sessions=active,
            latest_research_stage=str(stage),latest_status_backup=str(target),
            remaining_acceptance=[
                'Same14input memoryless checkpoint:all34corners versus historical main,S2,strongFixed and validated corresponding-capacity two-action baselines;credible original-path gain at both resolutions',
                'At least5percent dense-case gain versus credible same14input independently trained two-output comparison at both resolutions',
                'Independent training repetitions or an explicit reproducibility limit;paired swing evidence',
                'Mechanism review and original18page-based final manuscript only after research gates pass'])
        record_data=dict(directory=str(stage),registration_sha256=sha(stage/'registration.json'),
                         effective_status='running' if session is not None else 'finished')
        for filename in ['result.json','receipt.json']:
            if (stage/filename).is_file():record_data[filename]=dict(path=str(stage/filename),sha256=sha(stage/filename))
        history=data.setdefault('bounded_round_records',[])
        history.append(record_data)
        write(path,data);receipt.append(dict(path=str(path),previous_sha256=sha(target),current_sha256=sha(path)))
    write(backup/'receipt.json',receipt)
    print(json.dumps(dict(stage=str(stage),active_session=session,backups=str(backup),message=message)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('message');p.add_argument('--session',type=int)
    p.add_argument('--also-active',action='append',default=[])
    a=p.parse_args();other={k:int(v) for k,v in (s.rsplit(':',1) for s in a.also_active)}
    record(a.name,a.message,a.session,other)
