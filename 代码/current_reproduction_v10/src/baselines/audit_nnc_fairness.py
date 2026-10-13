"""Freeze the scope of the old un-retrained protection-removal diagnostic."""
from pathlib import Path
import json
import shutil
import hashlib

ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('D:/Nutstore/1/基于强化学习的运动规划方法')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'baseline_records/nnc_fairness_v3';out.mkdir(parents=True,exist_ok=True)
    names=('nnc_planar_environment.py','baseline_policy.yaml','nnc_geometry_index.py','nnc_scan_adapter.py','best_model.pth')
    scopes=[PROJECT/'代码',PROJECT/'论文用到的数据和模型',ROOT/'runtime',Path('D:/PPO_article_work/论文项目')]
    matches=[]
    for folder in scopes:
        for p in folder.rglob('*'):
            if p.is_file() and p.name in names:
                matches.append(dict(path=str(p),sha256=sha(p)))
    index=json.loads((PROJECT/'论文用到的数据和模型/来源与整理路径.json').read_text('utf-8'))['records']
    source=PROJECT/'论文用到的数据和模型/核验与配置/71ea403fc3d1_nnc_evidence_binding.json'
    shutil.copy2(source,out/'retained_binding.json')
    binding=json.loads(source.read_text('utf-8'));copies=[]
    for case,row in binding['records'].items():
        for key in ('result','trace'):
            entry=next((r for r in index if r['source']==row[key]),None)
            if not entry:continue
            p=PROJECT/'论文用到的数据和模型'/entry['relative']
            if not p.exists():continue
            if sha(p)!=row[key+'_sha256']:raise ValueError('Old NNC trace binding mismatch')
            target=out/(case+'_'+key+p.suffix);shutil.copy2(p,target)
            copies.append(dict(case=case,role=key,source=str(p),file=target.name,sha256=sha(p)))
    script=ROOT/'runtime/PPO_article_joint_runs/natural_closed_resume/compact_dense_research/check_original_nnc_current_cases.py'
    shutil.copy2(script,out/'retained_replay_script.py')
    facts=dict(classification='Project-specific protection-removal diagnostic; not original published NNC performance',
        retraining_updates=0,source_replay_sha256=sha(script),searched_filenames=list(names),searched_roots=list(map(str,scopes)),matches=matches,
        archived_copies=copies,original_paper_doi='10.1016/j.rcim.2019.101847',
        original_verified_metadata='Li, Bingran; Zhang, Hui; Ye, Peiqing; Wang, Jinsong (2020), Robotics and Computer-Integrated Manufacturing 61, 101847',
        original_primary_url='https://www.sciencedirect.com/science/article/pii/S0736584519300419',
        identified_intervention='The manuscript records removal of angular-kinematic feed caps and turn prebraking; replay disabled KCM without retraining.',
        unavailable_recovery_components=['Bound original nnc_planar_environment.py','Matched original baseline_policy.yaml and best_model.pth','Verified original published NNC source or complete independent reproduction'],
        allowed_use='Internal diagnosis of that altered project adapter only. Do not assign a complete runtime from prefixes.',
        disallowed_inference='No superiority or infeasibility conclusion about the original published NNC; no pure KCM causal ablation because other protections changed.')
    (out/'fairness_audit.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('NNC_FAIRNESS_AUDIT',len(matches),'candidate filenames;',len(copies),'bound evidence copies')


if __name__=='__main__':main()
