"""Read existing evidence, preserve prior status, and reopen the unmet research goal."""
from pathlib import Path
import sys, json, hashlib, shutil, subprocess, re
from datetime import datetime, timezone
import numpy as np

BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from natural_case_metrics import ordered_projection

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    case=BASE/'paper_evidence_review/natural_closed_case'
    binding=read(case/'paper_result_binding.json'); geometry=read(case/'geometry.json')
    results={}; checks=[]
    corners=np.array([c['reference_s_mm'] for c in geometry['sharp_corners']])
    for name, source in binding['primary_methods'].items():
        for kind in ('trace','result','metrics'):
            p=BASE/source[kind]; actual=digest(p)
            checks.append(dict(method=name,kind=kind,path=str(p),sha256=actual,matches=actual==source[kind+'_sha256']))
        z=np.load(BASE/source['trace']); rows=z['rows']; xy=rows[:,:2]
        speed=np.linalg.norm(rows[:,2:4],axis=1); s=ordered_projection(z['reference'],xy)
        length=np.linalg.norm(np.diff(z['reference'],axis=0),axis=1).sum()
        near=np.min(np.abs(s[:,None]-corners),axis=1)<=30.
        def summary(mask):
            v=speed[mask]
            return dict(time_s=float(mask.sum()*.001),cruise99_s=float((v>=99).sum()*.001),
                        mean_speed=float(v.mean()) if len(v) else None,
                        deficit_equivalent_s=float(np.maximum(100-v,0).sum()*.001/100),
                        travelled_mm=float(v.sum()*.001))
        result=dict(record=read(BASE/source['result']),all=summary(np.ones(len(rows),bool)),
                    near30=summary(near),far30=summary(~near),first_half=summary(s<length/2),
                    second_half=summary(s>=length/2),arrays={k:list(z[k].shape) for k in z.files},
                    quarters=[summary((s>=length*i/4)&(s<length*(i+1)/4)) for i in range(4)])
        result['sections']=read(BASE/source['metrics'])['sections']
        if 'observations' in z and len(z['observations']):
            obs=z['observations']; actions=z['actions']; t=np.arange(len(obs))*20
            t=np.clip(t,0,len(rows)-1); dnear=near[t]; far=~dnear
            result['action_summary']={str(i):dict(min=float(actions[:,i].min()),median=float(np.median(actions[:,i])),max=float(actions[:,i].max())) for i in range(3)}
            result['far_feed_quantiles']=np.quantile(actions[far,1],[0,.1,.5,.9,1]).tolist()
            result['far_feed_below99_fraction']=float((actions[far,1]<99).mean())
            result['far_speed_minus_request_quantiles']=np.quantile(speed[t[far]]-actions[far,1],[0,.1,.5,.9,1]).tolist()
            result['obs_nonlocal_quantiles']={str(i):np.quantile(obs[:,i],[0,.1,.5,.9,1]).tolist() for i in (15,16,17,19)}
            result['far_feed_per_quarter']=[float(actions[far&(s[t]>=length*i/4)&(s[t]<length*(i+1)/4),1].mean()) for i in range(4)]
        results[name]=result
    assert all(c['matches'] for c in checks)
    source_root=Path(r'D:\Nutstore\1\基于强化学习的运动规划方法\论文项目\submission_package_EAAI')
    pdf=source_root/'01_manuscript_without_author_details/manuscript_without_author_details.pdf'
    tex=source_root/'06_latex_source_files/latex_source_anonymous/manuscript.tex'
    baseline=dict(pdf=str(pdf),pdf_sha256=digest(pdf),tex=str(tex),tex_sha256=digest(tex),
                  prior_verification=str(Path(r'D:\PPO_article_recovery\historical_package_restore_20260921\final_verification.json')))
    assert baseline['pdf_sha256']=='074724d783318f227720ce09fe07410eb67221c572d661962c4ff2d7db64caa7'
    info=subprocess.check_output(['pdfinfo',str(pdf)]).decode('utf-8',errors='replace')
    baseline['pages']=int(re.search(r'Pages:\s+(\d+)',info).group(1)); assert baseline['pages']==18
    subprocess.run(['pdftotext','-enc','UTF-8',str(pdf),str(OUT/'baseline_full_text.txt')],check=True)
    write(OUT/'saved_evidence_audit.json',dict(baseline=baseline,checks=checks,methods=results))
    archive=OUT/'status_before_reopen'; archive.mkdir(exist_ok=True)
    statefile=BASE/'当前执行状态.json'; continuation=BASE/'PPO_article_论文交付与续接.md'
    freeze=case/'method_freeze.json'; resume=case/'paper_delivery_continuation.json'
    for p in (statefile,continuation,freeze,resume):
        target=archive/p.name
        if not target.exists():shutil.copy2(p,target)
    now=datetime.now(timezone.utc).isoformat()
    correction=dict(updated_utc=now,goal_complete=False,method_frozen=False,final_writing_accepted=False,
        reason='Joint 450.119 s is 4.934 s slower than S2 445.185 s. Prior frozen/completed delivery is historical, not valid research acceptance.',
        active_authorization='2026-09-21 user: continue until goal; each new training/simulation round <=1800 wall seconds; no agents, branch changes, cleanup, commit or push.',
        next_action='Diagnose saved full-lap speed loss, observation and memory, then bounded targeted research.',
        audit=str(OUT/'saved_evidence_audit.json'),baseline=baseline,
        history_backup=str(archive),sealed_test_status='Previously exposed sets are development/history, never unseen; no sealed-set tuning.')
    state=read(statefile)
    state.update(updated_utc=now,goal_complete=False,current=correction['reason'],current_action='saved_trace_diagnosis_then_targeted_research',
                 next=correction['next_action'],next_action=correction['next_action'],research_resume=correction,
                 running=False,active_experiment=None)
    state['paper_delivery'].update(method_frozen=False,scoped_delivery_complete=False,
        historical_delivery_preserved=True,current_acceptance='rejected_as_final: slower than S2; 18-page approved baseline required',next_action=correction['next_action'])
    write(statefile,state)
    old=read(freeze); old.update(method_frozen=False,no_more_soft_metric_optimization=False,
                               goal_complete=False,remaining=correction['next_action'],superseded_by=str(OUT/'research_resume.json'))
    write(freeze,old)
    old=read(resume);old.update(correction);write(resume,old)
    write(OUT/'research_resume.json',correction)
    continuation.write_text('# PPO_article 当前续接\n\n目标未达成，方法未冻结，最终写作未验收。\n\n'
        '主方法 450.119 s，S2 445.185 s，强固定前瞻 454.700 s；主方法慢于 S2 4.934 s。此前 23 页稿与“完成/冻结”记录仅作历史，旧状态已完整备份。\n\n'
        f'唯一基稿：`{pdf}`（18 页）。核验及当前研究入口：`{OUT / "research_resume.json"}`。\n\n'
        '继续现有自然闭合长路径，全部 34 角、单侧 1.25 mm、向量 V/A/J=100/2000/62500、完整有序覆盖、制动和至少 20 ms 停稳。巡航主阈值 >=99 mm/s。每轮新训练/仿真 <=30 分钟；必要研究已授权。不派子代理、不切换或重置分支、不清理、不删除、不提交、不推送。\n\n'
        '下一步：已有轨迹分解及观测/记忆诊断，针对性修复后完整公平审核。可信超越两类基线且机制成立后才冻结与最终写作。\n',encoding='utf-8')
    print(json.dumps({k:{q:v[q] for q in ('all','near30','far30','quarters','far_feed_quantiles','far_feed_per_quarter') if q in v} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
