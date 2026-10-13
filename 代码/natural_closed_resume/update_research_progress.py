"""Update live state from an explicit assessed candidate, preserving prior state."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil,sys
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('fixed');ap.add_argument('--current',required=True)
    ap.add_argument('--session',action='append',default=[]);args=ap.parse_args()
    candidate=Path(args.candidate);fixed=Path(args.fixed);root=OUT/'all34_dominance_research'
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');history=root/'state_history'/stamp;history.mkdir()
    files=[BASE/'当前执行状态.json',OUT/'live_research_summary.json',root/'active_research_checkpoint.json']
    sessions={k:int(v) for k,v in [q.split('=') for q in args.session]}
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    nominal=read(candidate/'full_result.json');refined=read(candidate/'half_ms/full_result.json')
    assessment_file=candidate/('paired_acceptance_'+fixed.name)/'assessment.json';assessment=read(assessment_file)
    metrics=read(candidate/'full_metrics.json')
    # Corner report has explicit lists of violations for each registered resolution.
    corners=assessment['corner_comparisons']
    all34=all(q['passed'] for q in corners.values())
    values=dict(updated_utc=stamp,current=args.current,goal_complete=False,method_frozen=False,research_acceptance_passed=False,
        final_manuscript_accepted=False,authoritative_acceptance=None,active_sessions=sessions,active_validation_sessions=sessions,
        candidate=str(candidate),fixed=str(fixed),last_verified_candidate=str(candidate),running_candidate=None,
        current_candidate_time_scope='Current candidate complete measured pair; strict acceptance remains separate.',
        current_candidate_complete_times_s=dict(Joint_1ms=nominal['complete_time_s'],Joint_half_ms=refined['complete_time_s'],
            Fixed_1ms=read(fixed/'full_result.json')['complete_time_s'],Fixed_half_ms=read(fixed/'half_ms/full_result.json')['complete_time_s'],S2_1ms=445.185,S2_half_ms=445.185),
        current_candidate_bindings={'Joint':str(candidate/'full_result.json'),'Joint_half_ms':str(candidate/'half_ms/full_result.json'),
            'assessment':str(assessment_file),'assessment_sha256':sha(assessment_file)},
        current_candidate_resolution_verified=nominal['success'] and refined['success'],current_candidate_all34_passed=all34,
        current_candidate_all34_nominal_passed=corners['1ms']['passed'],current_quantitative_core_passed=assessment['quantitative_core_passed'],
        current_time_comparisons=assessment['time_comparisons'],latest_revision_history=str(history),
        current_candidate_nominal_mechanism=dict(status='Candidate-specific geometry and intervention audits available; no method freeze.',
            cruise=metrics['cruise'],audit_folder=str(candidate/'actual_turn_geometry'),intervention_audit=str(candidate/'channel_interventions/mechanism_audit.json')),
        next_active_research=['检查已保留候选的全34及完整圈表现，避免重复局部目标搜索','合格候选完成细分辨率与自身机制审计','严格达标后才冻结并完成原稿上的论文和PDF验收'])
    for file in files:
        shutil.copy2(file,history/file.name);previous=read(file)
        for key in ['final_pdf','delivery_index']:
            if previous.get(key) is not None:previous['historical_'+key]=previous[key]
            previous[key]=None
        previous.update(values)
        if 'next_steps' in previous:previous['next_steps']=values['next_active_research']
        write_json(file,previous)
    print('LIVE_STATE_UPDATED',str(history),flush=True)

if __name__=='__main__':main()
