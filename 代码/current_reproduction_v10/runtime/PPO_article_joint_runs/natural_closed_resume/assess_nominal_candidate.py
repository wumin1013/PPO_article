"""Explicit nominal candidate screen; never substitutes for paired acceptance."""
from pathlib import Path
import argparse,json,sys
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('fixed');args=ap.parse_args()
    candidate=Path(args.candidate);fixed=Path(args.fixed)
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    result=read(candidate/'full_result.json');f=read(fixed/'full_result.json')
    files=[candidate/'full_result.json',fixed/'full_result.json']
    report=dict(candidate=str(candidate),fixed=str(fixed),success=result['success'],complete_time_s=result['complete_time_s'],
        goal_complete=False,method_frozen=False,paired_acceptance_not_established=True,all34_passed=False)
    if result['success']:
        jm=candidate/'full_metrics.json';fm=fixed/'full_metrics.json';sm=BASE/'paper_evidence_review/natural_closed_case/s2_paper_metrics.json'
        files += [jm,fm,sm];j=read(jm);maps=[{r['id']:r for r in read(p)['corners']} for p in [fm,sm]]
        corners=[]
        for c in j['corners']:
            gaps=[c['speed_deficit_mm']-m[c['id']]['speed_deficit_mm'] for m in maps]
            corners.append(dict(id=c['id'],passed=c['complete_window'] and max(gaps)<=1e-7,gap_fixed_mm=gaps[0],gap_s2_mm=gaps[1]))
        report.update(all34_passed=len(corners)==34 and all(r['passed'] for r in corners),corners=corners,
            failed_corners=[r['id'] for r in corners if not r['passed']],fixed_gain_s=round(f['complete_time_s']-result['complete_time_s'],9),
            s2_gain_s=round(445.185-result['complete_time_s'],9),corner_deficit_sum_mm=j['corner_total_speed_deficit_mm'],cruise99=j['cruise']['99'])
    report['source_sha256']={str(p):sha(p) for p in files}
    dest=candidate/'nominal_candidate_screen.json';assert not dest.exists();write_json(dest,report)
    print('NOMINAL_SCREEN',json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','corners']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
