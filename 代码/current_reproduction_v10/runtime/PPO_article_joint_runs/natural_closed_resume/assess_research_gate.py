"""Read-only numerical acceptance gate; never freezes a model automatically."""
from pathlib import Path
import argparse,json,sys
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from study_io import sha,write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def resolution_result(stage):
    path=stage/'resolution_0.0005/result.json'
    retry=path.parent/'checkpoint_continuation/result.json'
    if retry.exists() and read(retry)['success']:
        assert read(retry)['prefix_sha256']==sha(path.parent/'trace.npz')
        return retry
    while (path.parent/'exact_continuation/result.json').exists():
        continued=path.parent/'exact_continuation/result.json'
        assert read(continued)['prefix_sha256']==sha(path.parent/'trace.npz')
        path=continued
    return path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');args=ap.parse_args()
    stage=Path(args.candidate);protocol=read(OUT/'numerical_adoption_protocol.json')
    candidate=read(stage/'full_result.json');refined_path=resolution_result(stage)
    refined=read(refined_path) if refined_path.exists() else None
    case=BASE/'paper_evidence_review/natural_closed_case'
    old=read(case/'paper_result_binding.json')['primary_methods']
    s2_path=BASE/old['S2']['result']
    comparators={'S2':dict(nominal=read(s2_path),nominal_path=str(s2_path),
        refined=read(OUT/'s2_resolution_half_ms/result.json'),refined_path=str(OUT/'s2_resolution_half_ms/result.json'))}
    fixed=[]
    for folder in OUT.glob('fixed_*'):
        if not folder.is_dir():continue
        for path in folder.rglob('full_result.json'):
            r=read(path)
            if r['success']:fixed.append((r['complete_time_s'],path,r))
    historical=BASE/old['Fixed']['result'];r=read(historical)
    if r['success']:fixed.append((r['complete_time_s'],historical,r))
    _,path,r=min(fixed,key=lambda x:x[0]);half=resolution_result(path.parent)
    comparators['Fixed']=dict(nominal=r,nominal_path=str(path),refined=read(half) if half.exists() else None,refined_path=str(half))
    assessments={}
    for name,other in comparators.items():
        valid=bool(candidate['success'] and refined and refined['success'] and other['nominal']['success'] and other['refined'] and other['refined']['success'])
        a=dict(full_success_at_both_resolutions=valid,comparator=str(other['nominal_path']),refined_comparator=other['refined_path'])
        if valid:
            tn=candidate['complete_time_s'];th=refined['complete_time_s'];bn=other['nominal']['complete_time_s'];bh=other['refined']['complete_time_s']
            sensitivity=abs(tn-th)+abs(bn-bh);floor=protocol['s2_gain_floor_s' if name=='S2' else 'fixed_gain_floor_s']
            a.update(nominal_gain_s=bn-tn,half_ms_gain_s=bh-th,observed_sensitivity_sum_s=sensitivity,
                required_gain_s=max(floor,protocol['numerical_sensitivity_factor']*sensitivity),
                passed=min(bn-tn,bh-th)>max(floor,protocol['numerical_sensitivity_factor']*sensitivity)+1e-9)
        else:a.update(passed=False,reason='Complete successful evidence at both resolutions is missing or failed.')
        assessments[name]=a
    result=dict(candidate=str(stage),comparisons=assessments,numerical_pass=all(r['passed'] for r in assessments.values()),
        method_frozen=False,goal_complete=False,mechanism_review_required=True,
        numerical_interpretation='Observed two-resolution sensitivity screen, not a continuum error bound or independent-training replication.',
        candidate_result_sha256=sha(stage/'full_result.json'),candidate_refined_result=str(refined_path),protocol_sha256=sha(OUT/'numerical_adoption_protocol.json'))
    write_json(stage/'numerical_acceptance.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
