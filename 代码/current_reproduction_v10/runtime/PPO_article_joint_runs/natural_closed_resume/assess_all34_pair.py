"""Audit paired complete trajectories; missing results never become acceptance."""
from pathlib import Path
import sys,json,argparse
from decimal import Decimal
import numpy as np
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
R=OUT/'all34_dominance_research'

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def qualified(folder):
    file=folder/'complete_contract_audit.json'
    if not file.exists():file=folder/'full_result.json'
    return read(file),file


def time_comparison(tn,th,bnt,bht,protocol):
    """Compare recorded decimal times strictly, without binary boundary drift."""
    # Recorded durations lie on a >=0.5ms grid; nanosecond normalization removes
    # only decimal serialization tails such as444.32049999999997.
    tn,th,bnt,bht=[Decimal(str(t)).quantize(Decimal('0.000000001')) for t in [tn,th,bnt,bht]]
    sensitivity=abs(tn-th)+abs(bnt-bht)
    required=max(Decimal(str(protocol['time_gain_working_floor_s'])),
                 Decimal(str(protocol['numerical_sensitivity_factor']))*sensitivity)
    gn,gh=bnt-tn,bht-th
    return dict(nominal_gain_s=float(gn),half_ms_gain_s=float(gh),
                observed_sensitivity_sum_s=float(sensitivity),required_gain_s=float(required),
                passed=bool(gn>required and gh>required),
                comparison='Strict decimal comparison after1ns serialization normalization of>=0.5ms tick times; equality does not pass.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('fixed');args=ap.parse_args()
    joint=Path(args.candidate);fixed=Path(args.fixed)
    stage=joint/('paired_acceptance_'+fixed.name);stage.mkdir(exist_ok=False)
    protocol_path=R/'all34_numerical_acceptance_protocol.json';protocol=read(protocol_path)
    files=[joint/'half_ms/full_result.json',fixed/'half_ms/full_result.json',
           joint/'half_ms/corner_metrics.json',fixed/'half_ms/corner_metrics.json']
    missing=[str(p) for p in files if not p.exists()]
    if missing:
        write_json(stage/'pending.json',dict(goal_complete=False,complete=False,missing=missing));print('ACCEPTANCE_PENDING',missing);return
    jn,jnpath=qualified(joint);fn,fnpath=qualified(fixed)
    jh=read(joint/'half_ms/full_result.json');fh=read(fixed/'half_ms/full_result.json')
    snpath=BASE/'paper_evidence_review/natural_closed_case/s2_complete/result.json';shpath=OUT/'s2_resolution_half_ms/result.json'
    sn=read(snpath);sh=read(shpath)
    comparisons={};corner_reports={}
    source_files=[Path(__file__),jnpath,fnpath,snpath,shpath,protocol_path]+files
    for name,bn,bh in [('S2',sn,sh),('Fixed',fn,fh)]:
        success=all(x['success'] for x in [jn,jh,bn,bh]);result=dict(full_success_at_both_resolutions=success,passed=False)
        if success:
            tn,th=jn['complete_time_s'],jh['complete_time_s'];bnt,bht=bn['complete_time_s'],bh['complete_time_s']
            result.update(time_comparison(tn,th,bnt,bht,protocol))
        comparisons[name]=result
    for res in ['1ms','0.5ms']:
        if res=='1ms':
            jm_path=joint/'full_metrics.json';fm_path=fixed/'full_metrics.json';sm_path=BASE/'paper_evidence_review/natural_closed_case/s2_paper_metrics.json'
        else:
            jm_path=joint/'half_ms/corner_metrics.json';fm_path=fixed/'half_ms/corner_metrics.json';sm_path=R/'s2_half_ms_corner_metrics.json'
        if not all(p.exists() for p in [jm_path,fm_path,sm_path]):corner_reports[res]=dict(passed=False,missing=True);continue
        source_files.extend([jm_path,fm_path,sm_path]);jm,fm,sm=[read(p) for p in [jm_path,fm_path,sm_path]]
        records=[]
        for j,f,s in zip(jm['corners'],fm['corners'],sm['corners']):
            assert j['id']==f['id']==s['id']
            complete=all(c['complete_window'] for c in [j,f,s])
            if complete:
                df=j['speed_deficit_mm']-f['speed_deficit_mm'];ds=j['speed_deficit_mm']-s['speed_deficit_mm']
                passed=bool(df<=1e-7 and ds<=1e-7)
            else:df=ds=None;passed=False
            records.append(dict(id=j['id'],window_complete=complete,joint_deficit_mm=j.get('speed_deficit_mm'),
                fixed_deficit_mm=f.get('speed_deficit_mm'),s2_deficit_mm=s.get('speed_deficit_mm'),gap_fixed_mm=df,gap_s2_mm=ds,passed=passed))
        corner_reports[res]=dict(passed=len(records)==34 and all(r['passed'] for r in records),records=records,
            failed=[r['id'] for r in records if not r['passed']])
    quantitative=all(x['passed'] for x in comparisons.values()) and all(x['passed'] for x in corner_reports.values())
    report=dict(goal_complete=False,method_frozen=False,quantitative_core_passed=quantitative,
        candidate=str(joint),fixed=str(fixed),time_comparisons=comparisons,corner_comparisons=corner_reports,
        source_sha256={str(p):sha(p) for p in source_files},mechanism_visual_review_still_required=True,
        interpretation='Two-resolution deterministic evidence; not a continuum error certificate or independent PPO replication. No method or manuscript freeze from this numerical report alone.')
    write_json(stage/'assessment.json',report)
    print('PAIRED_ALL34_ASSESSMENT',quantitative,comparisons,{k:v['failed'] for k,v in corner_reports.items()},flush=True)

if __name__=='__main__':main()
