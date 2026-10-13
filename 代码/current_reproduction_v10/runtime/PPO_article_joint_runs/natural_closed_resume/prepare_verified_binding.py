"""Bind completed, audited development evidence without changing any policy."""
from pathlib import Path
import json, sys, hashlib, datetime, csv
import numpy as np
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from study_io import sha, write_json

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    joint=OUT/'joint_actual_context_feedback_training/exact_continuation'
    gate=read(joint/'numerical_acceptance.json')
    assert gate['numerical_pass']
    visual=OUT/'actual_context_mechanism_inspection/visual_qa.json'
    assert read(visual)['candidate_trace_sha256']==sha(joint/'full_trace.npz')
    fixed=Path(gate['comparisons']['Fixed']['comparator']).parent
    old=read(BASE/'paper_evidence_review/natural_closed_case/paper_result_binding.json')
    primary={'Joint':{},'S2':old['primary_methods']['S2'],'Fixed':{}}
    for label,stage in [('Joint',joint),('Fixed',fixed)]:
        for field,filename in [('trace','full_trace.npz'),('result','full_result.json'),('metrics','full_metrics.json')]:
            p=stage/filename;primary[label][field]=str(p);primary[label][field+'_sha256']=sha(p)
    for item in primary.values():
        for k in ['trace','result','metrics']:
            p=BASE/item[k];assert sha(p)==item[k+'_sha256'];item[k]=str(p)
        r=read(item['result']);m=read(item['metrics'])
        assert r['success'] and r['audit']['rest_pass'] and r['audit']['kinematic_pass']
        assert r['geometry']['gates_passed']==r['geometry']['gate_count']==2362
        assert r['analytic']['analytic_corridor_pass'] and m['all34_windows_complete'] and len(m['corners'])==34
    joint_repeats=[joint/'retained_seed921221_step6/exact_continuation/full_result.json',
                   joint/'retained_seed921222_step6/exact_continuation/full_result.json',joint/'full_result.json']
    froot=OUT/'fixed_actual_context_feedback_training/full_context_candidates'
    fixed_repeats=[froot/'retained_seed921221_retained/full_result.json',
                   froot/'retained_seed921222_retained/exact_continuation/full_result.json',
                   froot/'retained_seed921223_retained/full_result.json']
    for p in joint_repeats+fixed_repeats:
        r=read(p);assert 'TimeoutError' not in str(r.get('error')), 'Unfinished conditional evaluation'
    zero=joint/'probe_zero_steering/checkpoint_continuation/full_result.json'
    feed=joint/'probe_maximum_feed/full_result.json'
    assert read(zero)['success'] and not read(feed)['success']
    assert np.max(np.abs(np.load(zero.parent/'full_trace.npz')['actions'][:,0]))==0
    selection=read(joint/'selected_policy.json');model=Path(selection['model'])
    assert sha(model)==selection['model_sha256']
    binding=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        primary_methods=primary,primary_fixed_D_mm=read(fixed/'selected_policy.json')['spec']['fixed_d'],
        research_acceptance=dict(passed=True,scope='Disclosed development path and selected composite model only',
            numerical_gate=str(joint/'numerical_acceptance.json'),numerical_gate_sha256=sha(joint/'numerical_acceptance.json'),
            mechanism_review=str(visual),mechanism_review_sha256=sha(visual),
            soft_tradeoffs=read(visual)['soft_negatives'],independent_ppo_replications=0),
        model=str(model),model_sha256=sha(model),selected_policy=str(joint/'selected_policy.json'),
        selected_policy_sha256=sha(joint/'selected_policy.json'),
        reference=str(BASE/'paper_evidence_review/natural_closed_case/reference.csv'),
        reference_sha256=sha(BASE/'paper_evidence_review/natural_closed_case/reference.csv'),
        geometry=str(BASE/'paper_evidence_review/natural_closed_case/geometry.json'),
        geometry_sha256=sha(BASE/'paper_evidence_review/natural_closed_case/geometry.json'),
        fixed_training_stage=str(OUT/'fixed_actual_context_feedback_training'),
        conditional_selection_stages={n:str(OUT/(n.lower()+'_actual_context_feedback_training')) for n in ['Joint','Fixed']},
        ablations=[dict(label='Fixed D',result=str(fixed/'full_result.json'),kind='Optimized'),
                   dict(label='Zero steering',result=str(zero),kind='Intervention'),
                   dict(label='Maximum feed',result=str(feed),kind='Intervention')],
        resolution_results={'Joint':gate['candidate_refined_result'],'S2':gate['comparisons']['S2']['refined_comparator'],
                            'Fixed':gate['comparisons']['Fixed']['refined_comparator']},
        conditional_full_repeats={n:[dict(seed=921221+i,result=str(p),result_sha256=sha(p)) for i,p in enumerate(pp)]
                                  for n,pp in [('Joint',joint_repeats),('Fixed',fixed_repeats)]},
        method_frozen=True,final_manuscript_accepted=False,goal_complete=False,
        heldout_testing='None in this continuation; previously exposed sealed collections are not unseen tests.')
    evidence=BASE/'paper_evidence_review/verified_results';evidence.mkdir(exist_ok=False)
    write_json(evidence/'paper_result_binding.json',binding)
    tables=[];corners=[]
    for name,item in primary.items():
        m=read(item['metrics']);r=read(item['result']);z=np.load(item['trace']);rows=z['rows']
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        tables.append(dict(method=name,time_s=r['complete_time_s'],half_ms_time_s=read(binding['resolution_results'][name])['complete_time_s'],
            distance_mm=m['travel_distance_mm'],deficit_equivalent_s=float(np.sum(1-speed/100)*.001),
            cruise99_s=m['cruise']['99']['time_s'],longest_cruise99_s=m['cruise']['99']['longest_time_s'],
            corner_deficit_mm=m['corner_total_speed_deficit_mm'],max_distance_mm=r['analytic']['maximum_evaluated_distance_mm'],
            endpoint_error_mm=r['geometry']['endpoint_error_mm'],vmax=r['audit']['peaks'][0],amax=r['audit']['peaks'][1],jmax=r['audit']['peaks'][2]))
        corners.extend([dict(method=name,**c) for c in m['corners']])
    write_json(evidence/'summary_tables.json',dict(methods=tables,corners=corners,ablations=binding['ablations'],conditional_full_repeats=binding['conditional_full_repeats']))
    for filename,data in [('method_summary.csv',tables),('all34_corner_statistics.csv',corners)]:
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (evidence/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    print(json.dumps(dict(binding=str(evidence/'paper_result_binding.json'),methods=tables),indent=2))

if __name__=='__main__':main()
