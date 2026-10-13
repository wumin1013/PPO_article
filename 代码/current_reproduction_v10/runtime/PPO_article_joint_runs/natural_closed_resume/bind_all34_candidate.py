"""Freeze a reviewed candidate only after strict numerical and mechanism gates.

Preserves the previous manuscript evidence binding as history. Does not write TeX.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import json
import shutil
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import sha,write_json

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def source(**paths):
    result={}
    for key,path in paths.items():
        result[key]=str(path);result[key+'_sha256']=sha(Path(path))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('fixed');ap.add_argument('search')
    args=ap.parse_args();joint=Path(args.candidate);fixed=Path(args.fixed);search=Path(args.search)
    pairpath=joint/('paired_acceptance_'+fixed.name)/'assessment.json'
    pair=read(pairpath);reviewpath=joint/'core_mechanism_review.json';review=read(reviewpath)
    assert pair['quantitative_core_passed'] and review['passed']
    for path,digest in pair['source_sha256'].items():assert sha(Path(path))==digest
    for path,digest in review['source_sha256'].items():assert sha(Path(path))==digest
    selected=read(joint/'selected_policy.json');fixed_selection=read(fixed/'selected_policy.json')
    retained=read(search/'retained.json')
    selected_search=next(r for r in retained if r['spec']==selected['bend_spec'])
    # The complete-lap winner may be another retained conditional-search seed.
    # Bind that exact identity, without changing the original search ranking.
    checked_folder=joint.with_name(joint.name.removesuffix('_full'))
    checked_selection=read(checked_folder/'selected_policy.json')
    assert checked_selection['bend_spec']==selected['bend_spec']
    runtime=read(joint/'runtime_provenance/manifest.json')
    assert sha(Path(selected['model']))==selected['model_sha256']==runtime['model_sha256']
    for record in runtime['imported_source_files']:
        assert sha(Path(record['original']))==record['sha256']==sha(Path(record['snapshot']))
    case=BASE/'paper_evidence_review/natural_closed_case'
    stage=BASE/'paper_evidence_review/verified_results/all34_original_revision'
    stage.mkdir(parents=True,exist_ok=False)
    original=Path(r'D:\Nutstore\1\基于强化学习的运动规划方法\论文项目\submission_package_EAAI')
    binding=dict(created_utc=datetime.now(timezone.utc).isoformat(),method_frozen=True,
        goal_complete=False,final_manuscript_accepted=False,
        primary_methods=dict(Joint=source(trace=joint/'full_trace.npz',result=joint/'full_result.json',metrics=joint/'full_metrics.json'),
            Fixed=source(trace=fixed/'full_trace.npz',result=fixed/'full_result.json',metrics=fixed/'full_metrics.json'),
            S2=source(trace=case/'s2_complete/trace.npz',result=case/'s2_complete/result.json',metrics=case/'s2_paper_metrics.json')),
        resolution_results=dict(Joint=str(joint/'half_ms/full_result.json'),Fixed=str(fixed/'half_ms/full_result.json'),
            S2=str(OUT/'s2_resolution_half_ms/result.json')),
        research_acceptance=dict(passed=True,scope='Selected composite policy, same complete path, all34 registered corner windows and two physical resolutions',
            user_zero_deficit_tie_tolerance_mm=1e-7,time_working_floor_s=.1,strict_time_margin=True,
            independent_full_pipeline_ppo_replications=0),
        paired_assessment=str(pairpath),paired_assessment_sha256=sha(pairpath),
        core_mechanism_review=str(reviewpath),core_mechanism_review_sha256=sha(reviewpath),
        model=selected['model'],model_sha256=selected['model_sha256'],
        selected_policy=str(joint/'selected_policy.json'),selected_policy_sha256=sha(joint/'selected_policy.json'),
        fixed_policy=str(fixed/'selected_policy.json'),fixed_policy_sha256=sha(fixed/'selected_policy.json'),
        primary_fixed_D_mm=fixed_selection['fixed_d'],
        reference=str(case/'reference.csv'),reference_sha256=sha(case/'reference.csv'),
        geometry=str(case/'geometry.json'),geometry_sha256=sha(case/'geometry.json'),
        runtime_sources_manifest=str(joint/'runtime_provenance/manifest.json'),
        runtime_sources_manifest_sha256=sha(joint/'runtime_provenance/manifest.json'),
        final_joint_search=str(search),
        selected_conditional_search_seed=selected_search['seed'],
        selected_conditional_search_label=selected_search['selected_label'],
        complete_candidate_local_check_folder=str(checked_folder),
        recent_refinement_budget=str(OUT/'all34_dominance_research/recent_refinement_budget_with_transit_refits.json'),
        runtime_environment=str(OUT/'all34_dominance_research/current_runtime_environment.json'),
        efficiency_decomposition=str(joint/'complete_efficiency_decomposition/audit.json'),
        actual_turn_geometry=str(joint/'actual_turn_geometry/audit.json'),
        final_ppo_updates=0,observation_dimension=72,dimension_optimality_established=False,
        final_independent_model_count_per_arm=1,
        selection_metadata_note='Selection files preserve their pre-validation status fields; bound complete results and paired assessment establish current validity.',
        conditional_searches_are_not_independent_ppo_seeds=True,
        baseline_pdf=source(file=original/'01_manuscript_without_author_details/manuscript_without_author_details.pdf'),
        baseline_source=source(tex=original/'06_latex_source_files/latex_source_anonymous/manuscript.tex',
            bib=original/'06_latex_source_files/latex_source_anonymous/references.bib'),
        original_figures_1_3='Preserve exact files; suggestions supplied separately for author execution.',
        previous_binding=source(file=BASE/'paper_evidence_review/verified_results/paper_result_binding.json'))
    audit=read(joint/'channel_interventions/mechanism_audit.json')
    binding['interventions']={}
    for name,mode in [('Zero','zero_steering'),('Max','maximum_feed')]:
        folder=joint/'channel_interventions'/mode
        record=next(r for r in audit['records'] if r['mode']==mode)
        binding['interventions'][name]=dict(source(trace=folder/'trace.npz',result=folder/'result.json'),
            failed_corner=record['failed_corner'],kind='Output intervention without retraining')
    support=[Path(binding[k]) for k in ['recent_refinement_budget','runtime_environment','efficiency_decomposition','actual_turn_geometry']]
    support += [search/name for name in ['results.json','retained.json','registration.json','receipt.json','all34_checks.json']]
    support += [joint/'channel_interventions/mechanism_audit.json']
    support += [checked_folder/name for name in ['selected_policy.json','all34_checks.json','registration.json','receipt.json']]
    choice=OUT/'all34_dominance_research/selected_all34_candidate_decision.json'
    if choice.exists():
        assert read(choice)['selected']==str(joint)
        support.append(choice)
    support += [Path(path) for path in binding['resolution_results'].values()]
    binding['supporting_evidence_sha256']={str(p):sha(p) for p in support}
    validation_index=search/'retained_complete_validation_index.json'
    if validation_index.exists():
        index=read(validation_index)
        assert index['all_complete_pairs_available']
        for row in index['records']:
            for path,digest in row['source_sha256'].items():assert sha(Path(path))==digest
        assert any(r['candidate']==str(joint) for r in index['records'])
        binding['retained_complete_validation_index']=str(validation_index)
        binding['supporting_evidence_sha256'][str(validation_index)]=sha(validation_index)
    write_json(stage/'paper_result_binding.json',binding)
    write_json(stage/'research_acceptance.json',dict(binding=source(file=stage/'paper_result_binding.json'),
        passed=True,numerical=pair,mechanism=review,final_manuscript_accepted=False,goal_complete=False))
    rows=[]
    for method,paths in binding['primary_methods'].items():
        for resolution in ['1ms','0.5ms']:
            if resolution=='1ms':metrics=read(paths['metrics'])
            elif method=='S2':metrics=read(OUT/'all34_dominance_research/s2_half_ms_corner_metrics.json')
            else:metrics=read((joint if method=='Joint' else fixed)/'half_ms/corner_metrics.json')
            for c in metrics['corners']:
                rows.append(dict(method=method,resolution=resolution,id=c['id'],
                    included_angle_deg=c.get('included_angle_deg',c.get('interior_angle_deg')),
                    complete_window=c['complete_window'],speed_deficit_mm=c['speed_deficit_mm'],
                    minimum_speed_mm_s=c['minimum_speed_mm_s']))
    with (stage/'all34_corner_statistics.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history=OUT/'all34_dominance_research/state_history'/stamp;history.mkdir(parents=True,exist_ok=False)
    states=[BASE/'当前执行状态.json',OUT/'live_research_summary.json',OUT/'all34_dominance_research/active_research_checkpoint.json']
    for path in states:
        shutil.copy2(path,history/path.name)
        state=read(path)
        state.update(updated_utc=stamp,goal_complete=False,method_frozen=True,research_acceptance_passed=True,
            final_manuscript_accepted=False,authoritative_acceptance=source(file=stage/'research_acceptance.json'),
            current='核心研究验收通过；方法及数据已明确绑定。正在从18页基稿对应源码完成论文和图件，整项任务尚未完成。',
            active_sessions={},active_validation_sessions={},current_candidate=str(joint),
            current_candidate_complete_times_s={name:dict(nominal_s=read(binding['primary_methods'][name]['result'])['complete_time_s'],
                half_ms_s=read(path)['complete_time_s']) for name,path in binding['resolution_results'].items()},
            current_candidate_all34_passed=True,current_candidate_all34_nominal_passed=True,
            current_candidate_resolution_verified=True,
            latest_revision_history=str(history),current_paper_binding=str(stage/'paper_result_binding.json'),
            next_steps=['从唯一18页原稿对应源码作必要方法更正，重写原实验框架','保留前三图，完成九主图与20–25页PDF','逐页视觉和引用数据核验'])
        write_json(path,state)
    print('FROZEN_RESEARCH_BINDING',stage/'paper_result_binding.json',flush=True)

if __name__=='__main__':main()
