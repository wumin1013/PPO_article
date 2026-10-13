"""Saved-log statistics preserve failed pairs, study namespaces and provenance."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import preview_direct_report as report


def row(method,seed,success,time):
    return dict(stage='formal',split='validation',regime='nominal',method=method,
                seed=seed,path_id='mixed_lengths_validation',success=success,elapsed_time_s=time)


def test_direct_pairs_exclude_failures_and_keep_exact_labels():
    rows=[row('selected_fixed',11,True,10.),row('direct',11,True,9.),
          row('selected_fixed',22,False,1.),row('direct',22,True,8.),
          row('direct_no_anchor',11,False,2.),row('direct_no_anchor',22,False,2.)]
    summaries,pairs,suites=report.direct_pairs(rows,rows)
    selected=next(r for r in summaries if r['baseline']=='selected_fixed')
    assert selected['candidate']=='direct'
    assert selected['n_common_success']==1
    assert selected['baseline_minus_candidate_mean_s']==1.
    assert selected['relative_reduction_mean_percent']==10.
    assert selected['baseline_minus_candidate_sd_s'] is None
    assert selected['noncommon_outcomes']=={'baseline_failed':1}
    assert len(pairs)==1 and pairs[0]['seed']==11
    empty=next(r for r in summaries if r['baseline']=='direct_no_anchor')
    assert empty['n_common_success']==0
    assert empty['baseline_minus_candidate_mean_s'] is None
    assert all('gate' not in str(r) for group in (summaries,pairs,suites) for r in group)


def test_unanchored_preview_uses_its_independent_fixed_baseline():
    rows=[row('selected_fixed',11,True,10.),row('direct',11,True,9.),
          row('selected_fixed_no_anchor',11,True,8.),row('direct_no_anchor',11,True,7.)]
    summaries,pairs,_=report.direct_pairs(rows,rows)
    matched=next(r for r in summaries if r['comparison_role']=='matched_beta0_preview')
    assert matched['baseline']=='selected_fixed_no_anchor'
    assert matched['candidate']=='direct_no_anchor'
    assert matched['baseline_minus_candidate_mean_s']==1.
    assert matched['relative_reduction_mean_percent']==12.5
    assert len(pairs)==3
    assert not any(r['baseline']=='selected_fixed' and r['candidate']=='direct_no_anchor' for r in pairs)


def test_jsonl_tolerates_only_unfinished_last_write(tmp_path):
    path=tmp_path/'live.jsonl';inputs=report.saved.Inputs()
    path.write_text('{"step": 512}\n{"step":',encoding='utf-8')
    rows,partial=report.read_json_lines(path,inputs)
    assert rows==[{'step':512}] and partial
    assert inputs.sha(path)==hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('{"step": broken}\n{"step": 512}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Corrupt completed'):
        report.read_json_lines(path,inputs)


def test_training_diagnostics_rejects_mixed_runtime(tmp_path):
    run=tmp_path/'pilot/direct_seed101';run.mkdir(parents=True)
    (run/'actor_update_diagnostics.jsonl').write_text('',encoding='utf-8')
    (run/'provenance.json').write_text(json.dumps({'source_sha256':'wrong'}),encoding='utf-8')
    protocol=dict(pilot_seed=101,pilot_variants=['direct'],source_sha256='correct')
    with pytest.raises(ValueError,match='provenance mismatch'):
        report.training_diagnostics(tmp_path,protocol,report.saved.Inputs())


def test_gain_tex_keeps_separate_labels_and_common_change():
    protocol=dict(pilot_steps=32768,pilot_seed=101,formal_steps=131072,
                  formal_seeds=[11,22,33,44,55],shared_guidance={'gain_per_s':15.,'tangent_blend':.015})
    tex=report.tex_report([],[],[],protocol,None,[],prefix='preview_direct_gain')
    assert r'\label{sec:preview_direct_gain}' in tex
    assert 'shared heading gain 15' in tex
    assert 'not attributed to preview learning' in tex
    assert 'does not by itself prove an actual path shortcut' in tex
    assert 'Formal five-seed evaluation remains pending' in tex


def test_report_requires_direct_namespace_before_reading_results(tmp_path):
    with pytest.raises(ValueError,match='artifact prefix'):
        report.build_report(tmp_path/'unrelated_study')


def test_trace_contract_reconstructs_rewards_mapping_and_time():
    data={'timestamp_s':np.array([0.,.001,.002]),'reward':np.array([0.,3.,4.]),
          'lookahead_mm':np.array([3.,2.7,3.]),'control_status__direct_selected_mm':np.array([3.,2.7,3.]),
          'reward_components__r_progress':np.array([0.,4.,6.]),
          'reward_components__r_time':np.array([0.,-1.,-1.]),
          'reward_components__r_track':np.array([0.,0.,-1.])}
    class Input:
        def trace(self,path):return data
        def sha(self,path):return 'saved-trace-hash'
    run=dict(row('direct',11,True,.002),trace_file='trace.npz',_trace_path='trace.npz')
    output=report.evaluation_trace_summaries([run],Input())[0]
    assert output['reward_component_reconstruction_max_error']==0.
    assert output['distance_mapping_log_error_max_mm']==0.
    assert output['elapsed_time_log_error_s']==0.
    assert output['sample_interval_min_s']==output['sample_interval_max_s']==.001


def test_historical_regression_is_separate_and_hash_checked(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'PAPER',tmp_path/'paper')
    historic=report.PAPER/'generated/configs/full_method_snapshot.yaml'
    historic.parent.mkdir(parents=True);historic.write_text('historical: square\n',encoding='utf-8')
    base=tmp_path/'preview_direct';data=tmp_path/'data';data.mkdir()
    manifest=data/'manifest.json';manifest.write_text(json.dumps({'historical_regression_paths':['square']}),encoding='utf-8')
    run=base/'formal/direct_seed11';out=run/'evaluation_regression_nominal';out.mkdir(parents=True)
    (run/'checkpoint.pt').write_bytes(b'model');(run/'config.yaml').write_bytes(b'config')
    (out/'effective_config.yaml').write_bytes(b'effective');(out/'square.npz').write_bytes(b'trace-placeholder')
    inputs=report.saved.Inputs();protocol=dict(formal_variants=['direct'],formal_seeds=[11,22],formal_steps=32,source_sha256='source')
    provenance=dict(source_sha256='source',config_sha256=inputs.sha(run/'config.yaml'),path_manifest_sha256=inputs.sha(manifest))
    (run/'provenance.json').write_text(json.dumps(provenance),encoding='utf-8')
    (run/'training_summary.json').write_text(json.dumps({'steps':32}),encoding='utf-8')
    result=dict(variant='direct',seed=11,path_id='square',split='regression',regime='nominal',training_steps=32,
        checkpoint_sha256=inputs.sha(run/'checkpoint.pt'),config_sha256=inputs.sha(run/'config.yaml'),
        effective_config_sha256=inputs.sha(out/'effective_config.yaml'),code_sha256='source')
    result_path=out/'square.json';result_path.write_text(json.dumps(result),encoding='utf-8')
    snapshot=dict(protocol=protocol,inputs=inputs,expected=[],pending=[],rows=[])
    report.append_saved_regression(snapshot,base,data)
    assert len(snapshot['rows'])==1 and len(snapshot['pending'])==1
    assert snapshot['rows'][0]['split']=='regression'
    assert 'not unseen evidence' in snapshot['rows'][0]['regression_scope']
    result['code_sha256']='stale';result_path.write_text(json.dumps(result),encoding='utf-8')
    with pytest.raises(ValueError,match='regression provenance'):
        report.append_saved_regression(dict(protocol=protocol,inputs=report.saved.Inputs(),expected=[],pending=[],rows=[]),base,data)


def test_compact_paper_does_not_substitute_validation_for_test():
    protocol=dict(pilot_variants=list(range(10)),pilot_steps=32768,
                  formal_variants=['selected_fixed','direct','selected_fixed_no_anchor','direct_no_anchor'],
                  formal_seeds=[11,22,33,44,55],formal_steps=131072)
    snapshot=dict(protocol=protocol,selection={'fixed_distance_mm':3.,'fixed_distance_no_anchor_mm':3.},
                  rows=[dict(row('direct',11,True,4.),split='validation')])
    text=report.compact_paper(snapshot,[],[],[],[],'preview_direct_explore')
    assert r'\subsection{Frozen simulation evaluation}' in text
    assert '10 pilot' in text and '4 methods' in text
    assert 'Frozen held-out test results are pending' in text
    assert 'not substituted' in text
    assert r'\begin{table}' not in text


def test_compact_test_tables_keep_failure_and_matched_comparison():
    methods=['selected_fixed','direct','selected_fixed_no_anchor','direct_no_anchor']
    protocol=dict(pilot_variants=methods,pilot_steps=32,formal_variants=methods,formal_seeds=[11],formal_steps=64)
    rows=[]
    for method in methods:
        failed=method=='direct'
        rows.append(dict(row(method,11,not failed,4.),split='test',path_id='fixture_path',
            failure_reasons=['fixture_failure'] if failed else [],termination_reason='fixture',ordered_progress=1.,
            contour_error_max_mm=.1,contour_error_rmse_mm=.08,contour_error_mean_mm=.07))
    snapshot=dict(protocol=protocol,selection={'fixed_distance_mm':3.,'fixed_distance_no_anchor_mm':3.},rows=rows,budgets=[])
    aggregates=report.saved.aggregate(rows,rows);paired,_,_=report.direct_pairs(rows,rows)
    text=report.compact_paper(snapshot,aggregates,paired,[],[],'preview_direct_explore')
    assert r'fixture\_failure: 1' in text
    assert '1/1 failed nominal test runs' in text
    assert 'Matched-anchor common-success' in text
    assert r'\label{tab:preview_direct_explore_paper_cost}' in text
    assert 'are pending' not in text


def test_suite_interval_uses_seed_totals_not_path_or_point_counts():
    details=[dict(seed=seed,path_ids=['a','b','c'],baseline_total_s=20.,candidate_total_s=20.-delta,
                  baseline_minus_candidate_total_s=delta) for seed,delta in zip([11,22,33,44,55],[.2,.1,0.,-.1,.3])]
    rows=[dict(complete_suite_seed_details=details)]
    report.add_suite_confidence_intervals(rows)
    result=rows[0]
    expected_sd=np.std([.2,.1,0.,-.1,.3],ddof=1)
    margin=report.student_t.ppf(.975,4)*expected_sd/np.sqrt(5)
    assert result['baseline_minus_candidate_total_95ci_low_s']==pytest.approx(.1-margin)
    assert result['relative_reduction_total_mean_percent']==pytest.approx(.5)
    assert result['relative_reduction_total_95ci_low_percent']==pytest.approx((.1-margin)*5.)
    empty=[dict(complete_suite_seed_details=[])]
    report.add_suite_confidence_intervals(empty)
    assert empty[0]['relative_reduction_total_95ci_low_percent'] is None


def test_serial_timing_recalculates_saved_samples_and_keeps_model_unit(tmp_path,monkeypatch):
    base=tmp_path/'preview_direct';directory=base/'inference_benchmark';directory.mkdir(parents=True)
    (base/'freeze.json').write_text('{}',encoding='utf-8')
    (directory/'preview_inference_benchmark.py').write_text('# timing fixture\n',encoding='utf-8')
    np.save(directory/'shared_training_states.npy',np.zeros((2,20),np.float32))
    samples=np.arange(100,dtype=float)+1.
    np.savez_compressed(directory/'latency_samples.npz',formal__direct_seed11=samples)
    inputs=report.saved.Inputs();frozen={'models':{'formal/direct_seed11':'checkpoint'},'configs':{'formal/direct_seed11':'config'}}
    monkeypatch.setattr(report.saved,'verify_frozen',lambda *args:frozen)
    summary=dict(rows=[dict(run='formal/direct_seed11',method='direct',seed=11,checkpoint_sha256='checkpoint',config_sha256='config',
        calls=100,median_us=float(np.median(samples)),p95_us=float(np.percentile(samples,95)),mean_us=float(np.mean(samples)))],
        freeze_sha256=inputs.sha(base/'freeze.json'),script_sha256=inputs.sha(directory/'preview_inference_benchmark.py'),
        states_sha256=inputs.sha(directory/'shared_training_states.npy'),raw_sha256=inputs.sha(directory/'latency_samples.npz'),
        device='cpu',torch_threads=1,scope='Serial actor only',scheduling='Fixture',warmup_calls_per_policy=200)
    path=directory/'summary.json';path.write_text(json.dumps(summary),encoding='utf-8')
    snapshot=dict(inputs=inputs,protocol={'formal_variants':['direct'],'formal_seeds':[11]})
    _,costs=report.serial_actor_benchmark(snapshot,base,tmp_path)
    assert costs[0]['n_independent_models']==1
    assert costs[0]['median_us_across_seeds_mean']==50.5
    assert costs[0]['median_us_across_seeds_sd'] is None
    summary['rows'][0]['median_us']+=1;path.write_text(json.dumps(summary),encoding='utf-8')
    with pytest.raises(ValueError,match='differs from raw samples'):
        report.serial_actor_benchmark(snapshot,base,tmp_path)
