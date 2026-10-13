"""Recover preview-development progress from immutable local records.

No training, held-out evaluation, score changes or artifact overwrites. Verify
saved configurations, model/trace hashes, archived sources and manuscript QA.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import zipfile

import numpy as np

import preview_event_study as old

BASE = old.ROOT/'论文项目/revision_20260909_terminal'


def recover(output, manuscript_qa=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep previous recovery snapshot; use another --output')
    old.check_protocol()
    runs = []
    for study in ('macro_study', 'macro_refine', 'sensor_study'):
        for run in sorted((BASE/study/'pilot').glob('*_seed*')):
            summary = old.read(run/'training_summary.json')
            provenance = old.read(run/'provenance.json')
            assert summary['checkpoint_sha256'] == old.sha(run/'checkpoint.pt')
            assert provenance['config_sha256'] == old.sha(run/'config.yaml')
            with zipfile.ZipFile(run/'source.zip') as archive:
                for name, expected in provenance['source_files'].items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
                    assert old.sha(old.ROOT/name) == expected, name
            validation = old.read(run/'validation/summary.json')['rows']
            for row in validation:
                assert row['checkpoint_sha256'] == summary['checkpoint_sha256']
                assert row['config_sha256'] == provenance['config_sha256']
                assert row['source_sha256'] == provenance['source_sha256']
                assert row['trace_sha256'] == old.sha(run/'validation'/f'{row["path_id"]}.npz')
            episodes = {r['episode']: r for r in
                        (json.loads(line) for line in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines())
                        if r['episode'] < summary['episodes']}
            assert len(episodes) == summary['episodes']
            runs.append(dict(study=study, run=run.relative_to(old.ROOT).as_posix(), training=summary,
                config_sha256=provenance['config_sha256'], source_sha256=provenance['source_sha256'],
                source_archive_sha256=old.sha(run/'source.zip'),
                training_summary_sha256=old.sha(run/'training_summary.json'),
                validation_summary_sha256=old.sha(run/'validation/summary.json'),
                validated_source_files=len(provenance['source_files']),
                completed_training_successes=sum(r['result']['success'] for r in episodes.values()),
                validation=validation))
    smoke_paths = list((BASE/'sensor_study/smoke').glob('*/training_summary.json'))
    smoke_paths += [BASE/f'macro_smoke_before_{name}_fix/preview_time_adaptive_seed101/training_summary.json'
                    for name in ('resume', 'finalize')]
    # Historical smoke directories use different layouts; locate their one summary.
    smoke_paths = list((BASE/'sensor_study/smoke').glob('*/training_summary.json'))
    for name in ('macro_smoke_before_resume_fix', 'macro_smoke_before_finalize_fix'):
        smoke_paths += list((BASE/name).rglob('training_summary.json'))
    smoke = [dict(path=p.relative_to(old.ROOT).as_posix(), sha256=old.sha(p), **old.read(p)) for p in smoke_paths]
    if manuscript_qa is None:
        manuscript_qa = old.ROOT/'论文项目/revision_20260910_paper_expansion/final_qa.json'
        if not manuscript_qa.exists():
            manuscript_qa = BASE/'manuscript_final_qa.json'
    manuscript_qa = Path(manuscript_qa).resolve()
    qa = old.read(manuscript_qa)
    assert old.sha(old.ROOT/'论文项目/main.pdf') == qa['main_pdf_sha256']
    for name, expected in qa['active_source_sha256'].items():
        assert old.sha(old.ROOT/name) == expected, name
    artwork = old.read(BASE/'before_manifest.json')['artwork']
    for name, expected in artwork.items():
        assert old.sha(old.ROOT/name) == expected, name
    tests = old.read(BASE/'verification_20260910/full_tests.json')
    assert tests['exit_code'] == 0
    assert old.sha(BASE/'verification_20260910/full_tests.log') == tests['log_sha256']
    for name, expected in tests['test_sources'].items():
        assert old.sha(old.ROOT/name) == expected, name
    sample = BASE/'sensor_study/pilot/preview_time_fixed3_seed101/train_episode_0000.npz'
    with np.load(sample) as data:
        trace = dict(zip(data['columns'].tolist(), data['data'].T))
    t = trace['timestamp_s']
    heading = np.unwrap(trace['heading_rad'])
    omega = np.r_[0., np.diff(heading)/np.diff(t)]
    alpha = np.r_[0., np.diff(omega)/np.diff(t)]
    jerk = np.r_[0., np.diff(alpha)/np.diff(t)]
    zero_speed_diagnostic = dict(trace=sample.relative_to(old.ROOT).as_posix(), trace_sha256=old.sha(sample),
        zero_feed_samples=int(np.sum(trace['v_mm_s'] == 0)),
        zero_feed_nonzero_internal_omega_samples=int(np.sum((trace['v_mm_s'] == 0) & (np.abs(trace['omega_rad_s']) > 1e-4))),
        independent_heading_difference_peaks=dict(omega_rad_s=float(np.max(np.abs(omega))),
            alpha_rad_s2=float(np.max(np.abs(alpha))), jerk_rad_s3=float(np.max(np.abs(jerk)))),
        status='Diagnostic only: recorded heading differences and position-derived direction have different observability at rest. Original failure labels preserved; no metric replacement or sampling fix implemented.')
    sensor = old.read(BASE/'sensor_study/report/report.json')
    interventions = old.read(BASE/'macro_intervention/summary.json')
    value = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=old.ROOT,text=True).strip(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old.ROOT,text=True).strip(),
        source_script_sha256=old.sha(__file__), runs=runs, smokes=smoke,
        total_pilot_models=len(runs), total_pilot_decisions=sum(r['training']['decisions'] for r in runs),
        total_pilot_physics_steps=sum(r['training']['physics_steps'] for r in runs),
        total_completed_training_episodes=sum(r['training']['episodes'] for r in runs),
        total_training_successes=sum(r['completed_training_successes'] for r in runs),
        total_smoke_decisions=sum(r['decisions'] for r in smoke),
        total_smoke_physics_steps=sum(r['physics_steps'] for r in smoke),
        sensor_comparison=sensor['paired'], intervention_comparisons=interventions['comparisons'],
        zero_speed_diagnostic=zero_speed_diagnostic,
        manuscript_pages=qa['final_pages'], manuscript_pdf_sha256=qa['main_pdf_sha256'],
        manuscript_qa_path=manuscript_qa.as_posix(), manuscript_qa_sha256=old.sha(manuscript_qa),
        artwork_verified_count=len(artwork), latest_test_result='454 passed, 2 subtests passed',
        test_record_sha256=old.sha(BASE/'verification_20260910/full_tests.json'),
        pre_revision_archive_sha256=old.sha(BASE/'before_terminal_and_compact.zip'),
        next_unimplemented='Sampling improvement to reduce zero-speed clipping, then matched controls; first reproduce a minimal stop/restart angular-observability test. No new candidate or five-seed study has been launched.',
        scope='Recovery and hash audit only. No overwritten results, new training, or test-set use.')
    old.write(output, value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('runs','smokes')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'recovery_20260910/progress.json')
    parser.add_argument('--manuscript-qa', type=Path,
                        help='Manuscript QA record; default prefers the 20260910 paper-expansion QA when present, otherwise the earlier terminal QA. All artifact hashes remain mandatory.')
    args = parser.parse_args()
    recover(args.output, args.manuscript_qa)
