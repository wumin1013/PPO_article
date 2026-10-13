"""Unchanged two-step planner, audited once per newly frozen holdout path."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import zipfile

import preview_formal_two_step as original

ROOT = original.ROOT
FORMAL = ROOT/'论文项目/revision_20260910_continued/target_tracking_formal_v1'
OUT = FORMAL/'two_step_original_v1'


def access_gate():
    import preview_target_tracking_holdout
    return preview_target_tracking_holdout


def prepare():
    formal_path = FORMAL/'protocol.json'
    formal_protocol = original.read(formal_path)
    if formal_protocol['version'] != 'target-tracking-formal-v1':
        raise ValueError('Require the newly registered two-phase formal protocol')
    cfg = original.configuration()
    if (cfg['dt_s'], cfg['one_sided_tolerance_mm'], cfg['max_steps'], cfg['witness_spacing_mm']) != (.001, 1.25, 60000, .01):
        raise ValueError('Original baseline must retain the registered common tolerances and budget')
    source = original.source_state(formal_protocol)
    for name in ('PPO_project/tools/preview_target_tracking_two_step.py',
                 'PPO_project/tests/test_preview_target_tracking_two_step.py'):
        if formal_protocol['evaluation_sources']['source_files'].get(name) != original.sha(ROOT/name):
            raise ValueError('Baseline evaluator was not frozen in the formal protocol')
    value = dict(version='target-tracking-original-two-step-v1', configuration=cfg, source=source,
        formal_protocol_file=formal_path.relative_to(ROOT).as_posix(),
        formal_protocol_sha256=original.sha(formal_path), holdout_manifest=formal_protocol['holdout_manifest'],
        algorithm='Original preview_formal_two_step.run_one -> preview_two_step_eval.run_two_step_baseline',
        parameters='Original default SmoothingParams and SchedulerParams, identical to the previous original two-step audit',
        output_policy='Every original command retained; declared initial t=0 point only; no appended rest, clipping or repair',
        audit='Same1.25mm tolerance,0.01mm ordered witnesses, six position-derived constraints, original full20ms rest policy',
        budget='60000 original1ms samples; no additional terminal commands',
        model_gate='New target_tracking_holdout.validate_freeze must pass all ten new models before manifest/coordinates',
        comparison_rule='Time comparisons only on common complete feasible paths; never infer efficiency from shorter failed trajectories',
        statistical_unit='One deterministic planner trajectory per path; copying it across PPO seeds creates no new baseline replication',
        preparation_reads_holdout_coordinates=False, trained=False, machine_execution_authorized=False)
    path = OUT/'protocol.json'
    if path.exists():
        previous = original.read(path)
        value['registered_utc'] = previous['registered_utc']
        if previous != value:
            raise ValueError('Preserve registered original baseline protocol')
    else:
        value['registered_utc'] = datetime.now(timezone.utc).isoformat()
        original.write_new(path, value)
    archive_path = OUT/'source.zip'
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in source['source_files']:
                archive.write(ROOT/name, name)
    with zipfile.ZipFile(archive_path) as archive:
        for name, digest in source['source_files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Original planner source archive changed')
    return value


def evaluate(freeze_path):
    protocol = prepare()
    gate = access_gate()
    checked = gate.validate_freeze(freeze_path, ROOT)
    signature, cfg = checked['common_configuration'], protocol['configuration']
    if (signature['environment']['interpolation_period'] != cfg['dt_s']
            or signature['environment']['epsilon']/2 != cfg['one_sided_tolerance_mm']
            or signature['environment']['max_steps'] != cfg['max_steps']
            or signature['constraints'] != cfg['limits']):
        raise ValueError('Original baseline and both PPO methods must share evaluation constraints')
    if checked['freeze']['protocol']['sha256'] != protocol['formal_protocol_sha256']:
        raise ValueError('Baseline was registered for another formal protocol')
    manifest, data = gate.frozen_manifest(checked, ROOT)
    from generate_preview_continuation_paths import load_path
    binding = dict(protocol_sha256=original.sha(OUT/'protocol.json'),
        freeze_sha256=checked['freeze_sha256'], holdout_manifest_sha256=checked['freeze']['holdout_manifest']['sha256'])
    rows = []
    for item in manifest['paths']:
        rows.append(original.run_one(item['id'], load_path(item['id'], data), item['family'],
            OUT/'test'/item['id'], protocol, binding, 'Original deterministic baseline on twelve newly frozen paths'))
    value = dict(binding=binding, rows=rows, count=len(rows), successes=sum(r['success'] for r in rows),
        deterministic_runs_per_path=1, all_failures_retained=True, comparison_rule=protocol['comparison_rule'])
    path = OUT/'test/summary.json'
    if path.exists():
        if original.read(path) != original._json_value(value):
            raise ValueError('Preserve completed original baseline summary')
    else:
        original.write_new(path, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate'))
    parser.add_argument('--freeze', type=Path, default=FORMAL/'freeze.json')
    args = parser.parse_args()
    result = prepare() if args.command == 'prepare' else evaluate(args.freeze)
    print(dict(status='complete', command=args.command, count=result.get('count'), successes=result.get('successes')))
