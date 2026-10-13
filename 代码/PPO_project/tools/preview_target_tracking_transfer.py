"""Evaluation-only execution ablation of the selected full-batch actor.

Copy the full selected checkpoint byte-for-byte. Add only feed tracking plus
explicit parent provenance to its evaluation config. Training provenance and
evaluation-environment provenance are separate. Never train or access holdouts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import yaml

import preview_target_tracking_study as target

ROOT = target.full.core.old.ROOT
OUT = ROOT/'论文项目/revision_20260910_continued/target_tracking_transfer_v1'
SELECTION = target.BASE_OUT/'selection_budget4096.json'
EXPECTED_MODEL_SHA = 'f93eaf7944546833d8d71c19f9ec84f212b5e83fcaafa9a7f550a5367cabfb68'
old = target.full.core.old


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def parent_record():
    selection = read_json(SELECTION)
    selected = next(row['selected'] for row in selection['records'] if row['selected']['method'] == 'adaptive')
    parent = ROOT/selected['checkpoint']
    if selected['step'] != 256 or selected['checkpoint_sha256'] != EXPECTED_MODEL_SHA:
        raise ValueError('This transfer is registered for the selected full-batch adaptive step256 only')
    manifest = read_json(parent/'snapshot_manifest.json')
    for name, digest in manifest['files'].items():
        if old.sha(parent/name) != digest:
            raise ValueError(f'Parent snapshot artifact changed: {name}')
    if old.sha(parent/'checkpoint.pt') != EXPECTED_MODEL_SHA:
        raise ValueError('Parent checkpoint changed')
    if old.sha(parent/'validation/summary.json') != selected['validation_sha256']:
        raise ValueError('Parent selection/validation mismatch')
    provenance = read_json(parent/'provenance.json')
    summary = read_json(parent/'training_summary.json')
    if summary['decisions'] != 256 or summary['checkpoint_sha256'] != EXPECTED_MODEL_SHA:
        raise ValueError('Parent training summary mismatch')
    cfg = yaml.safe_load((parent/'config.yaml').read_text(encoding='utf-8'))
    metadata = dict(parent_snapshot=parent.relative_to(ROOT).as_posix(),
        parent_checkpoint_sha256=EXPECTED_MODEL_SHA,
        parent_config_sha256=old.sha(parent/'config.yaml'),
        parent_training_source_sha256=provenance['source_sha256'],
        parent_training_provenance_sha256=old.sha(parent/'provenance.json'),
        parent_training_summary_sha256=old.sha(parent/'training_summary.json'),
        parent_validation_sha256=selected['validation_sha256'],
        parent_selection_file=SELECTION.relative_to(ROOT).as_posix(),
        parent_selection_sha256=old.sha(SELECTION), parent_selection_available_budget=selection['budget'],
        parent_selected_training_decisions=256, parent_pretraining_macro_decisions=summary['pretraining_macro_decisions'],
        additional_training_decisions=0, additional_optimizer_updates=0,
        transfer_role='Same frozen actor execution ablation; not independently retrained target_tracking_v1')
    return parent, cfg, metadata


def transfer_configuration(parent_cfg, metadata):
    cfg = copy.deepcopy(parent_cfg)
    if 'feed_tracking' in cfg['revision'] or 'evaluation_transfer' in cfg['revision']:
        raise ValueError('Parent already has a transfer change')
    cfg['revision']['feed_tracking'] = target.configuration('preview_time', 'adaptive', cfg['seed'])['revision']['feed_tracking']
    cfg['revision']['evaluation_transfer'] = copy.deepcopy(metadata)
    return cfg


def source_bindings():
    if not (target.OUT/'protocol.json').exists():
        raise ValueError('Freeze the target-tracking evaluation environment first')
    protocol = target.prepare()  # Existing immutable target protocol, verified.
    evaluation_source = protocol['source']
    transfer_files = dict(evaluation_source['source_files'])
    transfer_files[Path(__file__).relative_to(ROOT).as_posix()] = old.sha(Path(__file__))
    transfer_source_hash = hashlib.sha256(json.dumps(transfer_files, sort_keys=True).encode()).hexdigest()
    return evaluation_source, transfer_files, transfer_source_hash


def prepare(output=OUT):
    output = Path(output).resolve()
    parent, parent_cfg, metadata = parent_record()
    evaluation_source, transfer_files, transfer_hash = source_bindings()
    cfg = transfer_configuration(parent_cfg, metadata)
    provenance = dict(**evaluation_source, **metadata,
        provenance_role='Evaluation environment only; original training provenance copied separately',
        source_binding_scope='source_sha256 means target-tracking evaluation environment, not model training source',
        evaluation_source_sha256=evaluation_source['source_sha256'],
        evaluation_protocol_sha256=old.sha(target.OUT/'protocol.json'),
        transfer_tool_source_sha256=transfer_hash,
        transfer_tool_file_sha256=old.sha(Path(__file__)),
        command='python '+' '.join(sys.argv), data_role='Development validation execution ablation',
        test_access=False)
    if output.exists():
        verify(output)
        return output
    output.mkdir(parents=True)
    (output/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    provenance['config_sha256'] = old.sha(output/'config.yaml')
    # Keep the entire model/critic/optimizer/RNG/config payload byte-identical.
    for source_name, dest_name in (
        ('checkpoint.pt', 'checkpoint.pt'), ('config.yaml', 'parent_config.yaml'),
        ('provenance.json', 'parent_training_provenance.json'),
        ('training_summary.json', 'parent_training_summary.json'), ('source.zip', 'parent_training_source.zip'),
        ('snapshot_manifest.json', 'parent_snapshot_manifest.json'),
        ('validation/summary.json', 'parent_validation_summary.json')):
        shutil.copyfile(parent/source_name, output/dest_name)
    shutil.copyfile(SELECTION, output/'parent_selection.json')
    write_json(output/'provenance.json', provenance)
    write_json(output/'transfer_sources.json', dict(source_sha256=transfer_hash, source_files=transfer_files))
    with zipfile.ZipFile(output/'evaluation_and_transfer_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in transfer_files:
            archive.write(ROOT/name, name)
    manifest = {path.name: old.sha(path) for path in output.iterdir() if path.is_file()}
    write_json(output/'transfer_manifest.json', dict(files=manifest, **metadata,
        checkpoint_is_byte_identical_copy=True, evaluation_source_sha256=evaluation_source['source_sha256'],
        transfer_tool_source_sha256=transfer_hash))
    verify(output)
    return output


def verify(output=OUT):
    output = Path(output).resolve()
    manifest = read_json(output/'transfer_manifest.json')
    for name, digest in manifest['files'].items():
        if old.sha(output/name) != digest:
            raise ValueError(f'Transfer artifact changed: {name}')
    parent, parent_cfg, metadata = parent_record()
    evaluation_source, _, transfer_hash = source_bindings()
    cfg = yaml.safe_load((output/'config.yaml').read_text(encoding='utf-8'))
    if cfg != transfer_configuration(parent_cfg, metadata):
        raise ValueError('Transfer changed numerical configuration beyond the registered feed correction')
    if old.sha(output/'checkpoint.pt') != old.sha(parent/'checkpoint.pt'):
        raise ValueError('Transfer model differs from parent; no training is authorized')
    if manifest['evaluation_source_sha256'] != evaluation_source['source_sha256'] or manifest['transfer_tool_source_sha256'] != transfer_hash:
        raise ValueError('Transfer source changed')
    return dict(output=str(output), checkpoint_sha256=old.sha(output/'checkpoint.pt'),
        parent_model_unchanged=True, additional_training_decisions=0,
        evaluation_source_sha256=evaluation_source['source_sha256'], transfer_tool_source_sha256=transfer_hash)


def comparison_rows(before, after):
    baseline = {row['path_id']: row for row in before}
    if len(after) != len(baseline) or {row['path_id'] for row in after} != set(baseline):
        raise ValueError('Same complete validation path set required')
    rows = []
    for row in after:
        parent = baseline[row['path_id']]
        paired = bool(parent['success'] and row['success'])
        rows.append(dict(path_id=row['path_id'], before_success=parent['success'], after_success=row['success'],
            before_time_s=parent['elapsed_time_s'], after_time_s=row['elapsed_time_s'],
            time_comparable=paired,
            relative_time_change_pct=(row['elapsed_time_s']/parent['elapsed_time_s']-1)*100 if paired else None,
            before_error_max_mm=parent['contour_error_max_mm'], after_error_max_mm=row['contour_error_max_mm'],
            before_failure_reasons=parent['failure_reasons'], after_failure_reasons=row['failure_reasons']))
    return rows


def evaluate(output=OUT):
    output = prepare(output)
    if (output/'summary.json').exists():
        raise FileExistsError('Preserve completed transfer evaluation')
    provenance = read_json(output/'provenance.json')
    # The frozen validator binds the target environment source. The original
    # model training source is recorded separately above and in every new row.
    with target.installed():
        if not (output/'raw_validation/summary.json').exists():
            target.full.core.evaluate(output, output/'raw_validation')
    raw = read_json(output/'raw_validation/summary.json')
    parent = read_json(output/'parent_validation_summary.json')['rows']
    expected = {path['id'] for path in old.manifest()['paths'] if path['split'] == 'validation'}
    if len(raw['rows']) != len(expected) or {row['path_id'] for row in raw['rows']} != expected:
        raise ValueError('The full registered validation set is required')
    rows = []
    for original in raw['rows']:
        path_id = original['path_id']
        trace = output/'raw_validation'/f'{path_id}.npz'
        for field, expected_value in (('checkpoint_sha256', EXPECTED_MODEL_SHA),
            ('config_sha256', provenance['config_sha256']), ('source_sha256', provenance['evaluation_source_sha256']),
            ('trace_sha256', old.sha(trace))):
            if original[field] != expected_value:
                raise ValueError(f'Raw evaluation binding mismatch: {field}')
        row = dict(original, **{key: provenance[key] for key in (
            'parent_snapshot', 'parent_checkpoint_sha256', 'parent_config_sha256', 'parent_training_source_sha256',
            'parent_training_provenance_sha256', 'parent_training_summary_sha256', 'parent_validation_sha256',
            'parent_selection_sha256', 'parent_selection_available_budget', 'parent_selected_training_decisions',
            'parent_pretraining_macro_decisions', 'additional_training_decisions', 'additional_optimizer_updates',
            'evaluation_source_sha256', 'transfer_tool_source_sha256')},
            data_role='Same selected actor, evaluation-only feed-execution ablation; development validation',
            evaluation_command='python '+' '.join(sys.argv), trace_file=str(trace.relative_to(output)),
            raw_evaluation_row_sha256=old.sha(output/'raw_validation'/f'{path_id}.json'))
        write_json(output/'rows'/f'{path_id}.json', row); rows.append(row)
    paired = comparison_rows(parent, rows)
    all_complete = all(row['before_success'] and row['after_success'] for row in paired)
    result = dict(rows=rows, paired_comparison=paired,
        before_success_count=sum(row['success'] for row in parent),
        after_success_count=sum(row['success'] for row in rows), count=len(rows),
        all_paths_complete_before_and_after=all_complete,
        before_total_time_s=sum(row['elapsed_time_s'] for row in parent) if all_complete else None,
        after_total_time_s=sum(row['elapsed_time_s'] for row in rows) if all_complete else None,
        additional_training_decisions=0, additional_optimizer_updates=0,
        parent_model_sha256=EXPECTED_MODEL_SHA,
        evaluation_source_sha256=provenance['evaluation_source_sha256'],
        transfer_tool_source_sha256=provenance['transfer_tool_source_sha256'],
        observation='Execution correction ablation on one selected seed101 actor; not an independently trained method comparison or holdout evidence')
    verify(output)
    write_json(output/'summary.json', result)
    print(json.dumps({key: result[key] for key in ('before_success_count', 'after_success_count',
        'before_total_time_s', 'after_total_time_s', 'additional_training_decisions', 'paired_comparison')},
        ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'evaluate', 'verify'))
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare(args.output_dir))
    elif args.command == 'evaluate': evaluate(args.output_dir)
    else: print(json.dumps(verify(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
