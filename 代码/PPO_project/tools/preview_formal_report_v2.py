"""Export the complete frozen evaluation with distinct group and scope fields.

The registered numerical evaluator/statistics remain byte-identical. This
standalone exporter fixes only the collision between CSV grouping metadata and
the paired estimator's explanatory `scope` field.
"""
from pathlib import Path
import argparse
import json
import sys

import preview_formal_holdout as original
import preview_formal_statistics as statistics


def grouped_rows(stats):
    methods, pairs = [], []
    groups = [('overall', 'all', stats['overall'])]
    groups.extend((group, label, value)
                  for group in ('by_family', 'by_seed', 'by_path')
                  for label, value in stats[group].items())
    for group, label, value in groups:
        methods.extend(dict(group_scope=group, label=label, method=method, **value[method])
                       for method in original.METHODS)
        pairs.append(dict(group_scope=group, label=label, **value['paired']))
    return methods, pairs


def build_report(freeze_path, output):
    gate = original.validate_freeze(freeze_path)
    manifest, _ = original.frozen_manifest(gate)
    output = Path(output).resolve()
    original.require(output.is_relative_to(gate['freeze_path'].parent / 'reports')
                     and not output.exists(), 'Choose a new formal/reports directory')
    root = original.ROOT
    rows, inputs = [], {gate['freeze_path'].relative_to(root).as_posix(): gate['freeze_sha256']}
    for model in gate['models']:
        record = model['record']
        directory = gate['freeze_path'].parent / 'holdout' / f"{record['method']}_seed{record['seed']}"
        for item in manifest['paths']:
            path = directory / f"{item['id']}.json"
            original.require(path.is_file(), f'All60 outcomes including failures are required: {path}')
            row = original.read_json(path)
            expected = dict(freeze_sha256=gate['freeze_sha256'], seed=record['seed'],
                method=record['method'], path_id=item['id'], family=item['family'],
                checkpoint_sha256=record['artifacts']['checkpoint.pt']['sha256'],
                config_sha256=record['artifacts']['config.yaml']['sha256'],
                source_sha256=gate['freeze']['source']['source_sha256'],
                holdout_manifest_sha256=gate['freeze']['holdout_manifest']['sha256'],
                selected_step=record['selected_step'])
            original.require(all(row.get(key) == value for key, value in expected.items()),
                             f'Outcome/frozen model identity mismatch: {path}')
            for field in ('trace', 'inference_samples'):
                member = original.bound_file(root, row[field])
                inputs[member.relative_to(root).as_posix()] = original.sha(member)
            inputs[path.relative_to(root).as_posix()] = original.sha(path)
            rows.append(row)
    stats = statistics.summarize(rows, list(original.SEEDS),
                                 {path['id']: path['family'] for path in manifest['paths']})
    method_rows, pairs = grouped_rows(stats)
    # Finish validation/rendering before creating any report artifacts.
    for name, digest in inputs.items():
        original.require(original.sha(root / name) == digest, f'Input changed during export: {name}')
    original.write_json_new(output / 'report.json', dict(freeze_sha256=gate['freeze_sha256'],
        rows=rows, statistics=stats,
        scope='All60 frozen model/path outcomes including failures; independent unit is training seed, '
              'with path pairing and whole-seed clustered confidence intervals. '
              'No trajectory sample is an independent training replicate.'))
    original.write_csv_new(output / 'raw_metrics.csv', rows)
    original.write_csv_new(output / 'method_statistics.csv', method_rows)
    original.write_csv_new(output / 'paired_statistics.csv', pairs)
    names = ('report.json', 'raw_metrics.csv', 'method_statistics.csv', 'paired_statistics.csv')
    sources = (Path(__file__), Path(original.__file__), Path(statistics.__file__))
    original.write_json_new(output / 'manifest.json', dict(inputs=inputs,
        outputs={name: original.sha(output / name) for name in names},
        export_source_files={str(path.resolve()): original.sha(path) for path in sources},
        command=[sys.executable, *sys.argv],
        export_change='Use group_scope for CSV grouping, preserving paired scope text and all numerical values'))
    return dict(output=str(output), rows=len(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=original.FORMAL / 'freeze.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.freeze, args.output), ensure_ascii=False))
