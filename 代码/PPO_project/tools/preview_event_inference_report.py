"""Create a data-only table from the saved serial actor benchmark."""
import json
from pathlib import Path
import statistics

import numpy as np

import preview_event_study as study
from preview_event_report import LABELS


def main():
    source = study.OUT / 'serial_inference/summary.json'
    raw = study.read(source)
    if study.sha(source.parent / 'latencies.npz') != raw['samples_sha256']:
        raise ValueError('Timing sample hash mismatch')
    frozen = study.read(study.OUT / 'freeze.json')
    if raw['freeze_sha256'] != study.sha(study.OUT / 'freeze.json'):
        raise ValueError('Timing freeze mismatch')
    with np.load(source.parent / 'latencies.npz') as values:
        for row in raw['rows']:
            key = f"{row['variant']}_seed{row['seed']}"
            measured = values[key]
            if len(measured) != row['calls'] or not np.isfinite(measured).all() or (measured <= 0).any():
                raise ValueError(f'Invalid raw timings: {key}')
            if not np.allclose([np.median(measured), np.percentile(measured, 95)],
                               [row['median_us'], row['p95_us']], atol=1e-10, rtol=1e-12):
                raise ValueError(f'Timing summary mismatch: {key}')
            if row['checkpoint_sha256'] != frozen['models']['formal/' + key] or row['config_sha256'] != frozen['configs']['formal/' + key]:
                raise ValueError(f'Timing model/config mismatch: {key}')
    rows = []
    for variant in study.FORMAL:
        samples = [r for r in raw['rows'] if r['variant'] == variant]
        if sorted(r['seed'] for r in samples) != list(study.SEEDS):
            raise ValueError('Incomplete independent-model timing set')
        rows.append(dict(variant=variant, models=len(samples),
            mean_model_median_us=statistics.mean(r['median_us'] for r in samples),
            sd_model_median_us=statistics.stdev(r['median_us'] for r in samples),
            mean_model_p95_us=statistics.mean(r['p95_us'] for r in samples)))
    lines = [r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Serial CPU actor timing over five trained models per method. Each model uses 200 warm-up calls and 2000 measured calls. The median column reports mean $\pm$ sample SD across model medians; the final column averages the model-wise 95th percentiles.}',
        r'\label{tab:event-inference}', r'\begin{tabular}{lrr}\toprule',
        r'Method & Model median ($\mu$s) & Model P95 ($\mu$s)\\\midrule']
    for row in rows:
        lines.append(f"{LABELS[row['variant']]} & {row['mean_model_median_us']:.2f} $\\pm$ {row['sd_model_median_us']:.2f} & {row['mean_model_p95_us']:.2f} "+r'\\')
    lines.extend([r'\bottomrule\end{tabular}', r'\end{table}'])
    table = study.ROOT / '论文项目/generated/preview_event_inference.tex'
    table.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    result = dict(rows=rows, scope=raw['scope'], input_sha256=study.sha(source),
        source_samples_sha256=raw['samples_sha256'], script_sha256=study.sha(__file__),
        table=table.relative_to(study.ROOT).as_posix(), table_sha256=study.sha(table),
        note='Same registered model order; uncontrolled host activity. No method speed superiority or full-cycle real-time claim.')
    study.write(source.parent / 'report.json', result)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
