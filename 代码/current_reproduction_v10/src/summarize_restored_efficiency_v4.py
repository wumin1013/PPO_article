"""Bind restored full trajectories to every available common-condition baseline."""
import csv
import json
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json

PRIOR = Path(r'D:\PPO_article_work\JNN_revision_20261011_v3')


def valid(record):
    return (record['success'] and record['kinematic_pass'] and record['rest_pass']
        and record['geometry']['geometry_complete']
        and record['geometry']['gates_passed'] == record['geometry']['gate_count'])


def main():
    records, hashes = [], {}
    for case in ('original', 'dense'):
        for dt, label in ((.001, '1000'), (.0005, '500')):
            learned_file = (ROOT / 'results/v4b_no_target_override/strong' / (case+'.json') if dt == .001
                else ROOT / 'results/strong_checkpoint_v4/v4e_resolution_control' / (case+'_dt500us.json'))
            learned = json.loads(learned_file.read_text('utf-8'))
            reference = np.load(learned_file.with_suffix('.npz'))['reference']
            candidates = []
            for method in ('S2', 'S2R', 'SCMO'):
                path = PRIOR / 'results/baseline_v3/development_unified' / f'{method}_{case}_dt{label}us_scale1000.json'
                candidates.append((method, json.loads(path.read_text('utf-8')), path, path.with_suffix('.npz')))
            if case == 'dense':
                path = ROOT / 'results/copp_development_v4e/dense.json'
                result = json.loads(path.read_text('utf-8'))
                final = next(i for i in result['iterations'] if i['iteration'] == 3)
                run = next(r for r in final['runs'] if r['dt_s'] == dt)
                candidates.append(('COPP TOPP3-SOCP adapter', run['metrics'], path, path.parent / run['trace']))
            for method, baseline, result_file, trace in candidates:
                assert valid(learned) and valid(baseline)
                with np.load(trace) as z:
                    assert np.array_equal(reference, z['reference'])
                assert learned['dt_s'] == baseline['dt_s'] == dt
                assert learned['limits'] == baseline['limits'] == [100., 2000., 62500.]
                assert abs(learned['tolerance_mm']-baseline['tolerance_mm']) < 1e-12
                t, b = learned['complete_time_s'], baseline['complete_time_s']
                records.append(dict(case=case, dt_s=dt, comparator=method, jnnc_s=t,
                    comparator_s=b, saving_s=b-t, saving_percent=100*(b-t)/b,
                    all_constraints_pass=True, tolerance_mm=learned['tolerance_mm'],
                    model_sha256=learned['model_sha256']))
                for item in (learned_file, learned_file.with_suffix('.npz'), result_file, trace):
                    hashes[str(item)] = sha(item)
    with (ROOT / 'audit/restored_efficiency_v4.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(ROOT / 'audit/restored_efficiency_v4.json', dict(created_utc=utc(),
        source_sha256=sha(__file__), rows=records, evidence_sha256=hashes,
        scope='Previously used complete development programs; one fixed historical model, two numerical resolutions. This is not a five-seed independent final test.',
        comparison='Reference arrays, tolerance including arc chord reservation, vector bounds, full command-interval geometry, ordered completion and final rest checked from complete saved executions.',
        runtime='1ms v4b and .5ms v4e are explicitly recorded configurations; both preserve the original model tensors. v4e adds fully verified terminal lookahead and fixes physical guide width at0.1mm.',
        copp='Unmodified official COPP0.2.3 TOPP3-SOCP solver with documented path/vector/time adapters; third iteration fixed. Does not claim Li2023 or IJMTM2026 full-paper reproduction.',
        new_simulation_performed=False))
    md = ['# Restored main-model efficiency', '',
          'These are complete executions of the same historical model. All shown comparisons passed the common acceptance contract.', '',
          '| Program | dt (ms) | Comparator | J-NNC (s) | Comparator (s) | Saving (%) |',
          '|---|---:|---|---:|---:|---:|']
    for r in records:
        md.append(f"| {r['case']} | {1000*r['dt_s']:g} | {r['comparator']} | {r['jnnc_s']:.4f} | {r['comparator_s']:.4f} | {r['saving_percent']:.4f} |")
    md += ['', 'The two numerical resolutions are paired numerical checks, not independent training replicates. Fresh fitting and PPO trials have separate identities and are not pooled with these times.', '',
           'The v3 low-speed figure used a different architecture/training configuration. Restoring the original actor recovered its recorded performance. An additional v3 guide-target rule had also changed its requests; removing that premature straight-ray substitution restored both original programs.', '',
           'Reproduce this table: `D:\\Anaconda\\envs\\PPO\\python.exe src/summarize_restored_efficiency_v4.py` from the v4 work directory. This command verifies saved records; it does not rerun simulation.', '']
    (ROOT / 'audit/restored_efficiency_v4.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md[:20]))


if __name__ == '__main__':
    main()
