"""Targeted fixed-grid check on a complete 1000 mm rest-to-rest straight."""
from pathlib import Path
import math
import numpy as np
import copp_path_adapter_v4 as core
import copp_geometry_only_v4 as frozen
import copp_adapter_v5b as current

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'results/copp_v5b_straight_grid_check'
    if output.exists():
        raise FileExistsError('Preserve every previous grid check')
    output.mkdir(parents=True)
    reference = np.asarray([[0., 0.], [1000., 0.]])
    np.savetxt(output/'reference.csv', reference, delimiter=',', header='x_mm,y_mm', comments='')
    settings = [dict(current.SETTINGS, max_line_cruise_spacing_mm=.5), current.SETTINGS]
    labels = ['uniform_half_mm', 'dense_ends_cruise5mm']
    core.write(output/'registration.json', dict(scope='Synthetic development grid sensitivity; no final programs.',
        method='Same signed geometry and COPP 0.2.3 TOPP3-SOCP; only interior straight spacing differs.',
        settings=dict(zip(labels, settings)), source_sha256=core.sha(__file__),
        adapter_sha256=core.sha(current.__file__), reference_sha256=core.sha(output/'reference.csv'),
        final_iteration=3, common_ideal_jerk_limited_time_s=1000/100+100/2000+2000/62500,
        ideal_time_scope='Analytic lower bound for this straight under common V/A/J, excludes terminal hold and solver reserves.'))
    core.prepare_geometry = current.prepare_geometry
    core.grid_geometry = current.grid_geometry
    core.sample = frozen.projected_sample
    results = [core.run_case(label, reference, 1.25, None, output, settings=setting)
               for label, setting in zip(labels, settings)]
    core.write(output/'summary.json', results)
    for row in results:
        print(row['path_id'], row['status'], [(r['dt_s'], r['metrics']['success'], r['metrics']['complete_time_s'])
              for r in row['iterations'][-1]['runs']])


if __name__ == '__main__':
    main()
