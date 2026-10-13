"""Corrected historical cases under the common original-geometry error budget."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from baselines.study_v3 import run_one
from common import write_json,sha,utc
from runtime_bootstrap import setup
setup()
from runtime_configuration import configure_runtime


def main():
    out=ROOT/'results/baseline_v3/development_unified';out.mkdir(parents=True,exist_ok=True)
    selections=json.loads((ROOT/'results/baseline_v3/selection.json').read_text('utf-8'))['selection']
    registration=dict(created_utc=utc(),case_order=['original','dense'],methods=['S2','S2R','SCMO'],dt_s=[.001,.0005],
        source_sha256=sha(Path(__file__)),policy='Nominal original tolerance 1.25 mm; shared polygon budget subtracts largest certified circular sagitta. Complete trajectories and failures retained.')
    p=out/'registration.json'
    if not p.exists():write_json(p,registration)
    for case in registration['case_order']:
        folder=ROOT/'baseline_records/supplementary_reproduction/cases'/case
        ref=np.loadtxt(folder/'reference.csv',delimiter=',',skiprows=1)
        geometry=json.loads((folder/'geometry.json').read_text('utf-8'))
        bound=max(x.get('chord_error_bound_mm',0.) for x in geometry['primitives'])
        tolerance=1.25-bound;configure_runtime('v3',tolerance=tolerance)
        for dt in registration['dt_s']:
            for method in registration['methods']:
                scale=selections[method]['selected']['corner_scale'] if method!='SCMO' else 1.
                run_one(ref,case,method,scale,dt,out,tolerance=tolerance,source_geometry=geometry)


if __name__=='__main__':main()
