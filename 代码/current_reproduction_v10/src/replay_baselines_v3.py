"""Repeat frozen deterministic final cells in a fresh evidence directory."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT,sha,write_json,utc


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'results') or output.exists():raise ValueError('Use a new folder inside results')
    frozen=json.loads((ROOT/'results/baseline_v3/final_freeze_v3b.json').read_text('utf-8'))
    for name,digest in frozen['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen baseline input changed '+name)
    from baselines.study_v3 import run_one
    from runtime_configuration import configure_runtime
    selection=json.loads((ROOT/'results/baseline_v3/selection.json').read_text('utf-8'))['selection']
    registry=json.loads((ROOT/'prospective/final_v3/registration.json').read_text('utf-8'))
    output.mkdir(parents=True);rows=[]
    for path in registry['paths']:
        reference=np.loadtxt(ROOT/'prospective/final_v3'/path['file'],delimiter=',',skiprows=1)
        configure_runtime('v3',tolerance=path['evaluation_tolerance_mm'])
        for dt in [.001,.0005]:
            for method in ['S2','S2R','SCMO']:
                scale=selection[method]['selected']['corner_scale'] if method in selection else 1.
                result=run_one(reference,path['id'],method,scale,dt,output,tolerance=path['evaluation_tolerance_mm'])
                rows.append(dict(path=path['id'],method=method,dt_s=dt,success=result['success'],complete_time_s=result['complete_time_s'],trace=result['trace']))
    write_json(output/'rerun_receipt.json',dict(created_utc=utc(),freeze_sha256=sha(ROOT/'results/baseline_v3/final_freeze_v3b.json'),
        expected=30,recorded=len(rows),source_sha256=sha(__file__),all_registered_outcomes_retained=True,records=rows))


if __name__=='__main__':main()
