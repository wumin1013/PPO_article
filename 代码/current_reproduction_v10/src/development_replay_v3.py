"""Replay unified final seed 101 on previously observed illustrative programs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import numpy as np
import torch
from common import ROOT,write_json,sha
from model_io import load_model
from evaluate import rollout


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['original','dense','asymmetric'],required=True)
    p.add_argument('--arm',choices=['adaptive','fixed'],required=True);p.add_argument('--dt',type=float,default=.001)
    a=p.parse_args();torch.set_num_threads(1)
    file=(ROOT/'prospective/reserved_programs/reserve_asymmetric_profile.csv' if a.case=='asymmetric' else
          ROOT/'baseline_records/supplementary_reproduction/cases'/a.case/'reference.csv')
    ref=np.loadtxt(file,delimiter=',',skiprows=1)
    model=ROOT/'results/unified_v3/paired/seed_101'/a.arm/'final.pt';actor,saved=load_model(model)
    out=ROOT/'results/development_v3/final_model'/f'{a.arm}_101_{a.case}_{a.dt}.npz'
    bound=.0019913804853999295 if a.case=='original' else 0.
    result=rollout(actor,ref,out,dt=a.dt,tolerance=1.25-bound,runtime_version='v3',
        fixed_d=saved['d'] if a.arm=='fixed' else None,wall_s=1680.)
    result.update(method=a.arm,seed=101,path=a.case,dt_s=a.dt,scope='Previously observed development illustration; not independent test evidence',
        model_sha256=sha(model),reference_sha256=sha(file),original_tolerance_mm=1.25,arc_chord_error_bound_mm=bound)
    write_json(out.with_suffix('.json'),result)


if __name__=='__main__':main()
