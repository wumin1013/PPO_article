"""Replay the frozen main controller into a new directory without external files."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, write_json
from frozen_policy import FrozenPolicy
from runtime_physical_time_v5 import install
from evaluate import rollout


class Actor:
    def __init__(self, checkpoint):
        self.policy=FrozenPolicy(checkpoint)

    def act(self, observation, deterministic=True):
        return self.policy.mean(observation).numpy(),None,None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('original','dense'),required=True)
    parser.add_argument('--dt',type=float,choices=(.001,.0005),default=.001)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=ROOT/'evidence/frozen_main_v6'
    manifest=json.loads((data/'manifest.json').read_text('utf-8'))
    item=next(r for r in manifest['runs'] if r['case']==args.case and r['method']=='JNN' and r['dt_s']==args.dt)
    checkpoint=data/manifest['main_model']
    assert sha(checkpoint)==manifest['model_sha256']
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    reference=np.loadtxt(data/'programs'/args.case/'reference.csv',delimiter=',',skiprows=1)
    torch.set_num_threads(1)
    settings=install(dt=args.dt,tolerance=item['metrics']['tolerance_mm'])
    output=args.output/(item['id']+'.npz')
    result=rollout(Actor(checkpoint),reference,output,dt=args.dt,
        tolerance=item['metrics']['tolerance_mm'],wall_s=2400.,runtime_version='v3')
    with np.load(data/item['trace']) as expected,np.load(output) as actual:
        required={'reference','positions','rows','observations','actions'}
        assert set(expected.files)==set(actual.files) and required.issubset(actual.files)
        equality={name:bool(np.array_equal(expected[name],actual[name])) for name in expected.files}
    receipt=dict(case=args.case,dt_s=args.dt,success=result['success'],
        complete_time_s=result['complete_time_s'],model_sha256=sha(checkpoint),
        reference_trace_sha256=item['trace_sha256'],replay_trace_sha256=sha(output),
        arrays_equal=equality,all_arrays_equal=all(equality.values()),settings=settings)
    write_json(args.output/'replay_receipt.json',receipt)
    print(json.dumps(receipt,indent=2))
    if not result['success'] or not receipt['all_arrays_equal']:raise SystemExit(1)


if __name__=='__main__':main()
