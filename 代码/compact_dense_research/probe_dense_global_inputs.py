"""One full-loop global-triplet masking diagnostic, not a trained reduced actor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from evaluate_historical_dense import ROOT, HIST, CASE, sha, write, verify_bound_inputs
from candidate_policy_factory import candidate_policy
from turn_aware_projector import environment_factory
from local_arc_support import install_local_arc_support, ClusteredPreviewFloorPolicy
from paper_complete_evaluation import evaluate
from natural_case_audit import analytic_command_audit

STAGE=ROOT/'historical_dense_global_triplet_probe'


class GlobalTripletProbe:
    def __init__(self,base):
        self.base=base
    def bind(self,env):
        return self.base.bind(env)
    def act(self,observation,deterministic=True):
        x=np.asarray(observation).copy()
        x[[16,17,19]]=0.
        return self.base.act(x,deterministic)


def worker():
    verify_bound_inputs();torch.set_num_threads(1)
    selection=json.loads((HIST/'selected_policy.json').read_text(encoding='utf-8'))
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    base,_=install_local_arc_support(candidate_policy(metadata,selection))
    policy=GlobalTripletProbe(ClusteredPreviewFloorPolicy(base,3.))
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    env=environment_factory(reference)
    result=evaluate(policy,env,STAGE/'trace.npz',time.perf_counter()+240)
    rows=np.load(STAGE/'trace.npz')['rows']
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    result['analytic']=analytic_command_audit(reference[0],rows,geometry)
    if not result['analytic']['analytic_corridor_pass']:
        result['success']=False;result['complete_time_s']=None;result['motion_scope']='failure_prefix_only'
    result.update(goal_complete=False,ppo_updates=0,mask_indices=[16,17,19],
        scope='Full-loop masking diagnostic; no retraining and no dimensional-reduction acceptance.')
    write(STAGE/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    if '--worker' in sys.argv:
        worker();return
    verify_bound_inputs();STAGE.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),STAGE/'source.py')
    write(STAGE/'registration.json',dict(kind='Global progress,gate ordinal and normalized clock input diagnostic',
        wall_budget_s=300,child_timeout_s=270,physics_deadline_s=240,ppo_updates=0,
        mask_indices=[16,17,19],mask_values=[0.,0.,0.],
        reference_sha256=sha(CASE/'reference.csv'),historical_selection_sha256=sha(HIST/'selected_policy.json'),
        source_sha256=sha(Path(__file__)),
        comparison=str(ROOT/'historical_dense_cluster_preview_probe/floor_3/result.json'),
        adoption='Use to identify observation dependence only; full feasibility is required for complete time. Not a newly trained69-dimensional controller.',
        stop='On physical failure or270s child limit. Keep all inherited causal memory and local inputs intact.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=270)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:
            status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-5000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-5000:])


if __name__=='__main__':main()
