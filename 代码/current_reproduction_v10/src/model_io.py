"""Version-aware loading; model bytes and their registration travel together."""
import json
from pathlib import Path
from types import MethodType
import torch
from paired_learning import Policy
from common import sha


def _bounded_forward(self,x):
    z=(x[...,None,:]-self.centers)/self.scale
    f=torch.exp(-.5*z.square().sum(-1)/self.radii.square())
    return self.offset+self.gain*torch.tanh(self.base(x)+self.correction(f))


def load_model(path):
    path=Path(path)
    registration=path.with_name('registration.json')
    if not registration.exists():raise ValueError('Model loading requires its version/provenance registration.json')
    reg=json.loads(registration.read_text('utf-8'))
    if path.name=='final.pt':
        receipt=json.loads(path.with_name('receipt.json').read_text('utf-8'))
        if sha(path)!=receipt['model_sha256']:raise ValueError('Checkpoint hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['policy']
    policy=Policy(saved['seed'],state['actor.centers'].numpy(),state['actor.scale'].numpy(),state['actor.radii'].numpy(),saved['d'],saved['arm'])
    if 'action_capacity' not in reg:
        policy.actor.forward=MethodType(_bounded_forward,policy.actor)
        parameterization='bounded_tanh_calibration_pilot'
    elif reg['action_capacity'].startswith('Unbounded'):
        parameterization='unbounded_mean_full_physical_action_range'
    else:raise ValueError('Unrecognized actor parameterization registration')
    policy.load_state_dict(state);policy.eval();saved['actor_parameterization']=parameterization
    return policy,saved
