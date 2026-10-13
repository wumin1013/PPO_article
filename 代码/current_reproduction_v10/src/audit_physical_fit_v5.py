"""Audit actual fixed-budget fitting records and independently reset seed101."""
import json
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from strong_policy_v4 import StrongPolicyV4,STATIC_KEYS,load_strong_policy,tensor_hash


def main():
    out=ROOT/'results/physical_fit_v5/pilot_seed101/adaptive'
    reg=json.loads((out/'registration.json').read_text('utf-8'))
    receipt=json.loads((out/'receipt.json').read_text('utf-8'))
    assert receipt['status']=='completed' and receipt['offline_iterations']==24000
    initial,_=load_strong_policy(out/'initial.pt');final,saved=load_strong_policy(out/'final.pt')
    assets=torch.load(out/'shared_static_assets.pt',weights_only=False,map_location='cpu')
    fresh=StrongPolicyV4(101,assets,'adaptive')
    assert tensor_hash(dict(fresh.named_parameters()))==tensor_hash(dict(initial.named_parameters()))==reg['initial_trainable_tensor_sha256']
    assert sha(out/'final.pt')==sha(out/'fit_24000.pt')==receipt['model_sha256']
    assert all(sha(ROOT/'src'/n)==h for n,h in reg['source_sha256'].items())
    records=[json.loads(line) for line in (out/'losses.jsonl').read_text('utf-8').splitlines()]
    assert len(records)==24000 and [r['iteration'] for r in records]==list(range(1,24001))
    for i,r in enumerate(records):
        expected=2e-4*(.3**(i//8000))
        assert abs(r['learning_rate']-expected)<1e-18
        assert np.isfinite(r['loss']) and np.isfinite(r['gradient_norm_before_clip'])
    with np.load(out/'training_row_map.npz') as data:
        cases=data['case'];source_kind=data['source_kind'];source_row=data['source_row']
        assert len(cases)==14495
        assert int((source_kind==0).sum())==4833 and int((source_kind==1).sum())==9662
        counts=[int((cases==c).sum()) for c in ('original','dense')]
        assert counts==reg['case_samples']
    rng=np.random.default_rng(101)
    for _ in records:
        for size in counts:rng.integers(size,size=256)
    assert rng.bit_generator.state==saved['sampling_rng_state']
    first=dict(initial.named_parameters())
    changes={name:bool(not torch.equal(first[name],v)) for name,v in final.named_parameters()}
    assert len(changes)==8 and all(changes.values())
    for key in STATIC_KEYS:assert torch.equal(getattr(final.actor,key),getattr(initial.actor,key))
    result=dict(created_utc=utc(),status='passed',audit_source_sha256=sha(__file__),
        seed101_fresh_initialization_independently_reconstructed=True,old_trainable_parameters_not_copied=True,
        fixed_final24000_checkpoint_byte_identical=True,loss_records=24000,finite_losses_and_gradients=True,
        learning_rate_segment_steps=[8000,8000,8000],sampling_rng_reconstructed=True,
        sampled_presentations=24000*512,case_samples=counts,eligible_training_rows=14495,
        updated_parameter_tensors=changes,static_buffers_unchanged=True,
        model_sha256=receipt['model_sha256'],registration_sha256=sha(out/'registration.json'),
        scope='One complete supervised adaptive training run; trajectory performance is evaluated separately')
    write_json(out/'completion_validation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
