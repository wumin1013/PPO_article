"""Read-only checks of the registered seed101 second fitting stage."""
import json
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from strong_policy_v4 import STATIC_KEYS, load_strong_policy, tensor_hash


def main():
    torch.set_num_threads(1)
    output=ROOT/'results/final_function_fit_v4/stage2_seed101/adaptive'
    registration=json.loads((output/'registration.json').read_text('utf-8'))
    receipt=json.loads((output/'receipt.json').read_text('utf-8'))
    parent_path=ROOT/registration['parent']
    data_path=ROOT/registration['teacher_data']
    parent,parent_saved=load_strong_policy(parent_path)
    initial,initial_saved=load_strong_policy(output/'initial.pt')
    final,final_saved=load_strong_policy(output/'final.pt')
    losses=[json.loads(line) for line in (output/'losses.jsonl').read_text('utf-8').splitlines()]
    steps=np.array([row['iteration'] for row in losses])
    rates=np.array([row['lr'] for row in losses])
    assert receipt['status']=='completed'
    assert len(losses)==12000 and np.array_equal(steps,np.arange(1,12001))
    assert np.all(rates[:8000]==6e-5) and np.all(rates[8000:]==6e-5*.3)
    assert all(np.isfinite(row[key]) for row in losses for key in ('loss','gradient_norm_before_clip'))
    assert final_saved['stage_iterations']==12000 and final_saved['offline_iterations']==24000
    assert final_saved['registration_sha256']==sha(output/'registration.json')
    assert sha(parent_path)==registration['parent_sha256']==receipt['parent_sha256']
    assert sha(data_path)==registration['teacher_data_sha256'] and receipt['dataset_unchanged']
    assert sha(output/'final.pt')==receipt['model_sha256']
    assert sha(output/'losses.jsonl')==receipt['losses_sha256']
    assert all(sha(ROOT/'src'/name)==value for name,value in registration['source_sha256'].items())
    assert tensor_hash(dict(parent.named_parameters()))==tensor_hash(dict(initial.named_parameters()))
    parent_params=dict(parent.named_parameters())
    changed={name:bool(not torch.equal(value,parent_params[name])) for name,value in final.named_parameters()}
    assert len(changed)==8 and all(changed.values())
    unchanged={key:bool(torch.equal(parent_saved['policy']['actor.'+key],final_saved['policy']['actor.'+key])) for key in STATIC_KEYS}
    assert all(unchanged.values())
    with np.load(data_path) as data:
        observations=data['observations'].copy()
        groups=[np.flatnonzero((data['case']==case)&(data['neighborhood_sign']==0 if kind=='original' else data['neighborhood_sign']!=0))
                for case in ('original','dense') for kind in ('original','neighbor')]
        assert len(observations)==20547
        assert int((data['neighborhood_sign']==0).sum())==6849
        assert [len(group) for group in groups]==[5370,10740,1479,2958]
    rng=np.random.default_rng(registration['rng_seed'])
    for _ in losses:
        for group in groups:
            rng.integers(len(group),size=128)
    assert rng.bit_generator.state==final_saved['sampling_rng_state']
    loaded,_=load_strong_policy(output/'final.pt')
    with torch.no_grad():
        reload_error=max(float((loaded.mean(part)-final.mean(part)).abs().max())
                         for part in np.array_split(observations,41))
    assert reload_error==0
    checkpoint,_=load_strong_policy(output/'fit_12000.pt')
    assert tensor_hash(dict(checkpoint.named_parameters()))==tensor_hash(dict(final.named_parameters()))
    result=dict(checked_utc=utc(),status='passed',source_sha256=sha(__file__),
        checkpoint_sha256=sha(output/'final.pt'),stage_iterations=12000,pipeline_iterations=24000,
        losses=12000,learning_rate_counts={'0.00006':8000,'0.000018':4000},
        initial_equals_parent=True,updated_parameter_tensors=changed,static_buffers_unchanged=unchanged,
        group_order=['original_source','original_neighbor','dense_source','dense_neighbor'],
        group_row_counts=[len(group) for group in groups],draws_per_group=128*12000,
        total_sample_draws=512*12000,sampling_rng_reconstructed=True,
        all_observations_reload_max_error=reload_error,final_equals_fixed12000_checkpoint=True,
        parent_and_teacher_and_registered_sources_unchanged=True,
        scope='Fitting implementation audit only; no trajectory success, independent seed, or final-test claim')
    write_json(output/'completion_validation.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
