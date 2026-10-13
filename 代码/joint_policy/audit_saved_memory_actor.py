"""Verify a learned memory actor against saved execution, without new rollouts."""
import argparse
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from joint_teacher_state_policy import features
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args();stage=BASE/args.stage;torch.set_num_threads(1)
    model=torch.load(stage/'research_candidate.pt',map_location='cpu',weights_only=False)
    paths=read(stage/'support_audit.json')['selected']['paths']
    result=read(stage/'results.json');chosen=next(c for c in result['candidates'] if c['tag']==result['best'])
    records=[r for r in chosen['records'] if r['path'] in paths]
    replay=collect_data(model,model['parent'],records,include_latents=True)
    replay['held_actions']=np.vstack([np.load(r['trace'])['actions'] for r in records])
    proof=SavedMemoryActionIdentity(model,replay,GateBandRuntime.decode_action)
    assert all(proof.unchanged(r['path'],model['coefficients']) for r in records)
    batch=features(model,replay['observations'],replay['innovations'])
    scalar=np.array([features(model,x,y) for x,y in zip(replay['observations'],replay['innovations'])])
    assert np.array_equal(batch,scalar)
    np.savez_compressed(stage/'selected_actor_replay.npz',**replay)
    write_json(stage/'selected_actor_identity_audit.json',dict(model_sha256=sha(stage/'research_candidate.pt'),
        paths=paths,decisions=len(replay['decisions']),actual_policy_decoded_actions_exact=True,
        scalar_batch_features_exact=True,helper_nonzero_actions_exact=True,new_physical_runs=0))
    print('LEARNED_ACTOR_REPLAY_EXACT',len(replay['decisions']),sha(stage/'research_candidate.pt'))


if __name__=='__main__':main()
