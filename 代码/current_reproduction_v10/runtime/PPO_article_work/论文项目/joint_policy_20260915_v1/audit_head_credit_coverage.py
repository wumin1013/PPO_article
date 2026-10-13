"""Read exploration and sampled-credit coverage without new rollouts."""
import numpy as np
import argparse
import torch
from observable_memory_joint_policy import actor_features
from joint_teacher_state_policy import OUTPUT_SCALE
from head_counterfactual_rollouts import exploration_sigma
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from study_io import BASE, write_json
from search_complete_return import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_head_counterfactual_ppo_collection_20260920')
    args = parser.parse_args()
    source = BASE/args.stage
    md = torch.load(source/'initial_policy.pt',weights_only=False)
    rows = []
    for e in read(source/'on_policy_episodes.json')['episodes']:
        with np.load(e['record']['trace']) as z:
            obs = z['observations'].copy()
        memory = np.zeros((2,6,3))
        previous = np.zeros((6,3))
        alpha = np.array(md['memory_decays'])[:,None,None]
        norms = []
        for x in obs:
            f = actor_features(md,x,np.zeros(3))
            memory = alpha*memory+(1-alpha)*previous
            previous = f.copy()
            sigma = exploration_sigma(md,np.concatenate([f,*memory],axis=0)*OUTPUT_SCALE)
            norms.append(np.linalg.norm(sigma))
        norms = np.array(norms)
        queries = np.array([s['decision'] for s in e['samples']])
        row = dict(path=e['path'],seed=e['seed'],supported_decisions=int(np.count_nonzero(norms)),
            query_decisions=queries.tolist(),peak_decision=int(norms.argmax()),peak_sigma=float(norms.max()),
            query_peak_fraction=float(norms[queries].max()/norms.max()),
            query_energy_fraction=float(np.sum(norms[queries]**2)/np.sum(norms**2)))
        rows.append(row)
        print('QUERY_COVERAGE',row,flush=True)
    replay_file = BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920/affected_replay.npz'
    with np.load(replay_file) as z:
        replay = {k:z[k].copy() for k in z.files}
    identity = SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    supported = []
    for p,(f,memory,_,_) in identity.paths.items():
        design = np.concatenate([f,memory[:,0],memory[:,1]],axis=1)*OUTPUT_SCALE
        norms = np.linalg.norm(exploration_sigma(md,design),axis=-1)
        order = np.argsort(norms)[::-1]
        energy = np.cumsum(norms[order]**2)/np.sum(norms**2)
        support = dict(path=p,total_decisions=len(f),nonzero=int(np.count_nonzero(norms)),
            count_99pct_energy=int(np.searchsorted(energy,.99)+1),
            count_999pct_energy=int(np.searchsorted(energy,.999)+1),peak_decision=int(norms.argmax()))
        supported.append(support)
        print('DETERMINISTIC_SUPPORT',support,flush=True)
    write_json(source/'query_coverage_audit.json',dict(episodes=rows,parent_support=supported,
        new_physical_runs=0,energy_is_design_magnitude_not_a_proven_credit_importance_measure=True))


if __name__=='__main__':
    main()
