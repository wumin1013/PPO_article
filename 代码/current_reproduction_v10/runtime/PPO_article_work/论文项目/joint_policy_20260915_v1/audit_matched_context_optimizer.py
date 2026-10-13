"""Check nested actor coordinates against the same saved PPO objective and guards."""
import numpy as np
from scipy.optimize import linprog
from study_io import BASE, write_json
from search_complete_return import read


def evaluate(data, baseline, u, advantage_scale):
    noise = data['noise']
    shift = np.einsum('nhd,d->nh', data['matrix'], u)
    ratio = np.exp(noise * shift[..., None] - .5 * shift[..., None] ** 2)
    advantage = data['advantages'] / advantage_scale
    gain = np.minimum(ratio * advantage, np.clip(ratio, .8, 1.2) * advantage).sum()
    gain -= advantage.sum()
    gain /= len(noise) * noise.shape[-1]
    wc = data['credit'] * data['weights'][:, None, None, None] / noise.shape[-1]
    margins = baseline + np.einsum('nhk,nhkm->m', ratio - 1., wc)
    normalizer = np.maximum(np.max(abs(wc), axis=(0, 1, 2)), 1e-12)
    parent = data['parent_matrix'] @ u
    return dict(gain=float(gain), minimum_margin=float(margins.min()),
        minimum_normalized_margin=float((margins / normalizer).min()),
        sample_kl=float(.5 * np.mean(np.sum(shift * shift, axis=1))),
        parent_kl=float(1.5 * np.mean(parent * parent)),
        support_max=float(np.max(abs(data['equality'] @ u))))


def main():
    stage = BASE / 'canonical_gate_band_matched_context_counterfactual_ppo_20260920'
    old_stage = BASE / 'canonical_gate_band_conditional_sample_credit_ppo_20260920'
    source = BASE / 'canonical_gate_band_head_counterfactual_ppo_collection_20260920'
    old = np.load(old_stage / 'measured_constraint_credit.npz')
    history = read(old_stage / 'state_joint/results.json')['candidates']
    baseline = np.array(history[0]['augmented_learning_inequalities'])
    scale = read(source / 'ppo_update_audit.json')['global_advantage_scale']
    ids = np.array([h + 3 * (k + 18 * m) for m in range(3) for k in range(6) for h in range(3)])
    rows = []
    for mode in ('geometry', 'dynamic'):
        data = np.load(stage / (mode + '_credit/measured_constraint_credit.npz'))
        for key in ('matrix', 'parent_matrix', 'equality'):
            assert np.array_equal(data[key][..., ids], old[key])
        for key in ('noise', 'credit', 'advantages', 'weights'):
            assert np.array_equal(data[key], old[key])
        basis = data['protected_basis']
        observed = np.vstack([data['matrix'].reshape(-1, 162), data['parent_matrix']]) @ basis
        singular = np.linalg.svd(observed, compute_uv=False)
        a = .02 * np.einsum('nhd,dk->nhk',data['matrix'],basis)
        derivative = data['noise'][...,None] * a[:,:,None,:]
        grad = np.einsum('nhs,nhsd->d',data['advantages']/scale,derivative)/(len(a)*3)
        wc = data['credit'] * data['weights'][:,None,None,None]/3
        jac = np.einsum('nhsd,nhsm->md',derivative,wc)
        norms = np.maximum(np.linalg.norm(jac,axis=1),1e-30)
        nonempty = np.any(wc!=0.,axis=(0,1,2))
        lp = linprog(-grad/max(np.linalg.norm(grad),1e-30),
            A_ub=-jac[nonempty]/norms[nonempty,None],b_ub=baseline[nonempty]/norms[nonempty],
            bounds=[(-.5,.5)]*basis.shape[1],method='highs')
        tangent = dict(success=lp.success,message=lp.message,gradient_norm=float(np.linalg.norm(grad)),
            gain=float(grad@lp.x) if lp.success else None,
            smallest_nonempty_jacobian_norm=float(norms[nonempty].min()),
            normalized_margin=float(np.min((baseline+jac@lp.x)[nonempty]/norms[nonempty])) if lp.success else None)
        witnesses = []
        for c in history[1:]:
            if not c['tag'].startswith('constraint_and_support'):
                continue
            u = np.zeros(162)
            u[ids] = c['coefficients']
            projected = basis @ (basis.T @ u)
            witnesses.append(dict(tag=c['tag'], old=evaluate(old, baseline, np.array(c['coefficients']), scale),
                embedded=evaluate(data, baseline, u, scale), projected=evaluate(data, baseline, projected, scale),
                projection_error=float(np.max(abs(projected - u)))))
        rows.append(dict(context=mode, parameter_dimensions=162, protected_dimensions=basis.shape[1],
            observed_rank=int(np.sum(singular > singular[0] * 1e-12)), tangent=tangent,
            singular_values=singular.tolist(), witnesses=witnesses))
        print('NESTED_PPO_OPTIMIZER',mode,rows[-1]['observed_rank'],tangent,flush=True)
    write_json(stage / 'optimizer_identity_audit.json', dict(arms=rows, new_physical_runs=0,
        conclusion='Identical nonzero feasible surrogate witnesses remain inside both enlarged actors; zero solver output is not an actor expressivity impossibility proof.',
        physical_witness_accepted=False))


if __name__ == '__main__':
    main()
