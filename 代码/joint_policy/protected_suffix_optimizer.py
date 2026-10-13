"""Numerical entry-prefix constraints and reusable joint suffix actor optimization."""
import numpy as np
import torch
from scipy.linalg import null_space
from deep_joint_credit_policy import credit_normalization
from local_suffix_credit_policy import LocalSuffixCreditPolicy, local_basis
from learn_prefix_preserving_suffix_credit import prefix_indices


def fit_actor(parent, baseline, source, queries, amplitude, arm, steps=250):
    support, protected = [], []
    for r in baseline['records']:
        with np.load(source/r['trace']) as z:
            ref, rows, obs = [z[k].copy() for k in ('reference', 'rows', 'observations')]
        support.append(obs); protected.append(obs[prefix_indices(ref, rows, obs)])
    support, protected = np.vstack(support), np.vstack(protected)
    qobs = np.array([q['observation'] for q in queries], np.float32)
    mean, std = credit_normalization(support); centers, radius = local_basis(qobs, mean, std)
    p = LocalSuffixCreditPolicy(parent, mean, std, centers, radius, precision='float64')
    fq, fa, fp = [p.features(torch.tensor(x)).detach() for x in (qobs, support, protected)]
    transforms = [torch.tensor(null_space(fp[:, :, h].numpy(), rcond=1e-12), dtype=torch.float64)
                  if arm == 'joint' or h == 1 else torch.zeros(len(qobs), 0, dtype=torch.float64) for h in range(3)]
    dimensions = [t.shape[1] for t in transforms]
    g = torch.tensor([q['gradient'] for q in queries], dtype=torch.float64)
    hessian = torch.zeros_like(g)
    for i, q in enumerate(queries):
        for head in range(3):
            lo, hi = q['records'][2*head:2*head+2]
            if lo['success'] and hi['success']:
                hessian[i, head] = max(0., (hi['absolute_loss']+lo['absolute_loss']-2*q['baseline_absolute_loss'])/amplitude[head]**2)
    weights = torch.tensor([q['stratum_size'] for q in queries], dtype=torch.float64)[:, None]
    limits = torch.tensor([.003, .025, .025], dtype=torch.float64)
    normalization = (abs(g)*weights*limits).sum()
    assert normalization > 0 and sum(dimensions) > 0
    z = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(n, dtype=torch.float64)) for n in dimensions])
    optimizer = torch.optim.Adam(z, lr=.003)
    coefficients = lambda: torch.stack([transforms[h]@z[h] for h in range(3)], -1)
    for _ in range(steps):
        c = coefficients(); delta = (fq*c).sum(-2)*p.output_scale; anchor = (fa*c).sum(-2)*p.output_scale
        loss = (weights*(g*delta+.5*hessian*delta.square())).sum()/normalization+.02*(anchor/limits).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            maximum = float(abs((fa*coefficients()).sum(-2)*p.output_scale/limits).max())
            if maximum > 1:
                for value in z: value.div_(maximum)
    with torch.no_grad(): p.coefficients.copy_(coefficients())
    maximum_error = max(float(abs(p.act(x, True)[0]-p.base.act(x, True)[0]).max()) for x in protected)
    assert maximum_error <= 1e-12, maximum_error
    with torch.no_grad():
        delta = p.correction(torch.tensor(qobs))
        linear = float((weights*g*delta).sum())
        quadratic = float(.5*(weights*hessian*delta.square()).sum())
    md = dict(architecture='local_suffix_credit', parent=parent, mean=mean, std=std, centers=centers, radius=radius,
        precision='float64', coefficients=p.coefficients.detach().numpy().copy(), step_scale=1.,
        environment_architecture='gate_band_canonical_kernel', learning_method='current_prefix_quadratic_suffix_'+arm)
    audit = dict(arm=arm, free_dimensions=dimensions, protected_states=len(protected),
        maximum_protected_latent_difference=maximum_error, predicted_linear_change=linear,
        predicted_quadratic_change=quadratic, optimizer_steps=steps, learning_rate=.003,
        action_limits=limits.tolist(), nonnegative_second_difference_count=int((hessian > 0).sum()))
    return md, audit
