"""Complete-only, signed metric response with a training-time function trust region."""
import numpy as np
import torch
from scipy.optimize import minimize


def signed_deficits(candidate, retained):
    if not all(r['success'] for r in candidate['records']):
        raise ValueError('Incomplete prefixes cannot define a complete-policy response derivative')
    old = {r['path']: r for r in retained}
    values = []
    for r in candidate['records']:
        baseline = old[r['path']]
        values.append(0.)
        if 'shape' in r:
            s, b = r['shape'], baseline['baseline_shape']
            for key, scale in [('in_edge_outward_mm', .002), ('out_edge_outward_mm', .005),
                               ('full_physical_reverse_deg', .5), ('full_normalized_energy', .02),
                               ('full_spatial_peak_curvature_mm_inv', .2)]:
                values.append((s[key]-b[key])/scale)
            values.append((b['apex_speed_mm_s']-s['apex_speed_mm_s'])/2.)
            values.append(10.*(r['target_motion']['first_turn_inside'] is not True))
        base_time = baseline['complete_time_s']-baseline['complete_time_change_s']
        values.append((r['complete_time_s']-base_time)/.02)
    values.append((3.-candidate['summary']['curved_cruise99_s'])*5.)
    return np.asarray(values)


def state_function_features(policy, observation):
    with torch.no_grad():
        x = torch.as_tensor(observation, dtype=torch.float32)
        if hasattr(policy, 'function_features'):
            return policy.function_features(x).numpy()
        odd, even = policy.hidden(x)
        f0 = (odd@policy.odd_projection)/policy.odd_scale
        f1 = ((even-policy.even_mean)@policy.even_projection)/policy.even_scale
        f1 = torch.cat([f1, torch.ones_like(f1[..., :1])], -1)
        matrix = torch.zeros((len(x), 3, 14))
        matrix[:, 0, :4] = .01*f0
        matrix[:, 1, 4:9] = .05*f1
        matrix[:, 2, 9:14] = .05*f1
    return matrix.numpy()


def complete_metric_response(current, paired, retained, exclude_failed_directions=True):
    base = signed_deficits(current, retained)
    columns, bounds, modes, nonlinearities = [], [], [], []
    for lo, hi in paired:
        low_good = all(r['success'] for r in lo['records'])
        high_good = all(r['success'] for r in hi['records'])
        if low_good and high_good:
            low, high = signed_deficits(lo, retained), signed_deficits(hi, retained)
            columns.append((high-low)/2.)
            bounds.append((-1., 1.))
            modes.append('central_complete')
            nonlinearities.append(float(np.linalg.norm(high+low-2*base)/max(np.linalg.norm(high-low), 1e-9)))
        elif low_good:
            columns.append(base-signed_deficits(lo, retained))
            bounds.append((-1., 0.) if exclude_failed_directions else (-1., 1.))
            modes.append('negative_complete_positive_excluded' if exclude_failed_directions else 'negative_complete_metric_side')
            nonlinearities.append(None)
        elif high_good:
            columns.append(signed_deficits(hi, retained)-base)
            bounds.append((0., 1.) if exclude_failed_directions else (-1., 1.))
            modes.append('positive_complete_negative_excluded' if exclude_failed_directions else 'positive_complete_metric_side')
            nonlinearities.append(None)
        else:
            columns.append(np.zeros_like(base))
            bounds.append((0., 0.))
            modes.append('both_failed_no_metric_derivative')
            nonlinearities.append(None)
    matrix = np.column_stack(columns)
    return base, matrix, bounds, modes, nonlinearities


def solve_local_response(current, paired, retained, features, steps):
    base, matrix, bounds, modes, nonlinearities = complete_metric_response(current, paired, retained)
    # This bounds a parameter proposal on current observations; no runtime clipping.
    scale = np.array([.0025, .0125, .0125])
    trust = (features*steps[None, None, :]/scale[None, :, None]).reshape(-1, len(steps))
    def objective(u):
        positive = np.maximum(base+matrix@u, 0.)
        return float(positive@positive+.01*(u@u)), 2*matrix.T@positive+.02*u
    solution = minimize(objective, np.zeros(len(steps)), jac=True, method='SLSQP', bounds=bounds,
        constraints=[dict(type='ineq', fun=lambda u: np.r_[1-trust@u, 1+trust@u],
                          jac=lambda u: np.vstack([-trust, trust]))],
        options=dict(maxiter=200, ftol=1e-8))
    u = solution.x
    if not solution.success or abs(trust@u).max() > 1.+1e-6:
        raise RuntimeError('Local trust solver did not converge: '+solution.message)
    predicted = float(np.maximum(base+matrix@u, 0.)@np.maximum(base+matrix@u, 0.))
    return steps*u, predicted, dict(response_modes=modes, nonlinearity_ratios=nonlinearities,
        solver_message=solution.message, predicted_objective=predicted,
        maximum_normalized_anchor_change=float(abs(trust@u).max()),
        function_change_limits_latent=scale.tolist(), response_rank=int(np.linalg.matrix_rank(matrix)))
