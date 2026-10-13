"""Compare linear and nonlinear motion context using identical measured responses."""
import numpy as np
from scipy.optimize import linprog
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    source = BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    model = read(source/'joint_phase_credit_models.json')
    history = read(source/'state_joint/results.json')['candidates']
    support = read(source/'support_audit.json')
    paths = support['selected']['paths']
    ctx = np.asarray(support['physical_contexts'])
    # Nonlinear interactions of observed speed and tangential acceleration only.
    basis = np.concatenate([ctx, (ctx[:, 1]*ctx[:, 2])[:, None],
                            (ctx[:, 1]**2)[:, None], (ctx[:, 2]**2)[:, None]], axis=1)
    jac = np.asarray(model['base_constraints'])
    g = np.asarray(model['base_per_path_loss_gradient'])
    c0 = np.asarray(history[0]['measured_inequalities'])
    owners = np.r_[np.repeat(np.arange(len(paths)), 11), np.repeat(np.arange(len(paths)), 16)]
    output = []
    for count in (3, 4, 6):
        matrix = np.stack([jac*basis[owners, None, c, :, None] for c in range(count)], axis=2).reshape(len(c0), -1)
        gradient = np.sum(np.stack([g*basis[:, None, c, :, None] for c in range(count)], axis=2), axis=0).reshape(-1)
        scales = np.maximum(np.maximum(abs(matrix).max(1), abs(c0)), 1e-12)
        solved = linprog(gradient, A_ub=-matrix/scales[:, None], b_ub=c0/scales,
                         bounds=[(-1., 1.)]*(18*count), method='highs',
                         options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
        ranks = [int(np.linalg.matrix_rank(basis[[p in support['phase_support_paths'][k] for p in paths], :count, k])) for k in (0, 1)]
        row = dict(context_count=count, parameters=18*count, context_ranks=ranks,
                   success=bool(solved.success), predicted_loss_change=float(solved.fun) if solved.success else None,
                   scaled_proposal=solved.x.tolist() if solved.success else None,
                   predicted_guard_minimum=float((c0+matrix@solved.x).min()) if solved.success else None)
        if solved.success:
            proposal = solved.x.reshape(3, count, 2, 3)
            row['pre_component_norm'] = float(np.linalg.norm(proposal[:, :, 0]))
            row['post_component_norm'] = float(np.linalg.norm(proposal[:, :, 1]))
        output.append(row)
        print('MOTION_CONTEXT_CAPACITY', {k: v for k, v in row.items() if k != 'scaled_proposal'}, flush=True)
    write_json(source/'nonlinear_motion_context_capacity.json', dict(
        source_sha256=sha(source/'joint_phase_credit_models.json'), paths=paths,
        basis_order=['1', 'speed', 'tangential_acceleration', 'speed_x_tangential_acceleration',
                     'speed_squared', 'tangential_acceleration_squared'],
        context=basis.tolist(), models=output, new_physical_runs=0,
        interpretation='Same measured local responses; context rank and predicted credit only. '
                       'No independent trajectory validation, learned-policy acceptance or generalization is implied.'))


if __name__ == '__main__':
    main()
