"""Current-parent quadratic energy operators from saved signed-density responses."""
import argparse
import time
import numpy as np
from energy_waveform_credit import energy_signals
from audit_xy_curvature_resolution import curvature_metrics
from search_complete_return import read
from study_io import BASE, ROOT, register, receipt, write_json, sha


ENERGY_NAMES = ('full_xy_normalized_energy', 'full_physical_normalized_energy')


def main():
    parser = argparse.ArgumentParser()
    for name in ('credit', 'source', 'stage'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--paths', type=int, nargs='+', default=list(range(16)))
    args = parser.parse_args()
    credit, source = BASE/args.credit, BASE/args.source
    assert (credit/'receipt.json').exists() and (source/'receipt.json').exists()
    raw = read(credit/'results.json'); opt = read(source/'optimizer.json')
    bindings=read(source/'registration.json')['source_sha256']
    assert bindings[(credit/'results.json').relative_to(ROOT).as_posix()]==sha(credit/'results.json')
    stage = register(args.stage, [__file__, BASE/'energy_waveform_credit.py',
        BASE/'prepare_current_waveform_credit.py', BASE/'audit_current_waveform_credit.py',
        credit/'results.json', credit/'receipt.json', source/'gradients.npz',
        source/'optimizer.json', source/'receipt.json'],
        dict(new_actor_decisions=0, new_physics_ticks=0, wall_s=180, isolated_paths=args.paths),
        'Use the current-parent complete axial consequences to linearize signed '
        'square-root energy densities, then square after composition. The nominal '
        'operator exactly reproduces both existing bending energies; unsupported '
        'derivatives are set to zero while their nominal contributions remain. '
        'This corrects an omitted positive quadratic term without changing either '
        'physical energy metric. It is a local model, not a true dynamics guarantee. '
        'Keep waveform and physical evaluation masks unchanged. No new rollouts, '
        'candidate update, development or sealed data.')
    start = time.perf_counter()
    with np.load(source/'gradients.npz') as z: jac = z['query_jacobian'].copy()
    lookup = {(r['query'], r['axis'], r['direction']):r for r in raw['records']}
    records = []; files = 0
    for b in raw['baselines']:
        if b['shape'] is None or b['path'] not in args.paths: continue
        path = b['path']
        with np.load(BASE/b['trace']) as z: ref, rows = z['reference'].copy(), z['rows'].copy()
        nominal, valid = energy_signals(ref, rows)
        common = valid.copy(); gradient = np.zeros((*nominal.shape, 768))
        for query in (q for q in raw['queries'] if q['path']==path):
            qid = query['query']
            for head in range(3):
                pair = [lookup[qid, head, sign] for sign in (-1, 1)]; signals = []
                for r in pair:
                    assert r['success'], 'Failed suffixes cannot supply complete energy derivatives'
                    with np.load(BASE/r['trace']) as z: full = np.vstack([rows[:query['anchor']*20], z['suffix_rows']])
                    signal, mask = energy_signals(ref, full, len(rows))
                    common &= mask; signals.append(signal); files += 1
                slope = (signals[1]-signals[0])/(pair[1]['latent_offset']-pair[0]['latent_offset'])
                block = slice(head*256, (head+1)*256)
                gradient[:, :, block] += slope[:, :, None]*jac[qid, head, block]
        gradient[~common] = 0.
        metric = curvature_metrics(ref, rows)
        constant = (nominal**2).sum(0)
        linear = np.einsum('tc,tcd->cd', 2*nominal, gradient)
        quadratic = np.asarray([gradient[:, c].T@gradient[:, c] for c in range(2)])
        assert max(abs(constant[c]-metric[n]) for c,n in enumerate(ENERGY_NAMES)) < 1e-10
        # Validate the compiled polynomial against the density-space definition.
        direction = np.random.default_rng(17+path).normal(size=768)*1e-4
        direct = ((nominal+np.einsum('tcd,d->tc', gradient, direction))**2).sum(0)
        compiled = constant+linear@direction+np.einsum('d,cdk,k->c', direction, quadratic, direction)
        assert np.max(abs(direct-compiled)) < 1e-10
        name = f'path{path}_energy.npz'
        np.savez_compressed(stage/name, constant=constant, linear=linear, quadratic=quadratic,
            nominal=nominal, nominal_valid=valid, common_valid=common, metric_names=np.asarray(ENERGY_NAMES))
        records.append(dict(path=path, file=name, excluded_derivative_samples=(valid.sum(0)-common.sum(0)).tolist(),
            nominal_identity_error=float(max(abs(constant[c]-metric[n]) for c,n in enumerate(ENERGY_NAMES)))))
        print('ENERGY_CREDIT', path, 'files', files, flush=True)
        assert time.perf_counter()-start < 180
    assert sorted(r['path'] for r in records)==sorted(args.paths)
    write_json(stage/'results.json', dict(parent_sha256=opt['parent_sha256'], records=records,
        metric_names=ENERGY_NAMES, suffix_files=files, actual_dynamics_bound=False, goal_complete=False))
    receipt(stage, dict(status='prepared_current_quadratic_density_energy', new_actor_decisions=0,
        new_physics_ticks=0, suffix_files=files, wall_s=time.perf_counter()-start))


if __name__ == '__main__': main()
