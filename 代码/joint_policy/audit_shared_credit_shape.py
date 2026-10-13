"""Read complete current trajectories: objective and physical turning locations."""
import time
import numpy as np
from calibrate_shared_network_credit import vector
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def main():
    source = BASE/'canonical_gate_band_fast_absolute_shape_joint_credit_20260919'
    data = read(source/'results.json')
    current = next(c for c in data['candidates'] if c['tag'] == data['best'])
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    stage = register('canonical_gate_band_shared_credit_shape_localization_20260919',
        [__file__, BASE/'calibrate_shared_network_credit.py', source/'results.json', source/'receipt.json'],
        dict(new_physics_ticks=0, wall_s=30),
        'Locate current full-path reversal, target jumps and remaining objective costs from '
        'saved actual XY/velocity/acceleration. Separate entry, main turn, exit approach and '
        'terminal braking; do not infer causation from temporal coincidence or change acceptance.')
    start = time.perf_counter()
    values = vector(current, retained)
    offset, output = 0, []
    keys = ['completion', 'entry', 'exit', 'reverse', 'energy', 'peak_curvature', 'apex_speed', 'first_turn', 'time']
    for r in current['records']:
        n = 9 if 'shape' in r else 2
        cost = dict(zip(keys if n == 9 else ['completion', 'time'], np.maximum(values[offset:offset+n], 0).tolist()))
        offset += n
        if 'shape' not in r: continue
        with np.load(source/r['trace']) as z:
            ref, rows = z['reference'], z['rows']
        u, v = np.diff(ref, axis=0); u /= np.linalg.norm(u); v /= np.linalg.norm(v)
        sign = np.sign(u[0]*v[1]-u[1]*v[0]); angle = np.arccos(np.clip(u@v, -1., 1.))
        speed = np.linalg.norm(rows[:, 2:4], axis=1)
        heading = sign*np.unwrap(np.arctan2(rows[:, 2]*u[1]-rows[:, 3]*u[0], rows[:, 2:4]@u))*(-1)
        ordinary = rows[:, 12] == 0
        omega = sign*(rows[:, 2]*rows[:, 5]-rows[:, 3]*rows[:, 4])/np.maximum(speed, 1e-9)**2
        reverse = np.maximum(-omega, 0)*.001*180/np.pi; reverse[speed <= 5] = 0
        phases = dict(entry=ordinary & (heading < .1*angle),
            turn=ordinary & (heading >= .1*angle) & (heading <= .9*angle),
            exit=ordinary & (heading > .9*angle), braking=~ordinary)
        phase_reverse = {k: float(reverse[mask].sum()) for k, mask in phases.items()}
        assert abs(sum(phase_reverse.values())-r['shape']['full_physical_reverse_deg']) < 1e-9
        row = dict(path=r['path'], turn_deg=r['shape']['turn_deg'], cost_components=cost,
            sum_squared_cost=sum(x*x for x in cost.values()), reversal_by_heading_phase_deg=phase_reverse,
            shape=r['shape'], target_motion=r['target_motion'])
        output.append(row)
        print('SHAPE_LOCATION', r['path'], row['sum_squared_cost'], phase_reverse,
              'reverse/energy/apex', [round(cost[k], 3) for k in ('reverse', 'energy', 'apex_speed')], flush=True)
    write_json(stage/'results.json', dict(records=output, physical_steps=0, adopted=False, goal_complete=False))
    receipt(stage, dict(status='completed_saved_shared_credit_shape_localization',
                        wall_s=time.perf_counter()-start, physical_steps=0, adopted=False, goal_complete=False))


if __name__ == '__main__': main()
