"""Locate remaining shape errors in saved actual XY and velocity traces."""
import math
import numpy as np
from joint_runtime import FILTER_TAU, wrap
from study_io import BASE, write_json, sha
from search_complete_return import read


def intervals(mask):
    padded = np.r_[False, mask, False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)))


def characterize(file):
    with np.load(file) as z:
        ref, rows, actions = [z[k].copy() for k in ('reference', 'rows', 'actions')]
    u, v = np.diff(ref, axis=0)
    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    sign = np.sign(u[0]*v[1]-u[1]*v[0])
    angle = abs(math.atan2(u[0]*v[1]-u[1]*v[0], u@v))
    normal = sign*np.array([-v[1], v[0]])
    out = (rows[:, :2]-ref[1])@normal
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    omega = sign*(rows[:, 2]*rows[:, 5]-rows[:, 3]*rows[:, 4])/np.maximum(speed**2, 1e-20)
    omega[speed <= 5.] = 0.
    apex = int(np.linalg.norm(rows[:, :2]-ref[1], axis=1).argmin())
    heading = np.arctan2(rows[:, 3], rows[:, 2])
    origin = math.atan2(u[1], u[0])
    heading = sign*np.array([wrap(h-origin) for h in heading])
    filtered = np.arctan2(rows[:, 9], rows[:, 8])
    previous = np.r_[origin, filtered[:-1]]
    target = previous+rows[:, 10]*.001/(1-math.exp(-.001/FILTER_TAU))
    target = sign*np.array([wrap(h-origin) for h in target])
    ordinary = rows[:, 12] == 0
    jumps = np.abs(np.diff(target))
    jumps[~(ordinary[1:] & ordinary[:-1])] = 0
    negative = []
    for lo, hi in intervals(omega < -math.radians(.5)):
        reverse = float(np.degrees(np.maximum(-omega[lo:hi], 0).sum()*.001))
        if reverse < .005:
            continue
        negative.append(dict(start_ms=int(lo), stop_ms=int(hi), duration_ms=int(hi-lo),
            reverse_deg=reverse, speed_min=float(speed[lo:hi].min()),
            speed_max=float(speed[lo:hi].max())))
    return dict(trace_sha256=sha(file), apex_ms=apex, outgoing_min_ms=int(out.argmin()),
        outgoing_min_mm=float(out.min()), outward_gt005_ms=int((out < -.005).sum()),
        negative_turn_intervals=negative, heading_max_deg=float(np.degrees(heading[speed>5].max())),
        outgoing_heading_deg=float(np.degrees(angle)),
        target_jumps=[dict(time_ms=int(i+1), magnitude_deg=float(np.degrees(jumps[i])),
                          held_before=actions[min(i//20, len(actions)-1)].tolist(),
                          held_after=actions[min((i+1)//20, len(actions)-1)].tolist())
                      for i in np.argsort(jumps)[-5:][::-1]],
        preapex_reverse_deg=float(np.degrees(np.maximum(-omega[:apex], 0).sum()*.001)),
        postapex_reverse_deg=float(np.degrees(np.maximum(-omega[apex:], 0).sum()*.001)))


def main():
    stage = BASE/'canonical_gate_band_neighborhood_teacher_state_distillation_20260919'
    results = read(stage/'results.json')
    candidate = next(c for c in results['candidates'] if c['tag'] == results['best'])
    output = []
    for path in [0, 3, 4, 14, 104, 105]:
        r = next(r for r in candidate['records'] if r['path'] == path)
        result = characterize(stage/r['trace'])
        output.append(dict(path=path, **result))
        print('SHAPE_LOCALIZATION', path, result['apex_ms'], result['outgoing_min_ms'],
              result['outgoing_min_mm'], result['negative_turn_intervals'], flush=True)
    write_json(stage/'remaining_shape_localization.json', dict(records=output,
        method='Read stored raw1ms XY,V,A and requested-heading records; no new physical rollout.',
        goal_complete=False))


if __name__ == '__main__':
    main()
