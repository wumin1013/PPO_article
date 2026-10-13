"""Isolated sampler regression with declared scalar phases, independent of tests.

The mock geometry is a straight line with artificial constant-feed sections.
It deliberately isolates sampling from the Sencer geometric construction.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import write_json, utc, sha
from baselines import ruckig_s2 as adapter
from baselines.legacy_s2 import prepare_geometry


def main():
    base = prepare_geometry(np.array([[0., 0.], [10., 0.]]))
    rows = []
    native_prepare = adapter.prepare_geometry
    try:
        for short in (0., 1e-8, 1e-5):
            length = 4. + short
            curve = SimpleNamespace(straight_lengths_mm=[1., short, 1.],
                curve_lengths_mm=[1., 1.], curve_speed_caps_mm_s=[10., 10.],
                length=length, metadata={'fixture': 'isolated scalar sampler'},
                position=lambda sigma: np.column_stack((sigma, np.zeros_like(sigma))))
            adapter.prepare_geometry = lambda *args, **kwargs: (curve, *base[1:])
            for dt in (.001, .0005):
                positions, meta = adapter.plan(np.array([[0., 0.], [length, 0.]]), dt=dt)
                phases = meta['execution']['scheduled_phases']
                empty = sum(p['kind'] == 'ruckig_straight' and
                    np.ceil(p['start_s']/dt) == np.ceil((p['start_s']+p['duration_s'])/dt)
                    for p in phases)
                passed = bool(np.isfinite(positions).all() and
                    np.all(np.diff(positions[:, 0]) >= -1e-10) and
                    np.linalg.norm(positions[-1]-[length, 0.]) < 1e-10 and
                    (short == 0. or empty >= 1))
                rows.append(dict(short_phase_mm=short, dt_s=dt,
                    empty_sampled_straights=int(empty), passed=passed,
                    samples=len(positions)))
    finally:
        adapter.prepare_geometry = native_prepare
    out = dict(created_utc=utc(), timing='Additional isolated check after v3b freeze; no algorithm or parameter changed',
        source_sha256=sha(Path(__file__)), adapter_sha256=sha(ROOT/'src/baselines/ruckig_s2.py'),
        fixture_scope='Sampling only; no claim to test geometric smoothing', checks=rows)
    write_json(ROOT/'literature/ruckig_validation/empty_phase_isolated_v11.json', out)
    print(rows)
    assert all(r['passed'] for r in rows)


if __name__ == '__main__':
    main()
