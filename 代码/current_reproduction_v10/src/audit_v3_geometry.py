"""Regression checks for retained gate geometry, interval audit and initialization."""
import numpy as np
from common import ROOT,write_json,utc,sha
from runtime_configuration import configure_runtime


def main():
    configure_runtime('v3')
    from geometry_spec import Corridor,GateTracker
    from compact_farvector_runtime import FarVectorRuntime
    file=ROOT/'prospective/reserved_programs/reserve_asymmetric_profile.csv'
    ref=np.loadtxt(file,delimiter=',',skiprows=1);env=FarVectorRuntime(ref)
    checks=[]
    for label,positions in [('reference',ref),('shortcut',ref[[0,-1]]),('reversed',ref[::-1]),('unfinished',ref[:len(ref)//2])]:
        tracker=GateTracker(Corridor(ref),positions[0])
        for point in positions[1:]:tracker.update(point)
        checks.append(dict(name=label,**tracker.snapshot(1e-8)))
    assert checks[0]['geometry_complete'] and not any(x['geometry_complete'] for x in checks[1:])
    assert env.tracker.frontier==0
    # Both endpoints can be inside a U corridor while their connecting command
    # leaves its capsule union. The interval test must reject that shortcut.
    u=Corridor(np.array([[0,0],[0,8],[8,8],[8,0]],float))
    assert u.point_distance(np.array([0,0]))==u.point_distance(np.array([8,0]))==0
    assert not u.segment_inside(np.array([0,0]),np.array([8,0]))
    write_json(ROOT/'audit/v3_gate_regression.json',dict(created_utc=utc(),reference_sha256=sha(file),
        source_sha256=sha(__file__),original_intersections=env.corridor.initialization_gate_intersections,
        reference_witness=env.corridor.reference_witness,actual_tracker_initial_frontier=env.tracker.frontier,
        checks=checks,endpoint_inside_interval_outside_rejected=True))
    print('V3 geometry regressions passed')


if __name__=='__main__':main()
