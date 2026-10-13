"""Explicit S2 geometry compatibility for the registered clockwise-arc case.

The original planner is immutable. Reflect the entire geometry to reuse its CCW
arc implementation, and reflect the output back. Extend the empirical corner
ratio below its old 30-degree domain, checking resulting geometry and peaks.
"""
import math
import numpy as np
from study_io import ROOT
from joint_runtime import load_source


def build(reference, geometry, reference_sha):
    source = ROOT/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1'
    g = load_source('natural_s2_geometry', source/'geometry.py')
    s = load_source('natural_s2_scheduler', source/'scheduler.py')
    c = load_source('natural_s2_command', source/'command.py')
    g.CORNER_ERROR_MM = 1.2447
    # This changes enclosure tightness, not the upper bound used for speed caps.
    # The original 2e-5 target exhausts 8192 leaves on a shallow new corner.
    g.PEAK_RELATIVE_TOLERANCE = 1e-3
    g.SOURCE_REGISTRATION_SHA = reference_sha
    reflection = np.array([1., -1.])
    raw=[]
    for p in geometry['primitives']:
        lo, hi = p['start_s_mm'], p['end_s_mm']
        if p['kind']=='line':
            raw.append(g.Line(np.asarray(p['start_mm'])*reflection, np.asarray(p['end_mm'])*reflection, lo, hi))
        else:
            assert p['sweep_rad']<0
            a0=-p['start_angle_rad']; a1=a0-p['sweep_rad']
            raw.append(g.Arc(np.asarray(p['center_mm'])*reflection,p['radius_mm'],a0,a1,lo,hi))
    extended=[]
    original_corner=g._corner
    def corner(a,b):
        phi=math.atan2(abs(float(g.cross(a.tangent,b.tangent))),float(a.tangent@b.tangent))
        if phi<=1e-12 or math.degrees(phi)>=30.-1e-7:
            return original_corner(a,b)
        theta=math.pi-phi; ratio=theta**.9927/2.0769
        d=16*g.CORNER_ERROR_MM/((7*ratio+16)*math.sin(phi/2))
        extended.append(math.degrees(phi))
        return dict(phi_rad=phi,theta_rad=theta,n_c_over_d=ratio,nominal_d_mm=d,nominal_LT_mm=(2*ratio+1)*d,
                    extrapolated_empirical_ratio=True,independent_peak_check_required=True)
    g._corner=corner
    item=dict(id='natural_closed_case',closed=True,reference_file='natural_closed_case/reference.csv',reference_sha256=reference_sha)
    chord=max(p.get('chord_error_bound_mm',0.) for p in geometry['primitives'])
    g._source_geometry=lambda _: (item,reference*reflection,raw,chord)
    curve=g.build_geometry(item['id'])
    schedule=s.schedule(curve.curve_lengths_mm,curve.curve_speed_caps_mm_s,curve.straight_lengths_mm,F=100.,A=1998.,J=62437.5)
    assembled=c.assemble(schedule,curve)
    assembled['pos']*=reflection
    return assembled, dict(adapter='global reflection, analytic original arcs, small-turn ratio extension',
                           peak_enclosure_relative_tolerance=g.PEAK_RELATIVE_TOLERANCE,
                           extrapolated_turns_deg=extended,geometry=curve.metadata,
                           execution=assembled['metadata'])
