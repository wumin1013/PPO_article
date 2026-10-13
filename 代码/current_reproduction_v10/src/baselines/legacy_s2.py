"""Source-preserving adapter for the existing Sencer-style S2 baseline.

Arbitrary finite polylines are supplied as ordered Line primitives. The original
geometry, curvature enclosures, quintic scheduler and inverse-arclength command
assembler are reused. Small-angle extrapolation is exactly the historical
natural_s2_adapter extension and is recorded, not claimed as a new reproduction.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import numpy as np

SRC=Path(__file__).resolve().parents[1]
ROOT=SRC.parent
sys.path.insert(0,str(SRC))
from runtime_bootstrap import setup, WORK
from metrics import metrics

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

def prepare_geometry(reference, *, dt=.001,tolerance=1.25,limits=(100.,2000.,62500.),corner_scale=1.,source_geometry=None):
    """Shared fixed geometry; scaling is selected on validation before final tests."""
    setup()
    reference=np.asarray(reference,float)
    if reference.ndim!=2 or reference.shape[1]!=2 or len(reference)<2 or not np.isfinite(reference).all():
        raise ValueError('Finite complete planar reference required')
    lengths=np.linalg.norm(np.diff(reference,axis=0),axis=1)
    if np.any(lengths<=1e-10): raise ValueError('Resolve zero-length source blocks explicitly')
    if dt<=0 or tolerance<=0 or not np.isfinite([dt,tolerance,*limits]).all() or min(limits)<=0:
        raise ValueError('Positive finite dt, tolerance and planar limits required')
    source=WORK/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1'
    g=_module('jmp_legacy_s2_geometry',source/'geometry.py')
    s=_module('jmp_legacy_s2_scheduler',source/'scheduler.py')
    c=_module('jmp_legacy_s2_command',source/'command.py')
    if not 0. < corner_scale <= 1.: raise ValueError('corner_scale must be in (0,1]')
    g.CORNER_ERROR_MM=tolerance*(1.2447/1.25)*corner_scale
    g.PEAK_RELATIVE_TOLERANCE=1e-3
    g.SOURCE_REGISTRATION_SHA=hashlib.sha256(reference.tobytes()).hexdigest()
    arcs=np.r_[0.,np.cumsum(lengths)]
    raw=[g.Line(reference[i],reference[i+1],arcs[i],arcs[i+1]) for i in range(len(lengths))]
    reflection=np.ones(2);chord=0.
    if source_geometry is not None:
        # Defining primitives are provided by the original complete program,
        # never inferred by fitting a sampled curve. Validate every source span.
        arc_rows=[p for p in source_geometry['primitives'] if p['kind']=='arc']
        if any(p['sweep_rad']<0 for p in arc_rows):
            if not all(p['sweep_rad']<0 for p in arc_rows):raise ValueError('Mixed arc orientations need a general native adapter')
            reflection=np.array([1.,-1.])
        raw=[]
        for p in source_geometry['primitives']:
            lo,hi=p['start_reference_index'],p['end_reference_index']
            if p['kind']=='line':
                a,b=np.asarray(p['start_mm']),np.asarray(p['end_mm']);d=b-a
                f=np.sum((reference[lo:hi+1]-a)*d,axis=1)/float(d@d)
                if np.max(np.linalg.norm(reference[lo:hi+1]-a-f[:,None]*d,axis=1))>2e-7 or np.any(np.diff(f)<-1e-9):
                    raise ValueError('Defining line differs from complete reference')
                raw.append(g.Line(a*reflection,b*reflection,p['start_s_mm'],p['end_s_mm']))
            else:
                theta=np.linspace(p['start_angle_rad'],p['start_angle_rad']+p['sweep_rad'],hi-lo+1)
                expected=np.asarray(p['center_mm'])+p['radius_mm']*np.column_stack((np.cos(theta),np.sin(theta)))
                if np.max(np.linalg.norm(expected-reference[lo:hi+1],axis=1))>2e-7:raise ValueError('Defining arc differs from complete reference')
                a0=p['start_angle_rad']*reflection[1];a1=a0+p['sweep_rad']*reflection[1]
                raw.append(g.Arc(np.asarray(p['center_mm'])*reflection,p['radius_mm'],a0,a1,p['start_s_mm'],p['end_s_mm']))
                chord=max(chord,p.get('chord_error_bound_mm',0.))
    extrapolated=[];original_corner=g._corner
    def corner(a,b):
        phi=math.atan2(abs(float(g.cross(a.tangent,b.tangent))),float(a.tangent@b.tangent))
        if phi<=1e-12 or math.degrees(phi)>=30.-1e-7: return original_corner(a,b)
        theta=math.pi-phi;ratio=theta**.9927/2.0769
        d=16*g.CORNER_ERROR_MM/((7*ratio+16)*math.sin(phi/2))
        extrapolated.append(math.degrees(phi))
        return dict(phi_rad=phi,theta_rad=theta,n_c_over_d=ratio,nominal_d_mm=d,
                    nominal_LT_mm=(2*ratio+1)*d,extrapolated_empirical_ratio=True)
    g._corner=corner
    item=dict(id='jmp_supplied_polyline',closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9),
        reference_file='caller_supplied_complete_polyline',reference_sha256=g.SOURCE_REGISTRATION_SHA)
    g._source_geometry=lambda path_id:(item,reference*reflection,raw,chord)
    curve=g.build_geometry(item['id'],vmax=limits[0],amax=.999*limits[1],jmax=.999*limits[2])
    if reflection[1]<0:
        native_position=curve.position
        curve.position=lambda sigma:native_position(sigma)*reflection
    curve.metadata['source_representation']='validated original analytic primitives' if source_geometry is not None else 'caller polyline'
    return curve,g,s,c,source,extrapolated

def plan(reference, *, dt=.001,tolerance=1.25,limits=(100.,2000.,62500.),corner_scale=1.,source_geometry=None):
    started=time.perf_counter()
    curve,g,s,c,source,extrapolated=prepare_geometry(reference,dt=dt,tolerance=tolerance,limits=limits,corner_scale=corner_scale,source_geometry=source_geometry)
    schedule=s.schedule(curve.curve_lengths_mm,curve.curve_speed_caps_mm_s,curve.straight_lengths_mm,
        F=limits[0],A=.999*limits[1],J=.999*limits[2])
    c.DT_S=dt;c.STATIONARY_HOLD_SAMPLES=math.ceil(.025/dt)
    assembled=c.assemble(schedule,curve)
    meta=dict(method='Legacy S2: Sencer-style quintic corner smoothing plus curve-constant feed/straight quintic transitions',
        source_sha256={name:sha(source/name) for name in ['geometry.py','scheduler.py','command.py']},
        adapter_sha256=sha(__file__),planning_wall_s=time.perf_counter()-started,
        tolerance_mm=tolerance,corner_error_parameter_mm=g.CORNER_ERROR_MM,limits=list(limits),
        corner_scale=corner_scale,
        internal_limits=[limits[0],.999*limits[1],.999*limits[2]],peak_relative_enclosure_tolerance=1e-3,
        small_angle_extrapolated_degrees=extrapolated,
        adaptation='Ordered source-provider replacement only; all primitives are straight blocks. Analytic arc provenance is not reconstructed from chords.',
        unsupported='Exact reversals, true turns above170degrees, missing entry/exit straight remnants, unresolved numerical peak enclosures.',
        geometry=curve.metadata,execution=assembled['metadata'])
    meta['geometry']['settings']['public_audit_limits']=dict(vmax_mm_s=limits[0],amax_mm_s2=limits[1],jmax_mm_s3=limits[2])
    return assembled['pos'],meta

def evaluate_validation(output):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest=json.loads((ROOT/'data/manifest.json').read_text('utf-8'))
    paths=[row for row in manifest['paths'] if row['split']=='validation']
    write(output/'registration.json',dict(scope='Validation only; no test execution or tuning',
        paths=[row['id'] for row in paths],source_sha256=sha(__file__),
        dt_s=.001,tolerance_mm=1.25,limits=[100.,2000.,62500.],corner_error_mm=1.2447,
        parameter_selection='Historical fixed settings; no per-validation tuning in this invocation'))
    results=[]
    for record in paths:
        reference=np.loadtxt(ROOT/'data'/(record['id']+'.csv'),delimiter=',',skiprows=1)
        started=time.perf_counter();error=None;meta=None
        try:
            positions,meta=plan(reference)
            result=metrics(reference,positions)
        except Exception as exc:
            positions=reference[:1].copy();error=type(exc).__name__+': '+str(exc)
            result=dict(success=False,complete_time_s=None,reason='planning_failure')
        result.update(path_id=record['id'],error=error,wall_s=time.perf_counter()-started)
        np.savez_compressed(output/(record['id']+'.npz'),reference=reference,positions=positions)
        if meta: write(output/(record['id']+'_planner.json'),meta)
        write(output/(record['id']+'.json'),result)
        results.append(result)
        print(json.dumps({k:result.get(k) for k in ['path_id','success','complete_time_s','reason','error','wall_s']}),flush=True)
    write(output/'summary.json',dict(results=results,successes=sum(row['success'] for row in results),
        attempted=len(results),test_paths_executed=0))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-output',type=Path,required=True)
    args=parser.parse_args();evaluate_validation(args.validation_output)

if __name__=='__main__': main()
