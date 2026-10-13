"""Raw-XY energy sensitivity at fixed spatial scale and sampling phase."""
import argparse
import numpy as np
from geometric_turn_metrics import spatial_turn_profile
from study_io import BASE,write_json,sha
from search_complete_return import read


def profile_metrics(trace,spacing,phase):
    with np.load(trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
    xy=np.vstack([ref[0],rows[:,:2]]);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-10];arc,xy=arc[keep],xy[keep]
    grid=np.arange(phase,arc[-1],spacing)
    points=np.column_stack([np.interp(grid,arc,xy[:,i]) for i in range(2)]);d=np.diff(points,axis=0)
    turns=np.arctan2(d[:-1,0]*d[1:,1]-d[:-1,1]*d[1:,0],(d[:-1]*d[1:]).sum(1))
    u,v=np.diff(ref,axis=0);angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)
    return dict(energy=float(1.25*np.square(turns).sum()/spacing/angle**2),
                peak_curvature=float(np.max(abs(turns))/spacing))


def energy(trace,spacing,phase):
    return profile_metrics(trace,spacing,phase)['energy']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_main_turn_memory_joint_learning_20260920')
    parser.add_argument('--paths',nargs='+',type=int)
    parser.add_argument('--selected-only',action='store_true')
    args=parser.parse_args();stage=BASE/args.stage
    result=read(stage/'state_joint/results.json');best=next(c for c in result['candidates'] if c['tag']==result['best'])
    lowest=min(result['candidates'],key=lambda c:c['score']);baseline=result['candidates'][0]
    records=[]
    comparisons=(baseline,best) if args.selected_only else (baseline,best,lowest)
    pairs=[(c,r) for c in comparisons for r in c['records']
           if r['path'] in (args.paths or [c['records'][0]['path']])]
    for c,r in pairs:
        trace=r['trace'];profiles=[]
        for spacing in (.1,.2):
            metrics=[profile_metrics(trace,spacing,spacing*f) for f in (0.,.25,.5,.75)]
            values=[m['energy'] for m in metrics]
            profiles.append(dict(spacing_mm=spacing,phase_fractions=[0.,.25,.5,.75],energy=values,
                peak_curvature=[m['peak_curvature'] for m in metrics],
                mean=float(np.mean(values)),range=float(np.ptp(values))))
        assert abs(profiles[0]['energy'][0]-r['shape']['full_normalized_energy'])<1e-10
        with np.load(trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        xy=np.vstack([ref[0],rows[:,:2]]);p=spatial_turn_profile(xy)
        peak_arc=float(p['grid'][1:-1][np.abs(p['turn']).argmax()])
        raw_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        peak_ms=int(np.searchsorted(raw_arc,peak_arc))-1
        records.append(dict(tag=c['tag'],path=r['path'],score=c['score'],shape=r['shape'],trace_sha256=sha(trace),
            profiles=profiles,spatial_peak_arc_mm=peak_arc,spatial_peak_ms=peak_ms))
    references={r['path']:r for r in records if r['tag']==baseline['tag']}
    for r in (r for r in records if r['tag']!=baseline['tag']):
        reference=references[r['path']]
        r['paired_phase_energy_changes']=[[a-b for a,b in zip(p['energy'],q['energy'])]
            for p,q in zip(r['profiles'],reference['profiles'])]
        r['paired_phase_curvature_changes']=[[a-b for a,b in zip(p['peak_curvature'],q['peak_curvature'])]
            for p,q in zip(r['profiles'],reference['profiles'])]
        print('ENERGY_RESOLUTION',r['tag'],r['path'],r['paired_phase_energy_changes'],
              'CURVATURE',r['paired_phase_curvature_changes'],flush=True)
    write_json(stage/'energy_resolution_audit.json',dict(records=records,new_physical_rollouts=0,
        acceptance_changed=False,interpretation='Same raw unsmoothed1ms XY, two declared spatial '
        'scales and four origin phases. This measures interpolation/sampling sensitivity; '
        'it neither proves a continuous-curve energy ordering nor changes any saved guard. '
        'Endpoint omissions from shifted grids lie on the recorded starting/stopping '
        'straight sections. Original phase0 metrics are reproduced exactly.'))


if __name__=='__main__':main()
