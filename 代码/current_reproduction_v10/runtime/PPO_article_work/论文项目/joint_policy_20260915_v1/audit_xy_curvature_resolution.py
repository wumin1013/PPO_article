"""Curvature from actual XY samples, with independent dynamics/phase checks."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE, register, receipt, write_json

CURVATURE_NAMES=('full_xy_normalized_energy','full_xy_peak_curvature_mm_inv',
                 'full_physical_normalized_energy','full_physical_peak_curvature_mm_inv')


def curvature_metrics(reference, rows):
    raw=raw_xy_curvature(reference,rows);physical=physical_curvature(reference,rows)
    return {**raw,'full_physical_normalized_energy':physical['energy'],
            'full_physical_peak_curvature_mm_inv':physical['peak']}


def raw_xy_curvature(reference, rows):
    """Three-point circumcircle curvature at the existing 1 ms command samples.

    Use both adjacent XY displacements > 0.005 mm, matching the established
    speed > 5 mm/s moving-domain diagnostic. No position filtering is applied.
    This is finite-resolution geometry, not curvature of the polygon vertices.
    """
    ref=np.asarray(reference);rows=np.asarray(rows)
    xy=np.vstack([ref[0],rows[:,:2]])
    segment=np.diff(xy,axis=0);length=np.linalg.norm(segment,axis=1)
    a,b=segment[:-1],segment[1:]
    chord=np.linalg.norm(a+b,axis=1)
    mask=(length[:-1]>.005)&(length[1:]>.005)&(chord>1e-12)
    curvature=2*(a[mask,0]*b[mask,1]-a[mask,1]*b[mask,0])/(length[:-1][mask]*length[1:][mask]*chord[mask])
    measure=.5*(length[:-1][mask]+length[1:][mask])
    u,v=np.diff(ref,axis=0)
    angle=float(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v))
    assert angle!=0 and len(curvature)>0
    peak=int(abs(curvature).argmax())
    return dict(full_xy_normalized_energy=float(1.25*np.sum(curvature**2*measure)/angle**2),
        full_xy_peak_curvature_mm_inv=float(abs(curvature[peak])),
        peak_tick=int(np.flatnonzero(mask)[peak]+1),moving_vertices=int(mask.sum()))


def physical_curvature(reference, rows):
    speed=np.linalg.norm(rows[:,2:4],axis=1);mask=speed>5
    cross=rows[mask,2]*rows[mask,5]-rows[mask,3]*rows[mask,4]
    curvature=cross/speed[mask]**3
    u,v=np.diff(reference,axis=0);angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)
    return dict(peak=float(abs(curvature).max()),
        energy=float(1.25*np.sum(curvature**2*speed[mask])*.001/angle**2))


def phase_peaks(reference, rows, spacing):
    xy=np.vstack([reference[0],rows[:,:2]])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-10];arc,xy=arc[keep],xy[keep]
    values=[]
    for phase in (0.,.25,.5,.75):
        grid=np.arange(phase*spacing,arc[-1],spacing)
        p=np.column_stack([np.interp(grid,arc,xy[:,axis]) for axis in (0,1)])
        e=np.diff(p,axis=0)
        turn=np.arctan2(e[:-1,0]*e[1:,1]-e[:-1,1]*e[1:,0],np.sum(e[:-1]*e[1:],axis=1))
        values.append(float(abs(turn).max()/spacing))
    return values


def main():
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    probe=BASE/'amplitude_secant_composition_probe_20260917'
    stage=register('actual_xy_curvature_resolution_audit_20260917',
        [__file__,parent/'training_selection.json',probe/'results.json',probe/'receipt.json',
         BASE/'geometric_turn_metrics.py',BASE/'complete_corner_metrics.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60),
        'Audit why fixed0.1mm resampling peak constraints disagree with actual dynamics '
        'under small learned-function updates. Compute unfiltered three-point curvature '
        'from all original1ms XY, curvature energy, independent v/a physical curvature, '
        'and0.1/0.2mm resampling at four grid phases. Apply the same speed>5mm/s '
        'moving-domain diagnostic; no raw path smoothing, geometry/corridor change, '
        'training or acceptance from this audit alone. Retain the old metrics. Check '
        'known straight/circular samples analytically before interpreting real traces.')
    started=time.perf_counter()
    checks=[]
    for radius in (.5,2.,10.):
        a=np.linspace(0,np.pi/2,101);xy=radius*np.column_stack([np.cos(a),np.sin(a)])
        ref=np.array([xy[0],[radius,radius],xy[-1]])
        rows=np.zeros((100,6));rows[:,:2]=xy[1:]
        value=raw_xy_curvature(ref,rows)['full_xy_peak_curvature_mm_inv']
        assert abs(value-1/radius)<1e-9
        checks.append(dict(radius_mm=radius,peak=value,error=abs(value-1/radius)))
    selected=read(parent/'training_selection.json');data=read(probe/'results.json')
    records=[]
    for path in range(16):
        traces=[('parent',BASE/selected['traces'][path])]
        for r in data['candidates']:
            j=next(i for i,s in enumerate(r['results']) if s['original_path']==path)
            traces.append((str(r['coefficient']),BASE/r['traces'][j]))
        for label,trace in traces:
            with np.load(trace) as z:ref,rows=z['reference'],z['rows']
            records.append(dict(path=path,label=label,raw_xy=raw_xy_curvature(ref,rows),
                physical=physical_curvature(ref,rows),phase01=phase_peaks(ref,rows,.1),
                phase02=phase_peaks(ref,rows,.2)))
    write_json(stage/'results.json',dict(analytic_checks=checks,records=records,goal_complete=False,
        definition='Raw XY circumcircle estimates at1ms, no position filter; both adjacent displacements >0.005mm.'))
    for path in (3,5,7,9):
        b,lo,hi=[r for r in records if r['path']==path]
        print('XY_CURVATURE',path,'raw_delta',hi['raw_xy']['full_xy_peak_curvature_mm_inv']-b['raw_xy']['full_xy_peak_curvature_mm_inv'],
            'physical_delta',hi['physical']['peak']-b['physical']['peak'],
            'base_phase01',b['phase01'],'positive_phase01',hi['phase01'],flush=True)
    receipt(stage,dict(status='completed_actual_xy_curvature_resolution_audit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
