"""Spatial diagnostics at a declared, physically resolvable XY sampling scale.

Energy is a finite-resolution approximation, not exact continuous curvature.
At 100 mm/s and 1 ms, raw segments reach 0.1 mm; finer interpolation can
alias their polygon vertices. Keep 0.2 mm sensitivity checks for sharp turns.
"""
import numpy as np


def spatial_turn_profile(xy, spacing=.1):
    xy=np.asarray(xy)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-10]
    arc,xy=arc[keep],xy[keep]
    if spacing<=0 or len(xy)<4 or arc[-1]<4*spacing:
        raise ValueError('Insufficient spatial support for turn metrics')
    grid=np.arange(0.,arc[-1],spacing)
    points=np.column_stack([np.interp(grid,arc,xy[:,axis]) for axis in (0,1)])
    segments=np.diff(points,axis=0)
    turn=np.arctan2(segments[:-1,0]*segments[1:,1]-segments[:-1,1]*segments[1:,0],
                   np.sum(segments[:-1]*segments[1:],axis=1))
    return dict(xy=xy,arc=arc,ticks=np.flatnonzero(keep),grid=grid,turn=turn)


def geometric_turn_metrics(reference, rows, spacing=.1):
    ref=np.asarray(reference);rows=np.asarray(rows)
    if len(ref)!=3:raise ValueError('Isolated turn required')
    profile=spatial_turn_profile(np.vstack([ref[0],rows[:,:2]]),spacing)
    xy,arc,grid,turn=(profile[k] for k in ('xy','arc','grid','turn'))
    apex_s=arc[np.argmin(np.linalg.norm(xy-ref[1],axis=1))]
    u,v=np.diff(ref,axis=0)
    angle=float(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v));sign=np.sign(angle)
    window=(grid[1:-1]>=apex_s-12.)&(grid[1:-1]<=apex_s+12.)
    selected=turn[window];locations=grid[1:-1][window]
    positive=np.maximum(sign*selected,0.)
    cumulative=np.cumsum(positive)
    width=None
    if len(cumulative) and cumulative[-1]>1e-6:
        low=locations[np.searchsorted(cumulative,.1*cumulative[-1])]
        high=locations[np.searchsorted(cumulative,.9*cumulative[-1])]
        width=float(high-low)
    energy=float(np.sum(selected**2)/spacing)
    return dict(turn_deg=float(np.rad2deg(angle)),spacing_mm=spacing,
        maximum_raw_segment_mm=float(np.diff(arc).max()),
        raw_resolution_supported=bool(np.diff(arc).max()<=spacing+1e-8),
        bending_energy_mm_inv=energy,normalized_bending_mm_inv=energy/(angle**2),
        spatial_peak_curvature_mm_inv=float(np.max(np.abs(selected))/spacing),
        spatial_reverse_turn_deg=float(np.rad2deg(np.maximum(-sign*selected,0.).sum())),
        middle80_turn_arc_length_mm=width,
        window_definition='Recorded XY, linear interpolation at uniform arc spacing; +/-12 mm arc window about closest vertex approach. No position smoothing, no velocity threshold.')
