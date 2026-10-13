"""Five new full synthetic programs, registered before any executor is run.

Program families refer to the complete ordered construction; arbitrary shifts,
rotations, scaling, prefixes and parameter changes of a registered construction
must stay in its family. No final program or direct variant occurs in training.
"""
import math
import numpy as np
from common import ROOT, sha, utc, write_json


def arc(center, radius, first, last, spacing=.6):
    n=max(2,math.ceil(abs(last-first)*radius/spacing))
    t=np.linspace(first,last,n+1)
    return np.asarray(center)+radius*np.c_[np.cos(t),np.sin(t)], radius*(1-math.cos(abs(last-first)/n/2))


def programs():
    # Dense corners are part of one open folded contour, not independent trials.
    p=[[0.,0.],[12.,0.]]
    for k in range(6):
        x=12+9*k
        p.extend([[x+3,3+(k%2)],[x+6,0],[x+9,0]])
    p.extend([[78,0],[86,8],[86,24],[68,24],[62,18],[44,18],[38,24],[12,24]])
    yield 'final_dense_fold',np.asarray(p),'six_chevron_bank_with_unequal_return','Dense 3--5 mm corner blocks followed by an asymmetric return; interacting adjacent transitions.',0.,None

    p=np.array([[0,0],[10,0],[16,6],[22,6],[28,0],[38,0],[44,6],[50,6],
                [56,0],[70,0],[78,8],[78,30],[62,30],[58,24],[48,24],
                [44,30],[28,30],[24,20],[14,20],[10,30],[0,30],[0,0]],float)
    yield 'final_chamfer_chain',p,'closed_unequal_chamfer_recess_chain','Complete closed contour with distributed short chamfers and unequal concave recesses.',0.,None

    a,e1=arc((24.,8.),8.,-math.pi/2,math.pi/2)
    b,e2=arc((8.,24.),8.,-math.pi/2,-math.pi)
    c,e3=arc((-10.,34.),10.,0.,math.pi/2)
    p=np.vstack(([[0.,0.],[24.,0.]],a[1:],[[8.,16.]],b[1:],[[0.,34.]],c[1:],[[-24.,44.],[-30.,38.],[-30.,20.]]))
    primitives=[dict(center=[24,8],radius=8,sweep_deg=180),dict(center=[8,24],radius=8,sweep_deg=-90),dict(center=[-10,34],radius=10,sweep_deg=90)]
    yield 'final_line_arc_chain',p,'three_arc_hook_with_terminal_offset','Three tangent circular blocks with two radii, separated by straight runs and a terminal offset.',max(e1,e2,e3)+1e-9,primitives

    # Two joined quintic bends with unequal lengths and curvature distribution.
    t=np.linspace(0,1,71);q=10*t**3-15*t**4+6*t**5
    first=np.c_[35*t,18*q]
    second=np.c_[35+45*t,18-29*q]
    p=np.vstack(([[-10.,0.]],first,second[1:],[[94.,-11.],[104.,-1.],[104.,17.]]))
    yield 'final_quintic_double_bend',p,'unequal_quintic_bends_with_terminal_elbow','Noncircular smooth bends with unequal curvature distributions and a final geometric corner.',0.,None

    # A nested open contour has distinct ordered branches, with no reused prefix.
    p=np.array([[0,0],[58,0],[58,42],[6,42],[6,8],[48,8],[48,32],
                [16,32],[16,16],[38,16],[38,24],[24,24]],float)
    yield 'final_inward_channels',p,'three_level_rectilinear_inward_channel','One nested complete open program; long straight runs followed by progressively shorter blocks.',0.,None


def main():
    folder=ROOT/'prospective'/'final_v3'
    folder.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name,p,family,motivation,error,primitives in programs():
        assert np.linalg.norm(np.diff(p,axis=0),axis=1).min()>1e-8
        file=folder/(name+'.csv')
        np.savetxt(file,p,delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12g')
        rows.append(dict(id=name,file=file.name,sha256=sha(file),group=family,origin='synthetic',
            motivation=motivation,vertices=len(p),length_mm=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
            original_tolerance_mm=1.25,arc_chord_error_bound_mm=error,evaluation_tolerance_mm=1.25-error,
            exact_arc_primitives=primitives,closed=bool(np.linalg.norm(p[0]-p[-1])<1e-9)))
    write_json(folder/'registration.json',dict(created_utc=utc(),generator_sha256=sha(__file__),paths=rows,
        status_at_registration='No policy, teacher or baseline evaluations have been run on these programs.',
        inclusion='All five programs; all seeds and every outcome retained. No replacement or filtering by completion/time.',
        representative_figure_paths=['final_dense_fold','final_line_arc_chain'],
        family_separation='Distinct complete ordered constructions; no source program, contiguous crop, rotated/scaled copy or direct parameter variant is shared with the four training and two validation programs or earlier reserved constructors.',
        old_reserved_status='All five earlier reserved programs are now development diagnostics and are not final-test evidence.',
        scope='Polyline command trajectories; only line_arc_chain represents analytic arcs with a reserved sagitta bound. Other sampled programs define their stored polylines.',
        seeds=[101,211,307,401,503],arms=['adaptive','fixed'],limits=[100.,2000.,62500.],
        resolutions_s=[.001,.0005],fine_resolution_seeds=[101],nominal_tolerance_mm=1.25,
        statistics='Report completion count before conditional time mean/SD. Seed is the independent unit for learning comparisons; paired intervals use complete paired seeds.',
        scientific_claims='Command-space simulation only; synthetic programs are not CAM records or machining measurements.'))


if __name__=='__main__':main()
