"""Whole-program split; all generated paths explicitly labelled synthetic."""
import numpy as np
from common import ROOT, sha, write_json, utc


def build():
    out = ROOT / 'data'
    if (out / 'manifest.json').exists():
        raise FileExistsError('A registered split must not be overwritten')
    out.mkdir(exist_ok=True)
    paths = {}
    paths['train_box'] = ('train', 'rectangular_open_program', np.array([[0,0],[24,0],[24,20],[48,20],[48,0],[72,0]], float))
    x = np.linspace(0, 90, 46)
    paths['train_sine'] = ('train', 'sinusoidal_open_program', np.c_[x, 8*np.sin(x/14)])
    paths['train_bends'] = ('train', 'irregular_bends_program', np.array([[0,0],[20,0],[33,13],[24,29],[45,42],[66,39],[80,52]], float))
    t = np.linspace(-np.pi/2, np.pi/2, 40)
    paths['train_semicircle'] = ('train', 'semicircular_program', np.vstack([[-25,0],[0,0],np.c_[20*np.cos(t[1:]),20+20*np.sin(t[1:])],[0,60]]))
    paths['validation_trapezoid'] = ('validation', 'trapezoidal_closed_program', np.array([[0,0],[35,0],[45,24],[13,32],[0,0]], float))
    paths['validation_zigzag'] = ('validation', 'moderate_zigzag_program', np.array([[0,0],[20,0],[30,12],[42,-2],[54,9],[70,4],[88,4]], float))
    # Five topology/shape groups held apart from every training/validation path.
    dense = [[0.,0.],[12.,0.]]
    for k in range(10):
        dense.extend([[12+3*k,3.],[15+3*k,3.],[15+3*k,0.]])
    dense.append([54.,0.])
    paths['test_short_segments'] = ('test', 'dense_short_block_staircase', np.array(dense))
    paths['test_acute_turns'] = ('test', 'alternating_acute_turns', np.array([[0,0],[24,0],[9,12],[34,16],[19,31],[47,29],[41,50],[68,42]], float))
    t = np.linspace(0, 2.4*np.pi, 100)
    radius = 7+2.6*t
    spiral = np.c_[radius*np.cos(t),radius*np.sin(t)]
    paths['test_curvature'] = ('test', 'nonuniform_curvature_spiral', spiral-spiral[0])
    # Exact primitive description also retained; executed reference is a polyline.
    theta = np.linspace(-np.pi/2,np.pi/2,41)
    mixed = np.vstack([[0,0],[36,0],np.c_[36+12*np.cos(theta[1:]),12+12*np.sin(theta[1:])],[0,24]])
    paths['test_line_arc'] = ('test', 'straight_semicircle_straight_program', mixed)
    paths['test_irregular_pocket'] = ('test', 'irregular_closed_pocket', np.array([[0,0],[39,0],[46,9],[42,18],[25,22],[34,38],[20,51],[3,43],[-8,27],[-4,11],[0,0]], float))
    records = []
    for name, (split, group, path) in paths.items():
        assert np.all(np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-9)
        file = out/(name+'.csv')
        np.savetxt(file,path,delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12g')
        records.append(dict(id=name,split=split,group=group,origin='synthetic',generator='prepare_paths.py',
                            sha256=sha(file),vertices=len(path),length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                            closed=bool(np.linalg.norm(path[0]-path[-1])<1e-9)))
    write_json(out/'manifest.json', dict(created_utc=utc(),source_sha256=sha(__file__),paths=records,
        test_use='Test programs are not executed for tuning. All methods and evaluation rules must be frozen first.',
        historical_paths='Long and dense published paths remain development only; no new test path is a segment/rigid transform/scale of them.',
        primitive_note='test_line_arc uses radius 12 mm, 40 arc chords; maximum analytic chord sagitta = 12*(1-cos(pi/80)) mm. Polyline feasibility and exact-circle deviation must be distinguished.',
        cam_status='No CAM software/program metadata established for the local legacy reference paths. These new programs are explicitly synthetic.'))
    print('Registered', len(records), 'whole-program groups')


if __name__ == '__main__':
    build()
