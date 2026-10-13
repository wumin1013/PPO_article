"""Create or verify sealed synthetic candidate geometry; never execute a planner."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REG = ROOT / 'evidence/final_holdout_v6_registration.json'
V2 = Path(r'D:\PPO_article_work\JMP_revision_20261010_v2')
TOL = 1.25
ROUNDING = math.sqrt(2.) * .5e-12
SAGITTA_CAP = .005
IDS = ['hilbert_cell_sweep', 'harmonic_radial_contour', 'piecewise_curvature_coupon',
       'asymmetric_neck_contour', 'annular_sector_raster']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def observed_inventory():
    manifest = read(ROOT / 'data/manifest.json')
    replay = read(ROOT / 'results/historical_strong_v5/registration.json')
    rows = []
    for item in manifest['paths']:
        p = ROOT / 'data' / (item['id'] + '.csv')
        g = ROOT / item['primitives_file'] if 'primitives_file' in item else None
        rows.append(dict(id=item['id'], group=item['group'], reference=str(p),
                         geometry=str(g) if g else None,
                         origin=item['origin'], previous_split=item['split']))
    for item in replay['cases']:
        if item['id'] in [r['id'] for r in rows]:
            continue
        rows.append(dict(id=item['id'], group=item.get('group', 'historical_complete_program'),
            reference=str(ROOT / item['reference']),
            geometry=str(ROOT / item['geometry']) if 'geometry' in item else None,
            inline_arc_primitives=item.get('exact_arc_primitives'),
            origin=item.get('origin', 'local_legacy_geometry_CAM_origin_unverified'),
            previous_split=item['historical_split']))
    for row in rows:
        p = Path(row['reference'])
        x = np.loadtxt(p, delimiter=',', skiprows=1)
        row.update(reference_sha256=sha(p), vertices=len(x), length_mm=float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum()),
                   closed=bool(np.linalg.norm(x[0]-x[-1]) < 1e-8),
                   bounding_box_mm=[x.min(0).tolist(), x.max(0).tolist()],
                   exclusion='Observed development family; all complete programs and direct variants excluded from final candidates')
        if row['geometry']:
            row['geometry_sha256'] = sha(row['geometry'])
    assert len(rows) == 20 and len({r['reference_sha256'] for r in rows}) == 20
    return dict(source_manifests=[dict(path=str(p), sha256=sha(p)) for p in
        [ROOT/'data/manifest.json', ROOT/'results/historical_strong_v5/registration.json']],
        complete_programs=rows, all_observed_for_isolation=True)


def external_inventory():
    roots = [r'D:\Nutstore\1\基于强化学习的运动规划方法', r'D:\PPO_article_work']
    files = []
    for root in roots:
        run = subprocess.run(['rg', '--files', '--hidden', root, '-g', '*.nc', '-g', '*.tap',
            '-g', '*.gcode', '-g', '!**/.git/**', '-g', '!**/node_modules/**', '-g', '!**/__pycache__/**'],
            capture_output=True, text=True, encoding='utf-8', check=False)
        assert run.returncode in (0, 1), run.stderr
        files.extend(Path(s) for s in run.stdout.splitlines() if s)
    groups = {}
    for p in files:
        groups.setdefault(sha(p), []).append(str(p))
    catalog = read(V2/'literature/external_data_manifest.json')
    result = []
    for digest, copies in groups.items():
        provenance = next(x for x in catalog if x['sha256'] == digest)
        canonical = V2/'literature'/provenance['relative_path']
        source = canonical.read_text(encoding='utf-8')
        xyz = np.array([[float(re.search(a+r'([-+0-9.]+)', line)[1]) for a in 'XYZ']
                        for line in source.splitlines() if line.startswith('G01 ')])
        result.append(dict(sha256=digest, canonical_file=str(canonical), copies=copies,
            provenance=provenance, G01_points=len(xyz), z_range_mm=float(np.ptp(xyz[:,2])),
            terminates_M30=source.strip().endswith('M30'),
            start_end_separation_mm=float(np.linalg.norm(xyz[-1]-xyz[0])),
            vendor_HSC_directive=source.splitlines()[0],
            classification='Published external G-code; CAM-generation provenance not established',
            use='Retained source inventory only. No projection, cropping, execution or final-test adoption.',
            project_machine_validation=False))
    return dict(search_roots=roots, extensions=['.nc', '.tap', '.gcode'], physical_files=len(files),
        unique_programs=len(result), entries=result, established_CAM_programs=0,
        note='Simulation 2 is a full planar published benchmark and could support separately registered external evaluation. Experiment is three-axis; do not flatten it or call it J-NNC machine evidence.')


class Program:
    def __init__(self, start):
        self.points = [np.asarray(start, dtype=float)]
        self.primitives = []
        self.s = 0.

    def line(self, end, label='line'):
        end = np.asarray(end, dtype=float)
        start = self.points[-1].copy()
        length = float(np.linalg.norm(end-start))
        assert length > 1e-9
        i = len(self.points)-1
        self.points.append(end)
        self.primitives.append(dict(kind='line', label=label, start_mm=start.tolist(), end_mm=end.tolist(),
            length_mm=length, start_reference_index=i, end_reference_index=i+1,
            start_s_mm=self.s, end_s_mm=self.s+length))
        self.s += length

    def arc(self, center, radius, angle, sweep, label):
        center = np.asarray(center, dtype=float)
        start = center+radius*np.array([np.cos(angle), np.sin(angle)])
        assert np.linalg.norm(start-self.points[-1]) < 1e-9
        n = int(math.ceil(abs(sweep)/(2.*math.acos(1.-SAGITTA_CAP/radius))))
        n = max(n, 2)
        a = angle + np.linspace(0., sweep, n+1)
        pts = center + radius*np.c_[np.cos(a), np.sin(a)]
        i = len(self.points)-1
        self.points.extend(pts[1:])
        length = radius*abs(sweep)
        chord = radius*(1.-math.cos(abs(sweep)/(2*n)))
        self.primitives.append(dict(kind='arc', label=label, start_mm=start.tolist(), end_mm=pts[-1].tolist(),
            center_mm=center.tolist(), radius_mm=radius, start_angle_rad=angle, sweep_rad=sweep,
            length_mm=length, start_reference_index=i, end_reference_index=i+n,
            start_s_mm=self.s, end_s_mm=self.s+length, chord_count=n,
            analytic_chord_sagitta_mm=chord, chord_error_bound_mm=chord+ROUNDING+1e-10))
        self.s += length


def hilbert_xy(n, index):
    x = y = 0
    t = index
    s = 1
    while s < n:
        rx, ry = 1 & (t//2), 1 & (t ^ (t//2))
        if ry == 0:
            if rx == 1:
                x, y = s-1-x, s-1-y
            x, y = y, x
        x, y = x+s*rx, y+s*ry
        t //= 4
        s *= 2
    return np.array([x,y], float)


def specifications():
    return [
        dict(id=IDS[0], family='recursive_Hilbert_order3_cell_sweep',
             purpose='Dense short blocks, nonalternating turn order and spatially close but ordered revisits.',
             construction='Complete order-3 Hilbert lattice traversal, pitch 5 mm, tangent entry 13 mm and exit 17 mm.',
             isolation='Recursive space-filling order; not a staircase, chevron bank or inward rectangular channel.'),
        dict(id=IDS[1], family='closed_five_lobed_harmonic_radial_polygon',
             purpose='A closed long block sequence with multiple curvature sign changes and unequal lobe widths.',
             construction='360 G01 facets from r(theta)=36+6*cos(5*theta)+2*sin(3*theta), theta=2*pi*k/360.',
             isolation='Positive radial graph with five unequal lobes; not a sine, spiral, triangular fillet or polygonal recess variant.'),
        dict(id=IDS[2], family='open_G01_piecewise_linear_curvature_coupon',
             purpose='A single 135-degree direction change with increasing, constant and decreasing nominal curvature.',
             construction='15 mm inlet; 20 mm linear-curvature ramp to 1/14 mm^-1; constant-curvature section; 20 mm ramp down; 23 mm outlet. G01 facet step <=0.5 mm.',
             isolation='Piecewise curvature schedule, not control-point changes to the cubic/quintic/sinusoidal development generators.'),
        dict(id=IDS[3], family='closed_unequal_270_degree_lobes_with_concave_neck',
             purpose='Exact long circular arcs joined by concave-neck line transitions in a complete closed contour.',
             construction='Two clockwise 270-degree lobes, R14 and R18, centers (0,0)/(47,0), four neck connectors through (22,+/-6).',
             isolation='Two reentrant necks and long major arcs; not a rounded triangle or the two opposite open minor-arc program.'),
        dict(id=IDS[4], family='open_concentric_annular_sector_raster',
             purpose='Repeated curvature reversals and radial connector turns between neighboring curved tracks.',
             construction='Four concentric 120-degree arcs at R12/19/26/34 mm with alternating direction, radial inlet R5 and outlet R44.',
             isolation='Concentric annular raster topology; not unequal opposite arcs with a terminal polyline or a semicircle.')]


def generate_programs():
    output = {}
    points = np.array([hilbert_xy(8, i)*5. for i in range(64)])
    first, last = (points[1]-points[0])/5., (points[-1]-points[-2])/5.
    p = Program(points[0]-13.*first)
    for x in points:
        p.line(x)
    p.line(points[-1]+17.*last)
    output[IDS[0]] = p

    a = np.linspace(0.,2*np.pi,361)
    radius = 36.+6.*np.cos(5*a)+2.*np.sin(3*a)
    points = radius[:,None]*np.c_[np.cos(a),np.sin(a)]
    points[-1] = points[0]
    p = Program(points[0])
    for x in points[1:]:
        p.line(x, 'radial_profile_G01_facet')
    output[IDS[1]] = p

    p = Program([0.,0.])
    p.line([15.,0.], 'inlet')
    heading = 0.
    pos = np.array([15.,0.])
    total_angle = 3*np.pi/4
    L, k = 20., 1/14.
    middle = (total_angle-L*k)/k
    for phase, length, k0, slope in [('ramp_up',L,0.,k/L),('constant',middle,k,0.),('ramp_down',L,k,-k/L)]:
        initial = pos.copy()
        theta0 = heading
        n = int(np.ceil(length/.5))
        for distance in np.linspace(0.,length,n+1)[1:]:
            theta = lambda u: theta0+k0*u+.5*slope*u*u
            dx, ex = quad(lambda u: np.cos(theta(u)),0.,distance,epsabs=1e-12,epsrel=1e-12)
            dy, ey = quad(lambda u: np.sin(theta(u)),0.,distance,epsabs=1e-12,epsrel=1e-12)
            assert ex+ey < 1e-8
            p.line(initial+[dx,dy], phase+'_G01_facet')
        pos = p.points[-1].copy()
        heading += k0*length+.5*slope*length*length
    assert abs(heading-total_angle) < 1e-12
    p.line(pos+23*np.array([np.cos(heading),np.sin(heading)]), 'outlet')
    output[IDS[2]] = p

    a = -np.pi/4
    start = 14*np.array([np.cos(a),np.sin(a)])
    p = Program(start)
    p.arc([0.,0.],14.,a,-3*np.pi/2,'left_major_arc')
    p.line([22.,6.],'upper_neck_left')
    a2 = 3*np.pi/4
    p.line([47.+18*np.cos(a2),18*np.sin(a2)],'upper_neck_right')
    p.arc([47.,0.],18.,a2,-3*np.pi/2,'right_major_arc')
    p.line([22.,-6.],'lower_neck_right')
    p.line(start,'lower_neck_left')
    output[IDS[3]] = p

    a = -np.pi/3
    p = Program(5*np.array([np.cos(a),np.sin(a)]))
    for i,radius in enumerate([12.,19.,26.,34.]):
        p.line(radius*np.array([np.cos(a),np.sin(a)]),'radial_connector_'+str(i))
        sweep = (1 if i%2==0 else -1)*2*np.pi/3
        p.arc([0.,0.],radius,a,sweep,'annular_pass_'+str(i))
        a += sweep
    p.line(44*np.array([np.cos(a),np.sin(a)]),'radial_outlet')
    output[IDS[4]] = p
    return output


def intersections(x, closed):
    hits=[]
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    for i in range(len(x)-1):
        a,b=x[i],x[i+1]
        for j in range(i+2,len(x)-1):
            if closed and i==0 and j==len(x)-2:
                continue
            c,d=x[j],x[j+1]
            if np.any(np.maximum(np.minimum(a,b),np.minimum(c,d)) > np.minimum(np.maximum(a,b),np.maximum(c,d))+1e-10):
                continue
            u,v=b-a,d-c
            denominator=cross(u,v)
            if abs(denominator)>1e-12:
                t,s=cross(c-a,v)/denominator,cross(c-a,u)/denominator
                if -1e-10<=t<=1+1e-10 and -1e-10<=s<=1+1e-10:
                    hits.append([i,j])
            elif abs(cross(c-a,u))<1e-10:
                hits.append([i,j])
    return hits


def similarity_matches(candidate, observed):
    matches=[]
    z=np.diff(candidate[:,0]+1j*candidate[:,1])
    for row in observed:
        x=np.loadtxt(row['reference'],delimiter=',',skiprows=1)
        for reverse in [False,True]:
            q=x[::-1] if reverse else x
            v=np.diff(q[:,0]+1j*q[:,1])
            if row['closed']:
                v=np.r_[v,v[:-1]]
            for i in range(max(0,len(v)-len(z)+1)):
                w=v[i:i+len(z)]
                for reflect in [False,True]:
                    zz=z.conjugate() if reflect else z
                    factor=w[0]/zz[0]
                    if np.max(abs(w-factor*zz)) <= 1e-7*max(1.,float(np.max(abs(w)))):
                        matches.append(dict(observed=row['id'],first_segment=i,reverse=reverse,reflection=reflect))
    return matches


def validate(path, geom, observed):
    x=np.loadtxt(path,delimiter=',',skiprows=1)
    d=np.diff(x,axis=0);lengths=np.linalg.norm(d,axis=1)
    closed=bool(np.linalg.norm(x[0]-x[-1])<1e-9)
    assert x.shape[1]==2 and np.isfinite(x).all() and lengths.min()>1e-9
    turns=np.arctan2(d[:-1,0]*d[1:,1]-d[:-1,1]*d[1:,0],(d[:-1]*d[1:]).sum(1))
    assert np.max(abs(turns)) < np.pi-1e-8
    hits=intersections(x,closed)
    assert not hits, hits
    index=0;errors=[]
    for p in geom['primitives']:
        assert p['start_reference_index']==index
        end=p['end_reference_index'];segment=x[index:end+1]
        assert np.linalg.norm(segment[0]-p['start_mm'])<1e-9
        assert np.linalg.norm(segment[-1]-p['end_mm'])<1e-9
        if p['kind']=='arc':
            radius=p['radius_mm'];center=np.array(p['center_mm'])
            actual=float(abs(np.linalg.norm(segment-center,axis=1)-radius).max())
            chord=radius*(1-math.cos(abs(p['sweep_rad'])/(2*p['chord_count'])))
            assert chord+ROUNDING <= p['chord_error_bound_mm'] and chord<=SAGITTA_CAP+1e-12
            errors.append(dict(label=p['label'],radius_residual_mm=actual,analytic_sagitta_mm=chord,
                               reserved_error_mm=p['chord_error_bound_mm']))
        index=end
    assert index==len(x)-1
    matches=similarity_matches(x,observed)
    assert not matches,matches
    return dict(vertices=len(x),segments=len(d),length_mm=float(lengths.sum()),closed=closed,
        minimum_segment_mm=float(lengths.min()),maximum_turn_deg=float(np.max(abs(turns))*180/np.pi),
        positive_turns=int((turns>1e-6).sum()),negative_turns=int((turns<-1e-6).sum()),
        nonadjacent_intersections=hits,full_primitive_coverage=True,arc_checks=errors,
        exact_vertex_sequence_similarity_matches=matches,
        similarity_scope='Whole candidate against all contiguous observed vertex sequences, including reverse/reflection and closed cyclic starts; shape-generation family review is additionally required, not a theorem for arbitrary resampling.',
        finite_geometry=True,no_exact_reversals=True,model_or_planner_runs=0)


def register():
    if REG.exists() or (OUT/'design_contract.json').exists():
        raise FileExistsError('Already sealed: do not replace candidates or their definition')
    observed=observed_inventory();external=external_inventory();specs=specifications()
    contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='candidate sealed pending final recipe',
        origin='synthetic',CAM=False,industrial_measured_data=False,case_order=IDS,designs=specs,
        source_code_sha256=sha(__file__),performance_selection=False,
        retention='Retain all five candidates and every eventual outcome; no result-conditioned replacement or deletion.',
        usage='Static geometry only until a final training/execution/selection recipe is frozen; this registration is not a completed final study.',
        primitive_semantics='G01 facets are the exact commanded reference in cases 1-3; the radial and curvature formulas are construction recipes, not unsampled analytic path claims. Cases 4-5 contain exact circular arcs with explicitly reserved chord error.')
    write(OUT/'design_contract.json',contract)
    write(OUT/'observed_program_inventory.json',observed)
    write(OUT/'external_program_inventory.json',external)
    programs=generate_programs();cases=[]
    for spec in specs:
        name=spec['id'];folder=OUT/name;folder.mkdir()
        program=programs[name];reference=folder/'reference.csv'
        np.savetxt(reference,np.array(program.points),delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12f')
        chord=max([p.get('chord_error_bound_mm',0.) for p in program.primitives],default=0.)
        geometry=dict(case_id=name,origin='synthetic',units='mm, rad',formal_reference='ordered complete line/arc program',
            primitives=program.primitives,nominal_tolerance_mm=TOL,arc_chord_error_bound_mm=chord,
            maximum_coordinate_rounding_error_mm=ROUNDING,evaluation_tolerance_mm=TOL-chord,
            original_program_no_cropping=True,closed=bool(np.linalg.norm(program.points[0]-program.points[-1])<1e-9),lap_count=1)
        write(folder/'geometry.json',geometry);write(folder/'design.json',spec)
        audit=validate(reference,geometry,observed['complete_programs']);write(folder/'static_audit.json',audit)
        cases.append(dict(id=name,group=spec['family'],reference=str(reference.relative_to(ROOT)).replace('\\','/'),
            reference_sha256=sha(reference),geometry=str((folder/'geometry.json').relative_to(ROOT)).replace('\\','/'),
            geometry_sha256=sha(folder/'geometry.json'),origin='synthetic',CAM=False,
            nominal_tolerance_mm=TOL,arc_chord_error_bound_mm=chord,evaluation_tolerance_mm=TOL-chord,
            static_audit=audit,status='candidate sealed pending final recipe'))
    # Geometry view only, with no policy or baseline trajectories.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for ax,item in zip(axes.flat,cases):
        x=np.loadtxt(ROOT/item['reference'],delimiter=',',skiprows=1)
        ax.plot(x[:,0],x[:,1],color='#28688a',lw=.8)
        ax.scatter(*x[0],color='#d5783b',s=16,label='Start')
        ax.set_title(item['id'].replace('_',' '),fontsize=9)
        ax.set_aspect('equal');ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    axes.flat[-1].axis('off');fig.tight_layout();fig.savefig(OUT/'geometry_overview.png',dpi=140);plt.close(fig)
    outputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    record=dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),status='candidate sealed pending final recipe',
        dataset_id='synthetic_final_candidates_v6',cases=cases,case_count=5,observed_exclusion_programs=20,
        final_training_recipe_sha256=None,execution_allowed_by_this_registration=False,
        model_runs=0,planner_runs=0,performance_results_available=False,completed_final_tests=0,
        selection='Five construction families specified before static checks; no model/planner performance filtering.',
        retention='All five remain in the prospective study. Any change creates a separately archived protocol amendment before execution.',
        final_acceptance_required='Initial rest, all ordered gates and full command intervals, nominal one-sided tolerance with arc reserve, common planar vector limits, braking and final rest; both 1 ms and 0.5 ms under a frozen final recipe.',
        source_and_output_sha256=outputs)
    write(REG,record)
    return record


def verify():
    reg=read(REG)
    assert reg['status']=='candidate sealed pending final recipe' and reg['case_count']==5
    for name,digest in reg['source_and_output_sha256'].items():
        assert sha(ROOT/name)==digest,name
    observed=read(OUT/'observed_program_inventory.json')
    for item in observed['complete_programs']:
        assert sha(item['reference'])==item['reference_sha256']
        if item['geometry']:
            assert sha(item['geometry'])==item['geometry_sha256']
    for item in reg['cases']:
        result=validate(ROOT/item['reference'],read(ROOT/item['geometry']),observed['complete_programs'])
        assert result==item['static_audit'],item['id']
    return reg


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['register','verify'])
    args=parser.parse_args()
    reg=register() if args.mode=='register' else verify()
    print(json.dumps(dict(status=reg['status'],cases=[dict(id=x['id'],vertices=x['static_audit']['vertices'],
        length_mm=x['static_audit']['length_mm'],tolerance_mm=x['evaluation_tolerance_mm']) for x in reg['cases']],
        model_runs=0,planner_runs=0),indent=2))
