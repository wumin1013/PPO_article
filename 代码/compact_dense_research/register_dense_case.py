"""Register geometry and acceptance before any performance comparison."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from geometry_spec import Corridor

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def polygon():
    side=[]
    motifs=[[(0.,0.),(8.,0.),(8.,12.),(32.,12.),(32.,0.),(40.,0.)],
            [(0.,0.),(8.,0.),(20.,12.),(32.,0.),(40.,0.)],
            [(0.,0.),(8.,0.),(16.,8./math.tan(math.pi/8)),(24.,0.),(40.,0.)]]
    for j in range(6):
        points=np.asarray(motifs[j%3])+[40*j,0.]
        side.extend(points[1:] if side else points)
    points=[]
    for r in range(4):
        angle=r*math.pi/2
        rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        origin=[[0.,0.],[240.,0.],[240.,240.],[0.,240.]][r]
        part=np.asarray(side)@rot.T+origin
        points.extend(part[1:] if points else part)
    p=np.asarray(points[:-1])
    # Remove only exactly codirectional collinear construction vertices.
    keep=[]
    for i in range(len(p)):
        a,b=p[i]-p[i-1],p[(i+1)%len(p)]-p[i]
        if abs(a[0]*b[1]-a[1]*b[0])>1e-8 or a@b<=0:keep.append(i)
    p=p[keep]
    lengths=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1)
    i=int(np.argmax(lengths))
    start=.5*(p[i]+p[(i+1)%len(p)])
    return np.vstack([start,p[np.r_[np.arange(i+1,len(p)),np.arange(i+1)]],start])

def main():
    stage=ROOT/'dense_closed_case';stage.mkdir(exist_ok=False)
    points=polygon();reference=[points[0]];primitives=[];s=0.
    for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
        length=float(np.linalg.norm(b-a));start_idx=len(reference)-1
        samples=max(1,int(math.ceil(length/25.)))
        reference.extend(a+(b-a)*j/samples for j in range(1,samples+1))
        angle=math.atan2(*(b-a)[::-1])
        primitives.append(dict(kind='line',label=f'block_{i:03}',start_mm=a.tolist(),end_mm=b.tolist(),
            length_mm=length,start_s_mm=s,end_s_mm=s+length,start_heading_rad=angle,end_heading_rad=angle,
            start_reference_index=start_idx,end_reference_index=len(reference)-1))
        s+=length
    reference=np.asarray(reference);reference[-1]=reference[0]
    path=stage/'reference.csv';np.savetxt(path,reference,delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12f')
    # Re-read exactly the exported task, before creating gate and corner indices.
    reference=np.loadtxt(path,delimiter=',',skiprows=1)
    corridor=Corridor(reference,closed=True)
    assert not corridor.ambiguities and not corridor.adjacent_gate_conflicts()
    corners=[]
    for i in range(1,len(reference)-1):
        a,b=corridor.tangents[i-1:i+1]
        change=math.atan2(a[0]*b[1]-a[1]*b[0],a@b)
        if abs(change)<1e-8:continue
        included=180.-abs(math.degrees(change))
        corners.append(dict(id=f'corner_{len(corners):02}',point_mm=reference[i].tolist(),
            reference_vertex_index=i,analytic_s_mm=float(corridor.arcs[i]),reference_s_mm=float(corridor.arcs[i]),
            interior_angle_deg=included,signed_heading_change_deg=math.degrees(change),
            category='acute' if included<89.999 else 'right' if included<90.001 else 'obtuse'))
    # Dense-corner windows are disjoint and identical for all methods. Original
    # case keeps its previously registered +/-10 mm windows without modification.
    for i,c in enumerate(corners):
        left=c['reference_s_mm']-(corners[i-1]['reference_s_mm'] if i else 0.)
        right=(corners[i+1]['reference_s_mm'] if i+1<len(corners) else s)-c['reference_s_mm']
        c.update(window_before_mm=min(10.,.49*left),window_after_mm=min(10.,.49*right))
    columns=[min(corners,key=lambda c:abs(c['interior_angle_deg']-angle))['id'] for angle in [135.,90.,45.]]
    geometry=dict(case_id='dense_closed_case',origin='Constructed dense-turn development benchmark; geometry registered before controller comparison',
        closed=True,lap_count=1,start_end_mm=reference[0].tolist(),primitives=primitives,sharp_corners=corners,
        mechanism_columns=columns,analytic_length_mm=s,reference_file='reference.csv',reference_sha256=sha(path),line_sampling_max_mm=25.)
    write(stage/'geometry.json',geometry)
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),case_id='dense_closed_case',
        geometry_frozen_for_comparison=True,method_frozen=False,goal_complete=False,source_sha256=sha(Path(__file__)),
        reference_sha256=sha(path),geometry_sha256=sha(stage/'geometry.json'),performance_runs_before_registration=0,
        construction='Four rotated sides; each side repeats rectangular, right-apex and45-degree-apex indentations twice. Side extent240mm; dimensions8/12/24mm from specified motifs.',
        full_length_mm=s,corner_count=len(corners),ordered_gate_count=len(corridor.gates),
        acceptance=dict(one_sided_euclidean_tolerance_mm=1.25,vector_v_a_j=[100.,2000.,62500.],policy_period_s=.020,
            physical_periods_s=[.001,.0005],minimum_physical_rest_s=.020,complete_ordered_coverage=True,
            main_cruise_threshold_mm_s=99.,dense_full_time_reduction_fraction=.05,
            primary_comparators=['S2','independently trained two-output actor'],independent_seeds=[922301,922302,922303],
            primary_aggregation='Arithmetic mean of all three selected independently trained models; all must complete. S2 deterministic complete time. No failed prefix receives a complete time.',
            selection='Shared scheduled checkpoints, success first then complete development time; record every seed and checkpoint.',
            allowed_vs_no_swing='Separate paired training comparison; additional no-swing geometric prior explicitly recorded.'),
        original_case_unchanged=str(BASE/'paper_evidence_review/natural_closed_case/reference.csv'),
        new_case_is_unseen_test=False,prohibited='No geometry changes based on comparative outcomes; no lower baseline speed or missing final segment.',
        physics_ticks=0,training_updates=0)
    write(stage/'registration.json',registration)
    print(json.dumps({k:registration[k] for k in ['full_length_mm','corner_count','ordered_gate_count','reference_sha256','performance_runs_before_registration']},indent=2))

if __name__=='__main__':main()
