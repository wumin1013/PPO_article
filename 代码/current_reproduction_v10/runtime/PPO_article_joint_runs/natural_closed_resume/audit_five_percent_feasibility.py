"""Static travel-time lower bound from required ordered gate crossings.

No controller execution, learning, or change to the task geometry.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT = Path(r'D:\PPO_article_joint_runs\natural_closed_resume')
CASE = BASE/'paper_evidence_review/natural_closed_case'
RUN = OUT/'all34_dominance_research/joint_transit_retained_seed922222_full'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

ref_path = CASE/'reference.csv'
geometry_path = CASE/'geometry.json'
reference = np.loadtxt(ref_path, delimiter=',', skiprows=1)
geometry = read(geometry_path)
vectors = np.diff(reference,axis=0)
tangents = vectors/np.linalg.norm(vectors,axis=1)[:,None]
# Every interior vertex is a required finite bisector gate. Distance to each
# adjacent segment <= epsilon implies distance to its supporting line <= epsilon.
# On the gate plane, distance to each line = r*cos(turn/2).
cos_half = np.sqrt(np.clip((1.+np.sum(tangents[:-1]*tangents[1:],axis=1))/2.,0.,1.))
assert cos_half.min() > 0.1
radii = (1.25+1e-8)/cos_half+1e-6
indices = {0,len(reference)-2}
for primitive in geometry['primitives']:
    for name in ['start_reference_index','end_reference_index']:
        i = int(primitive[name])
        if 0 <= i < len(reference)-1:
            indices.add(i)
indices = sorted(indices)
centers = reference[indices]
exact_enclosing_radii = np.array([0. if i==0 else radii[i-1] for i in indices])
assert exact_enclosing_radii.max() < 10.
center_distances = np.linalg.norm(np.diff(centers,axis=0),axis=1)
segment_lower = np.maximum(0.,center_distances-exact_enclosing_radii[:-1]-exact_enclosing_radii[1:])
# Also report a deliberately relaxed bound using 10 mm balls around every gate.
loose_radii = np.array([0. if i==0 else 10. for i in indices])
loose_distances = np.maximum(0.,center_distances-loose_radii[:-1]-loose_radii[1:])
source_path = BASE/'geometry_spec.py'
manifest = read(RUN/'runtime_provenance/manifest.json')
def find_source(x):
    if isinstance(x,dict):
        if x.get('original') == str(source_path):
            return x
        for v in x.values():
            found = find_source(v)
            if found is not None:
                return found
    elif isinstance(x,list):
        for v in x:
            found = find_source(v)
            if found is not None:
                return found
    return None
bound_source = find_source(manifest)
assert bound_source is not None and sha(source_path)==bound_source['sha256']
assert sha(ref_path) == geometry['reference_sha256']
s2 = read(CASE/'s2_complete/result.json')['complete_time_s']
joint = read(RUN/'full_result.json')['complete_time_s']
target = 0.95*s2
bound = float(segment_lower.sum()/100.)
loose_bound = float(loose_distances.sum()/100.)
report = {
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'kind':'Static lower bound from a subset of mandatory ordered finite gate crossings',
    'physics_ticks':0,'policy_calls':0,'training_updates':0,
    'reference_length_mm':geometry['analytic_length_mm'],
    'maximum_speed_mm_s':100.,'one_sided_tolerance_mm':1.25,
    'full_gate_count':len(reference)-2,'chosen_gate_count':len(indices)-1,
    'start_plus_gate_indices':indices,'maximum_selected_gate_radius_mm':float(exact_enclosing_radii.max()),
    'sum_center_chords_mm':float(center_distances.sum()),
    'gate_radius_distance_lower_bound_mm':float(segment_lower.sum()),
    'complete_time_lower_bound_s':bound,
    'relaxed_10mm_gate_ball_lower_bound_s':loose_bound,
    's2_complete_time_s':s2,'time_for_5percent_reduction_s':target,
    '5percent_target_below_lower_bound':target<loose_bound,
    'optimistic_maximum_possible_reduction_percent':100.*(s2-bound)/s2,
    'selected_joint_current_reduction_percent':100.*(s2-joint)/s2,
    'proof':[
        'Coverage requires each selected gate crossing in the listed order.',
        'Finite-segment distance constraints enclose gate crossing i in a ball centered at reference[i] with radius epsilon/cos(turn_i/2), conservatively expanded for numerical tolerance.',
        'By triangle inequality, travel between crossings is at least max(0, center_distance-r_i-r_j).',
        'Sum those disjoint ordered travel intervals and divide by the unchanged 100 mm/s vector speed cap.',
        'Circular parts are replaced by chords; omitted gates, acceleration, jerk, final travel after the last gate, braking and rest are ignored. These relaxations lower the bound.',
        'The separately reported 10mm-ball bound relaxes every selected gate far beyond its actual radius and still excludes the 5percent target.'
    ],
    'interpretation':'No policy or observation redesign can attain 5percent complete-time reduction over this unchanged S2 on this unchanged task under the existing constraints.',
    'no_claim_of_optimal_policy':True,
    'source_sha256':{str(p):sha(p) for p in [ref_path,geometry_path,source_path,RUN/'runtime_provenance/manifest.json',Path(__file__)]}
}
path = OUT/'all34_dominance_research/five_percent_complete_time_feasibility.json'
assert not path.exists()
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','start_plus_gate_indices','proof']},ensure_ascii=False,indent=2))
