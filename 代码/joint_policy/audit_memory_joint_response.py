"""Read matched single-coefficient complete responses from memory learning."""
import numpy as np
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_early_observable_memory_joint_learning_20260920'
    source=BASE/'canonical_gate_band_early_affine_coupled_joint_learning_20260920/coupled_initial_point.json'
    seed=np.asarray(read(source)['coefficients'])
    data=read(stage/'state_joint/results.json');rows=data['candidates']
    anchor=next(c for c in rows if np.array_equal(c['coefficients'],seed))
    heads=['steering','feed','preview'];basis=['instantaneous','memory_decay_0.5','memory_decay_0.8']
    metrics=['full_physical_reverse_deg','out_edge_outward_mm','full_normalized_energy',
             'full_spatial_peak_curvature_mm_inv','apex_speed_mm_s','reverse_lateral_total_mm',
             'reverse_lateral_significant_duration_s']
    responses=[]
    for row_index,c in enumerate(rows):
        if c['tag'].startswith('reused'):continue
        matches=[]
        for previous in rows[:row_index]:
            delta=np.asarray(c['coefficients'])-previous['coefficients']
            if np.count_nonzero(abs(delta)>1e-10)==1:matches.append(previous)
        if not matches:continue
        anchor=min(matches,key=lambda previous:abs(np.linalg.norm(np.asarray(c['coefficients'])-previous['coefficients'])-.05))
        d=np.asarray(c['coefficients'])-anchor['coefficients'];active=np.flatnonzero(abs(d)>1e-10)
        i=int(active[0]);pathwise=[]
        for r,b in zip(c['records'],anchor['records']):
            if 'shape' not in r or not r['success'] or not b['success']:continue
            pathwise.append(dict(path=r['path'],delta={k:r['shape'][k]-b['shape'][k] for k in metrics},
                full_time_change_s=r['complete_time_s']-b['complete_time_s'],
                cruise99_change_s=r['motion']['cruise_ge99_s']-b['motion']['cruise_ge99_s']))
        row=dict(tag=c['tag'],index=i,basis=basis[i//6],center_path=(206,304)[i//3%2],
                 reference_tag=anchor['tag'],reference_coefficients=anchor['coefficients'],
                 head=heads[i%3],coefficient_change=float(d[i]),score_change=c['score']-anchor['score'],
                 maximum_normalized_constraint_change=float(np.max(abs(np.asarray(c['measured_inequalities'])-anchor['measured_inequalities']))),
                 minimum=float(min(c['measured_inequalities'])),pathwise=pathwise)
        responses.append(row)
    result=dict(snapshot_result_sha256=sha(stage/'state_joint/results.json'),initial_point_sha256=sha(source),
        measured_vectors=len(rows),single_coefficient_responses=responses,
        interpretation='Each pair differs in exactly ONE coefficient. Reference vectors are '
        'recorded separately because COBYLA can move its initial simplex base when a trial '
        'improves loss. Different reference vectors must not be treated as a common-state '
        'factorial or combined into an exact local Jacobian. '
        'Every head still receives subsequent closed-loop feedback. These are not exact '
        'action-Q derivatives or additive effects; any composed update requires an independent '
        'full physical rollout. No PPO estimator claim.',physical_reruns=0)
    write_json(stage/'matched_memory_head_response_audit.json',result)
    for row in responses:
        print('MEMORY_HEAD_RESPONSE',row['index'],row['basis'],row['center_path'],row['head'],
              'reference',row['reference_tag'],'step',row['coefficient_change'],
              row['score_change'],row['minimum'],flush=True)


if __name__=='__main__':main()
