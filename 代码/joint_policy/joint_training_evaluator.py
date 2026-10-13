"""Shared measured objectives and unchanged adoption guards for training branches."""
import numpy as np
from head_counterfactual_rollouts import cost
from learn_early_memory_joint_policy import metric_functions
from duration_joint_metrics import guards
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards


def build_evaluator(data,selected,stage,parent,deadline):
    if any('shape' not in r for r in selected):
        # Preserve the existing isolated-corner evaluator, then append the existing
        # ordered-window protections for every affected nonisolated route.
        from multi_corner_raw_metrics import augment,guards as multi_guards,objective
        corners=[r for r in selected if 'shape' in r]
        multi=[r for r in selected if 'shape' not in r]
        for r in multi:augment(r)
        corner_entry=build_evaluator(data,corners,stage,parent,deadline)
        def mixed_entry(records,u,tag):
            by={r['path']:r for r in records}
            result=corner_entry([by[r['path']] for r in corners],u,tag)
            extra=[];extra_score=0.
            for b in multi:
                r=augment(by[b['path']]);extra.extend(multi_guards(r,b))
                extra_score+=objective(r,b)
                if r['success']:
                    extra.extend([(b['complete_time_s']-r['complete_time_s'])/.02,
                        (r['motion']['cruise_ge99_s']-b['motion']['cruise_ge99_s'])/.02,
                        (r['motion']['longest_cruise_ge99_s']-b['motion']['longest_cruise_ge99_s'])/.02])
                    extra_score+=sum(4*(w['shape']['reverse_lateral_total_mm']/.005)**2
                                     for w in r['multi_corner_windows'])
                else:extra.extend([-100.]*3)
            for key in ('measured_inequalities','augmented_measured_inequalities','augmented_learning_inequalities'):
                result[key]+=np.asarray(extra).tolist()
            result['score']+=extra_score;result['training_score']+=extra_score
            result.update(records=records,actual_rerun_paths=[r['path'] for r in selected],
                physical_pass=all(r['success'] and r['audit']['kinematic_pass'] and r['audit']['rest_pass']
                    and r['corridor']['margin_lower_mm']>=0. for r in records))
            return result
        return mixed_entry
    _,old_guards=metric_functions(data,selected,stage,parent,deadline)
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    paths=[r['path'] for r in selected]
    def entry(records,u,tag):
        first=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        safe=all(r['success'] and r['audit']['kinematic_pass'] and r['audit']['rest_pass']
            and r['corridor']['margin_lower_mm']>=0. for r in records)
        score=sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected))
        ep=sum((r['shape']['in_edge_outward_mm']/.002)**2+
            (0. if r['target_motion']['first_turn_inside'] else 1e5) for r in records) if safe else 1e9
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score,training_score=score+ep,
            physical_pass=bool(safe),measured_inequalities=first.tolist(),
            augmented_measured_inequalities=np.r_[first,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,first),extra].tolist(),
            actual_rerun_paths=paths,exact_zero_support_paths=[])
    return entry
