"""Bind an already evaluated missing training path; no baseline rerun."""
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    feedback=BASE/'canonical_gate_band_shape_priority_repaired_half_feedback_20260918'
    cause=BASE/'canonical_gate_band_half_shape_full_completion_audit_20260918'
    data=read(parent/'results.json');model=parent/data['chosen']['model'];pool=data['full_candidate']
    audit=read(cause/'results.json');assert audit['summary']['changed_paths']==[19]
    path=19;j=next(i for i,r in enumerate(pool['results']) if r['original_path']==path)
    trace=BASE/pool['traces'][j]
    stage=register('canonical_gate_band_missing_path19_reference_20260918',
        [__file__,model,parent/'results.json',parent/'receipt.json',trace,
         feedback/'results.json',feedback/'receipt.json',cause/'results.json',cause/'receipt.json'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=10,paths=[19]),
        'The repaired half-step passed all29 screen paths and per-corner shape '
        'but the missing training path19 starts braking20ms later. The exact '
        '8-cell P/V/A planner audit isolates a0.0466mm position lag against '
        'only0.01885mm original terminal-admission slack. Bind its already '
        'completed399a baseline for collecting ONLY this missing local model. '
        'No old path model is repeated and no rejected half-step is promoted. '
        'This supplements the training constraint support, not unseen or sealed '
        'testing. Its future constraint must use the unchanged planner and '
        'actual complete feedback, not hard-code a deployed path/tick rule.')
    baseline=dict(results=[pool['results'][j]],traces=[pool['traces'][j]],
                  metrics=[r for r in pool['metrics'] if r['path']==path])
    assert len(baseline['metrics'])==1
    write_json(stage/'results.json',dict(parent_sha256=sha(model),parent_model=str(model.relative_to(BASE)),
        baseline=baseline,metric_names=data['metric_names'],training_path=path,goal_complete=False))
    receipt(stage,dict(status='prepared_missing_training_path_reference',path=path,
        new_physics_ticks=0,new_actor_decisions=0,goal_complete=False))
    print('MISSING_PATH_REFERENCE',path,sha(model),flush=True)


if __name__=='__main__':main()
