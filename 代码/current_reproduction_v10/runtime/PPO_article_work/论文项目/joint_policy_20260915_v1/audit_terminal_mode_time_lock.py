"""Check whether the local learning branch restrictions freeze completion time."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def timing(trace):
    with np.load(trace) as z:rows=z['rows'].copy()
    ticks=np.flatnonzero(rows[:,12])
    assert len(ticks)>0
    begin=int(ticks[0]);count=len(rows)-begin-23
    assert begin%20==0 and count>0
    return dict(initiation_tick=begin,braking_steps=count,post_brake_hold_ticks=23,
        total_ticks=len(rows),identity_total_ticks=begin+count+23)


def main():
    old=BASE/'canonical_gate_band_duration_local_shape_learning_20260918'
    current=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    probes=BASE/'canonical_gate_band_segment_cruise_probe_20260918'
    events=BASE/'canonical_gate_band_segment_cruise_small_events_20260918'
    stage=register('canonical_gate_band_terminal_mode_time_lock_20260918',
        [__file__,old/'results.json',old/'receipt.json',current/'results.json',current/'receipt.json',
         probes/'results.json',probes/'receipt.json',events/'results.json',events/'receipt.json',
         BASE/'model_terminal_trigger_boundaries.py',BASE/'terminal_duration_credit.py',
         BASE/'solve_transition_shape_step.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,
             saved_screen_paths=29,saved_counterfactual_paths=4),
        'Audit a possible algorithmic obstruction to full-path efficiency. The '
        'local credit restrictions preserve both the20ms decision at which '
        'terminal braking starts and the integer1ms cubic braking duration. '
        'Verify on saved paths that complete recorded time equals initiation '
        'plus braking duration plus the unchanged23 recorded hold ticks (which '
        'include the required at-least20ms rest). Fixed initiation and duration '
        'therefore imply fixed completion time within this model branch, even '
        'if cruise requests improve. Compare the existing four finite feed '
        'probes to identify whether their real time gains require leaving that '
        'branch. This is not permission to relax braking, shorten rest, discard '
        'XY failures or reuse wrong branch derivatives. It motivates explicit '
        'validated event-changing credit for future efficiency learning; the '
        'first priority remains actual turn shape. Reuse all physical results.')
    start=time.perf_counter();before=read(old/'results.json');after=read(current/'results.json')
    b={r['original_path']:BASE/t for r,t in zip(before['full_candidate']['results'],before['full_candidate']['traces'])}
    a={r['original_path']:BASE/t for r,t in zip(after['full_candidate']['results'],after['full_candidate']['traces'])}
    record=[]
    for r in read(events/'results.json')['records']:
        path=r['path'];bt=timing(b[path]);at=timing(a[path])
        same=bt['initiation_tick']==at['initiation_tick'] and bt['braking_steps']==at['braking_steps']
        record.append(dict(path=path,parent=bt,candidate=at,same_terminal_mode=same,
            same_total_time=bt['total_ticks']==at['total_ticks']))
        assert not same or bt['total_ticks']==at['total_ticks']
    physical=read(probes/'results.json');assert physical['parent_sha256']==sha(old/before['chosen']['model'])
    probe_results=[]
    for r in physical['records']:
        path=r['path'];bt=timing(b[path]);at=timing(probes/r['trace'])
        gain=(bt['total_ticks']-at['total_ticks'])*.001
        assert abs(gain-r['complete_time_gain_s'])<1e-12
        probe_results.append(dict(path=path,parent=bt,probe=at,complete_time_gain_s=gain,
            initiation_change_ticks=at['initiation_tick']-bt['initiation_tick'],
            braking_duration_change_ticks=at['braking_steps']-bt['braking_steps'],
            shape_acceptance_inferred=False,diagnostic_only=True))
    result=dict(screen_records=record,finite_probe_records=probe_results,
        guarded_screen_paths=len(record),same_mode_screen_paths=sum(r['same_terminal_mode'] for r in record),
        mathematical_implication='Fixed brake initiation tick and duration with unchanged hold fixes complete time.',
        all_time_gaining_probes_change_mode=all(r['initiation_change_ticks'] or r['braking_duration_change_ticks']
            for r in probe_results if r['complete_time_gain_s']>0),
        only_applies_to_guarded_local_branch=True,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_terminal_mode_completion_time_audit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('TERMINAL_TIME_LOCK',result['same_mode_screen_paths'],
        [(r['path'],r['complete_time_gain_s'],r['initiation_change_ticks'],r['braking_duration_change_ticks']) for r in probe_results],flush=True)


if __name__=='__main__':main()
