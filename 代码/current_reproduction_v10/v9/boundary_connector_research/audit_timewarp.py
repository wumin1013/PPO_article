"""Independent read-only proof audit of the archived timewarp diagnostic.

All new files belong to boundary_connector_research.  The source and diagnostic
traces are never edited.  Suffix geometry is inherited only after equality of
all acceptance-relevant tracker state and all suffix samples is established.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
sys.path.insert(0,str(BASE/'src'))
from runtime_bootstrap import setup
setup()
from geometry_spec import Corridor,GateTracker


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(tracker):
    return dict(position=tracker.position.tolist(),inside=tracker.inside,
                frontier=tracker.frontier,pending_negative=tracker.pending_negative,
                start_error=tracker.start_error)


def audit():
    summary=ROOT/'boundary_timewarp_diagnostic/summary.json'
    data=json.loads(summary.read_text('utf-8'))
    rows=[]
    for row in data['rows']:
        source=Path(row['source'])
        target=Path(row['selected_trace'])
        record=json.loads(source.with_suffix('.json').read_text('utf-8'))
        assert sha(source)==row['source_sha256']==record['trace_sha256']
        assert sha(target)==row['selected_trace_sha256']
        with np.load(source) as f:old=f['positions'];reference=f['reference']
        with np.load(target) as f:new=f['positions']
        dt=row['dt_s'];join=round(.2/dt);saved=round(row['saved_s']/dt)
        new_join=join-saved
        corridor=Corridor(reference,epsilon=record['tolerance_mm'],
                          closed=np.linalg.norm(reference[0]-reference[-1])<1e-9)
        original=GateTracker(corridor,old[0]);trial=GateTracker(corridor,new[0])
        for p in old[1:join+1]:original.update(p)
        for p in new[1:new_join+1]:trial.update(p)
        equal_states=state(original)==state(trial)
        suffix_equal=np.array_equal(old[join:],new[new_join:])
        predecessor_equal=np.array_equal(old[join-2:join+1],new[new_join-2:new_join+1])
        v=np.diff(new,axis=0)/dt
        a=np.diff(np.vstack((np.zeros(2),v)),axis=0)/dt
        j=np.diff(np.vstack((np.zeros(2),a)),axis=0)/dt
        ov=np.diff(old,axis=0)/dt
        oa=np.diff(np.vstack((np.zeros(2),ov)),axis=0)/dt
        oj=np.diff(np.vstack((np.zeros(2),oa)),axis=0)/dt
        suffix_differences=all(np.array_equal(x[join:],y[new_join:])
                              for x,y in ((ov,v),(oa,a),(oj,j)))
        peaks=[float(np.linalg.norm(x,axis=1).max()) for x in (v,a,j)]
        physical=bool(np.all(np.array(peaks)/record['limits']<=1+2e-5))
        rests=(np.linalg.norm(v,axis=1)<=.001)&(np.linalg.norm(a,axis=1)<=.01)
        nonrest=np.flatnonzero(~rests)
        rest_s=(len(rests)-(int(nonrest[-1])+1 if len(nonrest) else 0))*dt
        terminal_equal=np.array_equal(old[-round(.025/dt):],new[-round(.025/dt):])
        accepted=bool(record['success'] and equal_states and original.inside and
                      suffix_equal and predecessor_equal and suffix_differences and
                      physical and rest_s>=.02-1e-10 and terminal_equal)
        rows.append(dict(case=row['case'],method=row['method'],dt_s=dt,
            source_sha256=sha(source),target_sha256=sha(target),
            original_tracker_state=state(original),new_tracker_state=state(trial),
            tracker_states_equal=equal_states,suffix_positions_exact=suffix_equal,
            last_three_join_samples_exact=predecessor_equal,
            suffix_difference_v_a_j_exact=suffix_differences,
            vector_peaks=peaks,terminal_rest_s=rest_s,terminal_samples_exact=terminal_equal,
            complete_time_s=(len(new)-1)*dt if accepted else None,
            saved_s=record['complete_time_s']-(len(new)-1)*dt,
            complete_acceptance_from_prefix_and_exact_suffix=accepted))
    output=dict(source_summary_sha256=sha(summary),auditor_sha256=sha(__file__),
        semantic_review='For the identical Corridor, update depends on position, inside, frontier and pending_negative. start_error is preserved by the unchanged first position. events and segment_count only label evidence and do not affect acceptance. Identical last three prefix samples ensure the first unchanged suffix sample has identical backward-difference velocity, acceleration and jerk. The whole prefix receives interval and ordered-gate checks; full suffix geometry is inherited from a successful SHA-bound original trace only after these conditions hold.',
        scope='Offline planned-command refinement only. This is not a new closed-loop policy rollout, training seed, or independent test.',
        rows=rows,all_passed=all(row['complete_acceptance_from_prefix_and_exact_suffix'] for row in rows))
    destination=Path(__file__).with_name('timewarp_independent_audit.json')
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(all_passed=output['all_passed'],rows=len(rows),output=str(destination))))


if __name__=='__main__':audit()
