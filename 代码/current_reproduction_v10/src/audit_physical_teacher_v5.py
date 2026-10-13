"""Independent persisted-data audit for a completed physical teacher collection."""
import argparse
import json
from collections import Counter
import numpy as np
from common import ROOT,sha,utc,write_json


def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi


def load(path):
    with np.load(path) as data:return {key:data[key].copy() for key in data.files}


def main(run):
    out=ROOT/'results/physical_teacher_v5'/run
    reg=json.loads((out/'registration.json').read_text('utf-8'))
    receipt=json.loads((out/'receipt.json').read_text('utf-8'))
    assert receipt['status']=='completed', 'Completion audit requires all registered branches'
    assert all(sha(ROOT/'src'/name)==digest for name,digest in reg['source_sha256'].items())
    assert all(sha(out/name)==digest for name,digest in receipt['output_sha256'].items())
    base=load(out/'base_teacher_records.npz');branch=load(out/'branch_teacher_records.npz')
    records=[json.loads(line) for line in (out/'branch_records.jsonl').read_text('utf-8').splitlines()]
    assert len(base['observations'])==4833 and len(records)==9666
    assert np.array_equal(base['source_decision'][:4096],np.rint(np.linspace(0,22203,4096)).astype(int))
    assert np.array_equal(base['source_decision'][4096:],np.arange(737))
    assert np.all(base['case'][:4096]=='original') and np.all(base['case'][4096:]=='dense')
    ids=[r['branch_id'] for r in records]
    assert sorted(ids)==list(range(9666)) and len(set(ids))==9666
    expected_pilot=[2*i+j for i in reg['pilot_global_base_indices'] for j in (0,1)]
    assert ids[:12]==expected_pilot
    assert [r['execution_index'] for r in records]==list(range(9666))
    noise=np.load(out/'registered_positive_noise.npy')
    expected_noise=np.clip(np.random.default_rng(reg['noise_seed']).standard_normal((4833,3)),-4,4)*reg['noise_std']
    assert np.array_equal(noise,expected_noise)
    counts=Counter();ticks=0;maximum_position_column_error=0.;max_vector_peaks=np.zeros(3)
    from runtime_bootstrap import setup
    setup()
    from geometry_spec import Corridor,GateTracker
    corridors={}
    for row in reg['cases']:
        with np.load(ROOT/row['trace']) as source:
            reference=source['reference'].copy()
        corridors[row['case']]=Corridor(reference,epsilon=row['evaluation_tolerance_mm'],
            closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
    by_batch={}
    for record in records:by_batch.setdefault(record['trace_file'],[]).append(record)
    for name,batch_records in by_batch.items():
        data=load(out/name)
        assert sha(out/name)==batch_records[0]['trace_sha256']
        for record in batch_records:
            i=record['base_id'];sign=record['sign'];branch_id=record['branch_id']
            assert branch_id==2*i+(0 if sign==1 else 1)
            assert record['case']==base['case'][i] and record['source_decision']==int(base['source_decision'][i])
            assert np.array_equal(record['perturbation'],sign*noise[i])
            assert np.array_equal(record['action_raw'],base['targets_three_raw'][i]+sign*noise[i])
            r0=record['row_offset'];n=record['row_count'];p0=record['position_offset'];pn=record['position_count']
            rows=data['rows'][r0:r0+n];positions=data['positions'][p0:p0+pn]
            assert n+1==pn and len(rows)==n and len(positions)==pn
            assert np.array_equal(positions[0],base['position'][i])
            assert np.array_equal(positions[1:],rows[:,:2])
            tracker=GateTracker(corridors[record['case']],positions[0])
            for key in ('frontier','pending_negative','segment_count','start_error','inside'):
                setattr(tracker,key,base[key][i].item())
            for point in positions[1:]:tracker.update(point)
            assert tracker.frontier==record['frontier_after']
            assert tracker.inside==record['corridor_inside']
            if n:
                velocity=np.diff(positions,axis=0)/.001
                acceleration=np.diff(np.vstack([base['velocity'][i],velocity]),axis=0)/.001
                jerk=np.diff(np.vstack([base['acceleration'][i],acceleration]),axis=0)/.001
                peaks=np.array([np.linalg.norm(x,axis=1).max() for x in (velocity,acceleration,jerk)])
                assert np.array_equal(peaks,record['finite_command_vector_peaks'])
                assert bool(np.all(peaks/np.array([100.,2000.,62500.])<=1.00002))==record['kinematic_pass']
                max_vector_peaks=np.maximum(max_vector_peaks,peaks)
                label=record['label_record_index']
                assert label>=0 and branch['branch_id'][label]==branch_id
                assert np.array_equal(branch['position'][label],positions[-1])
                assert branch['ticks'][label]==base['ticks'][i]+n
                assert bool(branch['eligible_nonterminal_state'][label])==record['eligible_nonterminal_state']
            else:assert record['label_record_index']==-1
            ticks+=n
            status='exception' if record['error'] else ('terminal_success' if record['success'] else ('terminal_failure' if record['done'] else 'active'))
            counts[status]+=1
    errors=[];clipped_counts=[]
    for name,data in [('base',base),('branch',branch)]:
        theta=np.clip(data['targets_three_raw'][:,0],-1+1e-10,1-1e-10)*np.pi/2
        expected=wrap(data['teacher_guide_angle']+theta-data['fixed_guide_angle'])
        assert np.max(abs(expected-data['fixed_theta_unclipped']))<1e-14
        assert np.max(abs(expected/(np.pi/2)-data['targets_two_raw'][:,0]))<1e-14
        assert np.array_equal(data['targets_three_raw'][:,1],data['targets_two_raw'][:,1])
        assert np.array_equal(np.clip(data['targets_three_raw'],-1,1),data['targets_three'])
        assert np.array_equal(np.clip(data['targets_two_raw'],-1,1),data['targets_two'])
        flagged=abs(data['targets_two_raw'][:,0])>1-1e-10
        assert np.array_equal(flagged,data['fixed_steering_clipped'])
        computed=wrap(data['fixed_guide_angle']+np.clip(data['targets_two_raw'][:,0],-1+1e-10,1-1e-10)*np.pi/2-data['teacher_guide_angle']-theta)
        assert np.max(abs(computed-data['fixed_direction_error_rad']))<1e-14
        maximum=float(abs(computed[~flagged]).max(initial=0.))
        assert maximum<1e-10
        errors.append(dict(records=name,maximum_unclipped_direction_error_rad=maximum))
        clipped_counts.append(dict(records=name,clipped=int(flagged.sum()),total=len(flagged)))
    assert ticks==receipt['physics_ticks'] and dict(counts)==receipt['outcome_counts']
    assert int(branch['eligible_nonterminal_state'].sum())==receipt['eligible_branch_states']
    result=dict(created_utc=utc(),status='passed',audit_source_sha256=sha(__file__),
        registration_sha256=sha(out/'registration.json'),receipt_sha256=sha(out/'receipt.json'),
        base_states=4833,branches=9666,branch_labels=len(branch['observations']),source_selection_exact=True,
        every_branch_id_exactly_once=True,antithetic_noise_and_raw_actions_exact=True,
        complete_recorded_command_samples=ticks,all_trace_positions_match_rows=True,
        per_branch_vector_audit_recomputed=True,maximum_command_vector_peaks=max_vector_peaks.tolist(),
        every_command_interval_and_ordered_branch_gate_recomputed=True,
        outcomes=dict(counts),fixed_clipping=clipped_counts,direction_errors=errors,
        all_output_and_registered_source_hashes_verified=True,
        no_training=True,scope='Persisted data and local branch audit; no complete-policy benchmark or new independent seed result')
    write_json(out/'completion_validation.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='collection_001')
    main(parser.parse_args().run)
