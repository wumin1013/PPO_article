"""Reconstruct saved decision states; no new physical integration or training."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import math
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,sha,write
from compact_previewcue_runtime import PreviewCueRuntime as LongViewRuntime
from direct_compact_runtime import DirectCompactRuntime

SOURCES={
    'original_teacher':(ROOT/'historical_fitted_residual_ppo_seed922302_original_full/full_trace.npz',ROOT/'historical_fitted_residual_ppo_seed922302_original_full/gate_events.json'),
    'dense_teacher':(ROOT/'direct14_multicase_balanced_seed922302_round2/eval_0024_dense.npz',ROOT/'direct14_multicase_balanced_seed922302_round2/eval_0024_dense_gates.json'),
    'original_retained':(ROOT/'direct14_feed_response_probe_seed922302_round8/gain_1.25_original.npz',ROOT/'direct14_feed_response_probe_seed922302_round8/gain_1.25_original_gates.json'),
    'dense_retained':(ROOT/'direct14_feed_response_probe_seed922302_round8/gain_1.25_dense.npz',ROOT/'direct14_feed_response_probe_seed922302_round8/gain_1.25_dense_gates.json')}


def main():
    output=ROOT/'previewcue_saved_data';output.mkdir(exist_ok=False)
    reports=[]
    for name,(trace,gatefile) in SOURCES.items():
        with np.load(trace) as data:
            reference=data['reference'];rows=data['rows'];expected=data['observations'][:,:14];actions=data['actions']
        events=json.loads(gatefile.read_text(encoding='utf8'))
        crossed=np.array([e['command_segment'] for e in events])
        runtime=LongViewRuntime(reference);env=runtime.env
        original=[];extended=[];headings=[];frontiers=[]
        for i in range(len(expected)):
            tick=i*20
            if tick:
                row=rows[tick-1]
                env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
                if np.linalg.norm(row[8:10])>1e-12:
                    env.heading=math.atan2(row[9],row[8])
            env.ticks=tick;env.decisions=i;env.tracker.frontier=int(np.searchsorted(crossed,tick,side='left'))
            original.append(DirectCompactRuntime.observation(runtime));extended.append(runtime.observation())
            headings.append(env.heading);frontiers.append(env.tracker.frontier)
        original=np.array(original);extended=np.array(extended)
        error=float(np.max(np.abs(original-expected)))
        # Validate reconstruction against recorded observations before using the new encoder.
        assert error<2e-6,(name,error,np.unravel_index(np.abs(original-expected).argmax(),original.shape))
        unchanged=list(range(13))
        assert np.array_equal(original[:,unchanged],extended[:,unchanged])
        np.savez_compressed(output/(name+'.npz'),observations=extended,original_observations=expected,
            actions=actions,heading=np.array(headings),frontier=np.array(frontiers))
        reports.append(dict(name=name,trace=str(trace),trace_sha256=sha(trace),gates=str(gatefile),gates_sha256=sha(gatefile),
            decisions=len(expected),maximum_reconstruction_error=error,bitwise_fraction=float((original==expected).mean()),
            unchanged_features=unchanged,observation_dimension=14,far_point_mm=30,global_remaining_distance_input=False,local2_6_15mm_points_preserved=True,
            source_kind='Already exposed development trajectory; not unseen test data'))
        print(name,len(expected),'reconstruction_error',error,flush=True)
    write(output/'audit.json',dict(sources=reports,physics_ticks_new=0,ppo_updates_new=0,fit_epochs_new=0,
        output_kind='Offline reconstructed features only; no claim of new trajectory success',
        encoder_source_sha256=sha(ROOT/'compact_previewcue_runtime.py'),script_sha256=sha(Path(__file__))))


if __name__=='__main__':main()
