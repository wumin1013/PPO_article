"""Prepare candidate14features on verified round17states without simulation."""
import json
import numpy as np
from pathlib import Path
from evaluate_historical_dense import ROOT,sha,write
from compact_previewcue_runtime import PreviewCueRuntime
from compact_farvector_runtime import FarVectorRuntime
from reconstruct_farvector_observations import set_physical


def main():
    stage=ROOT/'direct14_guided_batchreset_ppo_seed922302_round17'
    output=ROOT/'farvector_round17_data';output.mkdir(exist_ok=False)
    reports=[]
    for case in ['original','dense']:
        trace=stage/f'eval_0024_{case}.npz'
        gates=stage/f'eval_0024_{case}_gates.json'
        with np.load(trace) as data:
            ref=data['reference'];rows=data['rows'];expected=data['observations'];actions=data['actions']
        crossed=np.array([x['command_segment'] for x in json.loads(gates.read_text(encoding='utf-8'))])
        env=FarVectorRuntime(ref);old=[];new=[]
        for i in range(len(expected)):
            tick=20*i
            if tick:
                set_physical(env.env,rows[tick-1])
            env.env.tracker.frontier=int(np.searchsorted(crossed,tick,side='left'))
            old.append(PreviewCueRuntime.observation(env));new.append(env.observation())
        old=np.array(old);new=np.array(new)
        error=float(np.max(abs(old-expected)))
        assert error<2e-6
        keep=[k for k in range(14) if k!=5]
        assert np.array_equal(old[:,keep],new[:,keep])
        np.savez_compressed(output/(case+'.npz'),observations=new,original_observations=expected,actions=actions)
        reports.append(dict(case=case,samples=len(expected),trace=str(trace),trace_sha256=sha(trace),
            gates_sha256=sha(gates),max_original_reconstruction_error=error,
            observed_reference_cos_min=float(old[:,5].min()),observed_negative_cos_count=int((old[:,5]<0).sum())))
        print(case,len(expected),error,flush=True)
    write(output/'audit.json',dict(sources=reports,observation_dimension=14,replaced_feature=5,
        new_physics_ticks=0,new_training_updates=0,source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'compact_farvector_runtime.py']},
        scope='Already exposed complete round17states; candidate encoder data only,not a new model'))


if __name__=='__main__':
    main()
