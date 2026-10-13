"""Re-encode actual successful two-action requests; no simulated motion or PPO."""
import json
import math
import shutil
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT, write, sha
from direct_two_action_control import DirectTwoRuntime
from compact_control import CompactRuntime
from reconstruct_farvector_observations import set_physical


def main():
    stage=ROOT/'direct14_actual_two_action_dataset_round44'
    stage.mkdir(exist_ok=False)
    sources=[('dense_1ms',ROOT/'paired_two_action_initialization_seed922302/d_3/eval_0600.npz',
              ROOT/'paired_two_action_initialization_seed922302/d_3/eval_0600.json',None,.001,.4),
             ('dense_half_ms',ROOT/'paired_two_action_fitted_seed922302_half_ms/trace.npz',
              ROOT/'paired_two_action_fitted_seed922302_half_ms/result.json',
              ROOT/'paired_two_action_fitted_seed922302_half_ms/gate_events.json',.0005,.4),
             ('original',ROOT/'direct14_two_action_initialization_round27/selected_original.npz',
              ROOT/'direct14_two_action_initialization_round27/selected_original.json',
              ROOT/'direct14_two_action_initialization_round27/selected_original_gates.json',.001,.2)]
    parts=[];ys=[];weights=[];groups=[];records=[]
    for name,path,result_path,gates,dt,weight in sources:
        result=json.loads(result_path.read_text(encoding='utf-8'))
        assert result['success'] and result['trace_sha256']==sha(path)
        with np.load(path) as data:
            ref=data['reference'];rows=data['rows'];old=data['observations'];actions=data['actions']
            times=data['decision_times_s'] if 'decision_times_s' in data else np.arange(len(old))*.02
        assert len(old)==len(actions) and np.all(actions[:,2]==3.)
        ticks=np.rint(times/dt).astype(int)
        assert np.all(np.diff(ticks)==round(.02/dt))
        env=DirectTwoRuntime(ref,preview_rule_mm=3.)
        if gates is None:
            for row in rows:env.tracker.update(row[:2])
            snapshot=env.tracker.snapshot(1.25)
            assert snapshot['geometry_complete']
            events=env.tracker.events
            gates=stage/f'{name}_reconstructed_gates.json';write(gates,events)
            env=DirectTwoRuntime(ref,preview_rule_mm=3.)
        else:events=json.loads(gates.read_text(encoding='utf-8'))
        crossing=np.array([e['command_segment'] for e in events])
        encoded=[];errors=[]
        ids=np.arange(len(old))
        if name=='original':
            ids=np.unique(np.r_[np.flatnonzero((abs(old[:,6])>.025)|(abs(old[:,13])>.015)|(np.linalg.norm(old[:,1:3],axis=1)<.995)),
                np.linspace(0,len(old)-1,512).astype(int),np.arange(100),np.arange(len(old)-100,len(old))])
        for i in ids:
            tick=ticks[i]
            if tick:set_physical(env.env,rows[tick-1])
            env.env.tracker.frontier=int(np.searchsorted(crossing,tick,side='left'))
            x=env.observation();encoded.append(x)
            if old.shape[1]==14:errors.append(float(abs(x-old[i]).max()))
            else:
                unchanged=CompactRuntime.observation(env.env)[:13]
                errors.append(float(abs(unchanged-old[i,:13]).max()))
        assert max(errors)<2e-6,(name,max(errors))
        x=np.array(encoded,dtype=np.float32)
        y=(actions[ids,:2]/np.array([math.pi/2,50.])-np.array([0.,1.])).astype(np.float32)
        assert np.all(abs(y)<=1.000001)
        parts.append(x);ys.append(y);weights.append(np.full(len(x),weight/len(x)));groups.extend([name]*len(x))
        records.append(dict(case=name,samples=len(x),weight=weight,dt_s=dt,
            trace=str(path),trace_sha256=sha(path),result_sha256=sha(result_path),
            gates=str(gates),gates_sha256=sha(gates),maximum_unchanged_feature_error=max(errors),
            action_source='Actual held steering radians and feed mm/s;not residual outputs or initial-heading conversion'))
    np.savez_compressed(stage/'supervision.npz',observations=np.vstack(parts),targets=np.vstack(ys),
        weights=np.concatenate(weights).astype(np.float32),case=np.array(groups))
    shutil.copy2(__file__,stage/Path(__file__).name)
    write(stage/'audit.json',dict(sources=records,script_sha256=sha(__file__),
        dataset_sha256=sha(stage/'supervision.npz'),policy_inputs=14,outputs=2,policy_memory_scalars=0,
        new_physics_ticks=0,new_ppo_updates=0,
        scope='Offline re-encoding of previously exposed successful motions. No inherited teacher is present in deployed14input policy.'))
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
