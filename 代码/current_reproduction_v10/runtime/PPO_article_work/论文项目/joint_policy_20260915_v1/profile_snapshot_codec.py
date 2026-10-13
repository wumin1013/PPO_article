"""Compare in-memory exact snapshot codecs on a stopped, copied state."""
import io
import time
import numpy as np
import torch
from study_io import BASE, register, receipt, write_json


def equivalent(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


if __name__=='__main__':
    source=BASE/'v4_fixed3_terminal_capture_v1/state.pt'
    stage=register('snapshot_codec_profile_v1',[__file__,source,BASE/'train_pilot.py'],
        dict(captured_state_loads=1,codecs=2,serializations=6,roundtrip_loads=2,
             models=0,physics_ticks=0,training_updates=0),
        'Measure pickle protocol overhead on a preserved state; verify exact recursive equality without touching running storage or scientific code.')
    torch.set_num_threads(1)
    state=torch.load(source,map_location='cpu',weights_only=False)
    results=[]
    for protocol in [2,5]:
        times=[]
        for repetition in range(3):
            buffer=io.BytesIO()
            start=time.perf_counter()
            torch.save(state,buffer,pickle_protocol=protocol)
            times.append(time.perf_counter()-start)
        size=buffer.tell()
        buffer.seek(0)
        start=time.perf_counter()
        restored=torch.load(buffer,map_location='cpu',weights_only=False)
        duration=time.perf_counter()-start
        assert equivalent(state,restored)
        results.append(dict(pickle_protocol=protocol,save_seconds=times,load_seconds=duration,
            bytes=size,recursive_roundtrip_equal=True))
    write_json(stage/'results.json',dict(codecs=results,in_memory_only=True,
        training_throughput_claimed=False,active_code_changed=False))
    receipt(stage,dict(status='exact_snapshot_codec_roundtrips_profiled',
        captured_state_loads=1,serializations=6,roundtrip_loads=2,models=0,physics_ticks=0))
    print(results)
