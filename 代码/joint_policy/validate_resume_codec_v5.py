"""Check state equality and serialization cost for the numerical-buffer codec."""
import io
import time
import numpy as np
import torch
from study_io import BASE,register,receipt,write_json
from resume_codec_v5 import pack_state,unpack_state,save_resume,load_resume
from profile_snapshot_codec import equivalent


if __name__=='__main__':
    source=BASE/'v4_fixed3_terminal_capture_v1/state.pt'
    stage=register('compact_resume_validation_v5',[__file__,source,BASE/'resume_codec_v5.py',
        BASE/'profile_snapshot_codec.py',BASE/'snapshot_codec_profile_v1/receipt.json'],
        dict(captured_state_loads=1,memory_serializations=3,memory_roundtrip_loads=1,
            atomic_file_roundtrips=1,empty_buffer_checks=1,models=0,physics_ticks=0),
        'Verify a compact exact-state storage format before any future training uses it; leave all v4 storage and sources unchanged.')
    torch.set_num_threads(1)
    state=torch.load(source,map_location='cpu',weights_only=False)
    measurements=[]
    for iteration in range(3):
        buffer=io.BytesIO()
        start=time.perf_counter()
        packed=pack_state(state)
        torch.save(packed,buffer,pickle_protocol=5)
        measurements.append(time.perf_counter()-start)
    size=buffer.tell()
    buffer.seek(0)
    restored=unpack_state(torch.load(buffer,map_location='cpu',weights_only=False))
    assert equivalent(state,restored)
    start=time.perf_counter()
    save_resume(stage/'roundtrip.pt',state)
    file_seconds=time.perf_counter()-start
    restored_file=load_resume(stage/'roundtrip.pt')
    assert equivalent(state,restored_file)
    empty=dict(state,batch={key:[] for key in state['batch']},episode_rows=[])
    assert equivalent(empty,unpack_state(pack_state(empty)))
    write_json(stage/'results.json',dict(save_pack_memory_seconds=measurements,bytes=size,
        atomic_file_save_seconds=file_seconds,full_state_recursive_equal=True,
        restored_batch_rows=len(restored['batch']['reward']),
        restored_episode_chunks=len(restored['episode_rows']),empty_buffers_equal=True,
        actual_training_speedup_not_yet_measured=True,active_training_unchanged=True))
    receipt(stage,dict(status='compact_macro_recovery_state_exact_roundtrips_passed',
        captured_state_loads=1,memory_serializations=3,memory_roundtrip_loads=1,
        atomic_file_roundtrips=1,empty_buffer_checks=1,models=0,physics_ticks=0))
    print((stage/'results.json').read_text(encoding='utf-8'))
