"""Compact numerical buffers while preserving the complete macro state."""
import numpy as np
import torch
from study_io import atomic_replace


def pack_state(state):
    packed=dict(state)
    packed['_resume_format']='joint_compact_v5'
    packed['batch']={key:np.asarray(values) for key,values in state['batch'].items()}
    chunks=state['episode_rows']
    packed['_episode_chunk_lengths']=[len(chunk) for chunk in chunks]
    packed['episode_rows']=np.vstack(chunks) if chunks else np.empty((0,13))
    return packed


def unpack_state(packed):
    state=dict(packed)
    if state.pop('_resume_format',None)!='joint_compact_v5':
        raise ValueError('Unrecognized recovery format')
    state['batch']={key:(list(values) if key in ['observation','latent'] else values.tolist())
        for key,values in state['batch'].items()}
    lengths=state.pop('_episode_chunk_lengths')
    rows=state['episode_rows']
    boundaries=np.r_[0,np.cumsum(lengths)]
    state['episode_rows']=[rows[left:right] for left,right in zip(boundaries[:-1],boundaries[1:])]
    return state


def save_resume(path,state):
    temporary=path.with_suffix('.tmp')
    torch.save(pack_state(state),temporary,pickle_protocol=5)
    atomic_replace(temporary,path)


def load_resume(path):
    return unpack_state(torch.load(path,map_location='cpu',weights_only=False))
