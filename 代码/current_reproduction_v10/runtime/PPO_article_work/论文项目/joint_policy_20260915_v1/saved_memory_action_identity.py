"""Prove unchanged closed trajectories using identical decoded actions on saved states."""
import numpy as np
from joint_teacher_state_policy import features
from observable_memory_joint_policy import memory_correction,actor_features


class SavedMemoryActionIdentity:
    def __init__(self,md,replay,decode_action):
        assert not md.get('context_memory')
        self.decode_action=decode_action
        self.output_precision=md.get('residual_output_precision')
        f=(np.asarray([actor_features(md,x,z) for x,z in zip(replay['observations'],replay['innovations'])])
           if md.get('dynamic_context_count',0) else features(md,replay['observations'],replay['innovations']))
        self.paths={}
        alpha=np.asarray(md['memory_decays'])[:,None,None]
        for path in np.unique(replay['path_ids']):
            ids=np.flatnonzero(replay['path_ids']==path)
            assert np.array_equal(replay['decisions'][ids],np.arange(len(ids)))
            memory=np.zeros((len(alpha),f.shape[-2],3));previous=np.zeros_like(f[0]);traces=[]
            for i in ids:
                memory=alpha*memory+(1.-alpha)*previous;previous=f[i].copy();traces.append(memory.copy())
            self.paths[int(path)]=(f[ids],np.asarray(traces),replay['parent_latents'][ids],replay['held_actions'][ids])

    def unchanged(self,path,coefficients):
        f,memory,latents,actions=self.paths[path];coefficients=np.asarray(coefficients).reshape(-1,3)
        for current,trace,original,action in zip(f,memory,latents,actions):
            correction=memory_correction(current,trace,coefficients)
            latent=(original.astype(np.float64)+correction if self.output_precision=='float64'
                    else (original+correction).astype(original.dtype))
            if not np.array_equal(self.decode_action(latent),action):return False
        return True
