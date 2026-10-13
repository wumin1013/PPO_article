"""Exact structural removal of causal traces with zero action coefficients.

Inactive entries are not stored or advanced. The original product-array layout
is reconstructed only for reduction, preserving NumPy summation order.
"""
import numpy as np
from observable_memory_joint_policy import ObservableMemoryJointPolicy, actor_features
from joint_teacher_state_policy import OUTPUT_SCALE, motion_geometry_context
from decision_innovation_policy import innovation_features


class SparseObservableMemoryPolicy(ObservableMemoryJointPolicy):
    def __init__(self, original):
        self.metadata = original.metadata
        self.base = original.base
        self.env = None

    def bind(self, env):
        super().bind(env)
        self.original_memory_shape = self.memory.shape
        self.original_previous_shape = self.previous_features.shape
        decay_count, count, _ = self.original_memory_shape
        self.instant_count = count
        blocks = np.stack([self.coefficients[(i+1)*count:(i+2)*count] for i in range(decay_count)])
        self.active_flat = np.flatnonzero(blocks.ravel() != 0.)
        self.active_coefficients = blocks.ravel()[self.active_flat].copy()
        self.active_decays = self.decays[self.active_flat//(count*3)]
        self.active_feature_flat = self.active_flat % (count*3)
        self.nonzero_blocks = [bool(np.any(block)) for block in blocks]
        self.memory = self.memory.ravel()[self.active_flat].copy()
        self.previous_features = np.zeros(len(self.active_flat), dtype=float)
        return env

    def act(self, observation, deterministic=True):
        original = self.base.act(observation, deterministic)[0]
        _, innovation = innovation_features(self.env, original)
        features = actor_features(self.metadata, observation, innovation)
        if self.last_decision != self.env.decisions:
            if self.last_decision is None:
                assert self.env.decisions == 0
            else:
                assert self.env.decisions == self.last_decision+1
            self.memory = self.active_decays*self.memory+(1.-self.active_decays)*self.previous_features
            if self.metadata.get('context_memory'):
                context = np.tanh((motion_geometry_context(observation)-self.metadata['center_context'])/self.metadata['context_scale'])
                previous = np.stack([features,features,features*context[:,0,None],features*context[:,1,None]])
                self.previous_features = previous.ravel()[self.active_flat].copy()
            else:
                self.previous_features = features.ravel()[self.active_feature_flat].copy()
            self.last_decision = self.env.decisions
        correction = (features*self.coefficients[:self.instant_count]).sum(axis=-2)*OUTPUT_SCALE
        products = np.zeros(self.original_memory_shape, dtype=float)
        products.ravel()[self.active_flat] = self.memory*self.active_coefficients
        for use, product in zip(self.nonzero_blocks, products):
            if use:
                correction += product.sum(axis=-2)*OUTPUT_SCALE
        self.last_instant_features = features.copy()
        self.last_memory_features = self.memory.copy()
        if self.metadata.get('residual_output_precision') == 'float64':
            return original.astype(np.float64)+correction,0.,0.
        return (original+correction).astype(original.dtype),0.,0.


def sparsify_memory(policy):
    if isinstance(policy, ObservableMemoryJointPolicy):
        policy = SparseObservableMemoryPolicy(policy)
    if hasattr(policy, 'base'):
        policy.base = sparsify_memory(policy.base)
    return policy
