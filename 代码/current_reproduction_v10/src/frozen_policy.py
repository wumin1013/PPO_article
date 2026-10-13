"""Deterministic expression in Appendix B, using the bound checkpoint tensors."""
from pathlib import Path
import torch
import torch.nn.functional as F

class FrozenPolicy:
    def __init__(self, checkpoint):
        saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        self.weights = saved['policy']
        self.outputs = int(saved['interface'][1])
        self.preview_mm = saved.get('preview_rule_mm')
        if self.outputs == 3:
            assert saved['policy_class'] == 'LocalizedJointPolicy'
            assert saved.get('feed_gain', 1.) == 1.
            self.prefix = 'actor.backbone.backbone.'
        else:
            assert self.outputs == 2 and saved['policy_class'] == 'LocalizedTwoPolicy'
            self.prefix = 'actor.backbone.'

    def _features(self, x, prefix):
        w = self.weights
        distance = torch.sqrt((((x.unsqueeze(-2) - w[prefix+'centers']) /
            w[prefix+'feature_scale']).square()).sum(-1).clamp_min(1e-20)) / w[prefix+'radius']
        return (1.-distance).clamp_min(0.).pow(4)*(4.*distance+1.)

    @torch.no_grad()
    def mean(self, observations):
        x = torch.as_tensor(observations, dtype=torch.float32)
        assert x.shape[-1] == 14
        value = x
        for layer in [0, 2, 4]:
            prefix = self.prefix+str(layer)+'.'
            value = F.linear(value, self.weights[prefix+'weight'], self.weights[prefix+'bias'])
            if layer != 4:
                value = torch.tanh(value)
        if self.outputs == 3:
            delta = .4*torch.tanh(F.linear(self._features(x, 'actor.backbone.'),
                self.weights['actor.backbone.correction.weight']))
            value = torch.cat([value[..., :2], value[..., 2:3]+delta], dim=-1)
        joint = torch.tanh(F.linear(self._features(x, 'actor.'), self.weights['actor.correction.weight']))
        return value+joint*self.weights['actor.output_scale']

    def decoded(self, observations):
        import numpy as np
        bounded = np.clip(self.mean(observations).numpy().astype(float), -1.+1e-10, 1.-1e-10)
        steering = np.pi/2*bounded[..., 0]
        feed = 50.*(1.+bounded[..., 1])
        preview = 6.4+5.6*bounded[..., 2] if self.outputs == 3 else np.full_like(feed, self.preview_mm)
        return np.stack([steering, feed, preview], axis=-1)
