"""Float64 local action differences anchored to an unchanged deployed action."""
import copy
import torch


class CenteredSmoothActor:
    """An offline smooth extension for derivatives, never a deployed actor."""
    def __init__(self,policy):
        self.policy=copy.deepcopy(policy).double()

    @torch.no_grad()
    def center(self,observation,actual_latent):
        self.offset=actual_latent-self.policy.distribution(torch.as_tensor(observation,dtype=torch.float64)).mean.numpy()

    @torch.no_grad()
    def act(self,observation,deterministic=True):
        mean=self.policy.distribution(torch.as_tensor(observation,dtype=torch.float64)).mean.numpy()
        return mean+self.offset,0.,0.
