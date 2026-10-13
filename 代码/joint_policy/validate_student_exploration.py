"""Bounded density, covariance and bias-gradient checks before physical exploration."""
from pathlib import Path
import numpy as np
import torch
from scipy.stats import multivariate_t
from conditional_exploration import RHO
from student_exploration import StudentExplorationPolicy,DF
from study_io import BASE,register,receipt,write_json


def main():
    stage=register('student_exploration_validation_20260916',[Path(__file__),BASE/'student_exploration.py',
        BASE/'conditional_exploration.py',BASE/'joint_policy.py',
        BASE/'coordinated_exploration_coverage_20260916/receipt.json'],
        dict(synthetic_models=2,synthetic_rows=32,autograd_calls=2,physics_ticks=0,training_updates=0),
        'Verify exact multivariate Student conditional density against SciPy, shared mean-shift gradient and covariance matching before any physical probe; no training.')
    torch.set_num_threads(1);torch.manual_seed(20260916);results=[]
    for mode in ('joint','fixed'):
        policy=StudentExplorationPolicy(72,mode=mode);k=3 if mode=='joint' else 2
        obs=torch.randn(16,72);previous=torch.randn(16,72);oldz=torch.randn(16,3)
        latent=torch.randn(16,3);live=torch.tensor([False,True]*8)
        center,scale=policy.conditional_parameters(obs,previous,oldz,live)
        logp,entropy=policy.conditional_terms(obs,latent,previous,oldz,live)
        expected=np.array([multivariate_t.logpdf(z[:k],loc=m[:k],shape=np.diag(s[:k]**2),df=DF)
            for z,m,s in zip(latent.numpy(),center.detach().numpy(),scale.detach().numpy())])
        error=float(np.max(np.abs(expected-logp.detach().numpy())));assert error<1e-4
        covariance=scale.square()*DF/(DF-2)
        expected_cov=torch.exp(2*policy.log_std)*(1-live.float()[:,None]*RHO**2)
        assert torch.allclose(covariance,expected_cov,rtol=1e-6,atol=1e-8)
        weights=torch.linspace(-1,1,16)
        gradient=torch.autograd.grad((logp*weights).mean(),policy.actor[-1].bias)[0]
        residual=(latent-center).detach();scale=scale.detach()
        square=((residual/scale).square()*policy.active_mask).sum(-1)
        score=(DF+k)/(DF+square[:,None])*residual/scale.square()*(1-live.float()[:,None]*RHO)*policy.active_mask
        expected_gradient=(score*weights[:,None]).mean(0)
        gradient_error=float((gradient-expected_gradient).abs().max());assert gradient_error<2e-5
        assert torch.isfinite(entropy).all() and (k==3 or gradient[2]==0)
        results.append(dict(mode=mode,logp_max_error=error,bias_gradient_max_error=gradient_error,covariance_matched=True))
    write_json(stage/'results.json',dict(results=results,df=DF,
        boundary='Algebraic and numerical interface checks only; no exploration coverage or task improvement proven.'))
    receipt(stage,dict(status='student_conditional_density_validation_passed',synthetic_models=2,
        synthetic_rows=32,autograd_calls=2,physics_ticks=0,training_updates=0))
    print(results)


if __name__=='__main__':main()
