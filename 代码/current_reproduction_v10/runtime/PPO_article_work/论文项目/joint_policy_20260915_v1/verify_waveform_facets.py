"""Analytical and supporting-plane checks for the waveform constraint operator."""
import time
import numpy as np
from waveform_constraint_model import waveform_facets
from study_io import BASE,register,receipt,write_json


def main():
    stage=register('waveform_affine_constraint_contract_20260917',
        [__file__,BASE/'waveform_constraint_model.py'],
        dict(synthetic_cases=102,new_physics_ticks=0,new_actor_decisions=0,wall_s=10),
        'Check the mathematical local-model operator only. Synthetic values are '
        'not experimental trajectories. Verify a moving curvature peak and newly '
        'negative angular-rate sample analytically, then test exact contact and '
        'the supporting-plane lower-bound property at random coefficient points. '
        'These tests establish no physical model accuracy or policy performance.')
    start=time.perf_counter();nom=np.zeros((2,6));g=np.zeros((2,6,1));valid=np.ones((2,6),bool)
    nom[:,3]=[1.,.99];g[:,3,0]=[-1.,1.]
    peak=waveform_facets(nom,g,valid,90.,np.array([.1]))['full_xy_peak_curvature_mm_inv'][0]
    assert abs(peak-1.09)<1e-12
    nom[:,5]=[-1.,.01];g[:,5,0]=[2.,-2.]
    reversal=waveform_facets(nom,g,valid,90.,np.array([.5]))['full_physical_reverse_deg'][0]
    assert abs(reversal-.00099)<1e-12
    rng=np.random.default_rng(917631);max_contact=0.;max_violation=0.
    for _ in range(100):
        nom=rng.normal(size=(12,6));g=rng.normal(size=(12,6,4))
        valid=rng.random((12,6))>.2;valid[0]=True
        center=rng.normal(size=4);other=rng.normal(size=4)
        current=waveform_facets(nom,g,valid,1.,center)
        values=waveform_facets(nom,g,valid,1.,other)
        for name,(value,row,offset) in current.items():
            max_contact=max(max_contact,abs(row@center+offset-value))
            max_violation=max(max_violation,row@other+offset-values[name][0])
    assert max_contact<1e-12 and max_violation<1e-12
    result=dict(moving_peak_expected=1.09,moving_peak_actual=peak,
        new_reverse_expected=.00099,new_reverse_actual=reversal,
        maximum_contact_error=max_contact,maximum_support_violation=max_violation,
        physical_performance_evidence=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='verified_convex_waveform_supporting_facets',
        wall_s=time.perf_counter()-start,new_physics_ticks=0,new_actor_decisions=0))
    print('WAVEFORM_AFFINE_CHECK',result,flush=True)


if __name__=='__main__':main()
