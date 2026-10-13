"""Preserve and quantify the old physical-update preview-scaling error."""
import time
import numpy as np
from canonical_gate_state import checkpoint_environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    old=BASE/'canonical_gate_band_current_parameter_geometries_20260917'
    new=BASE/'canonical_gate_band_correct_decoder_parameter_geometries_20260917'
    proposals=['canonical_gate_band_physical_cruise_endpoint_restore_20260917',
        'canonical_gate_band_physical_cruise_margin_solve_20260917']
    accepted=BASE/'canonical_gate_band_physical_cruise_small_learning_20260917'
    stage=register('canonical_gate_band_physical_geometry_decoder_fix_audit_20260917',
        [__file__,old/'geometry.npz',old/'results.json',old/'receipt.json',new/'geometry.npz',
         new/'results.json',new/'receipt.json',accepted/'results.json',accepted/'receipt.json',
         BASE/'joint_runtime_v2.py']+[BASE/p/n for p in proposals for n in ('direction.npz','optimizer.json','receipt.json')],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30),
        'The old learning-geometry preparer used base-runtime preview=2.7+1.9*tanh(z), '
        'but these checkpoints execute preview=6.4+5.6*tanh(z). The old algebraic bound '
        'test compared that wrong map with itself. Quantify the nominal mismatch and '
        'every measured physical proposal under the corrected decoder metric/support. '
        'Verify exact identity of the latent control to reuse it. Preserve actual saved '
        'XY/V/A/J/completion observations, but withdraw the intended-unit matched-geometry '
        'attribution. No rewrite of old receipts, physical limits or trajectory outcomes.')
    start=time.perf_counter()
    with np.load(old/'geometry.npz') as z:a={k:z[k].copy() for k in z.files}
    with np.load(new/'geometry.npz') as z:b={k:z[k].copy() for k in z.files}
    identical={k:bool(np.array_equal(a[k],b[k])) for k in ('metric_latent','support','lower_latent','upper_latent','cruise_features','cruise_latent','cruise_weight')}
    assert all(identical.values())
    latent=b['latent'];actual=b['decoded_actions'];units=b['physical_units'];metric=b['metric_physical']
    wrong=np.array([0.,50.,2.7])+np.array([np.pi/2,50.,1.9])*np.tanh(latent)
    nominal_error=np.max(abs(wrong-actual),axis=0)
    factory=checkpoint_environment_factory('gate_band_canonical_kernel')
    decoder=factory(np.array([[0.,0.],[20.,0.]])).decode_action
    records=[]
    for p in proposals:
        with np.load(BASE/p/'direction.npz') as z:d=z['direction'].ravel()
        for scale in (-.125,.125,.5,1.):
            x=scale*d;delta=(b['support']@x).reshape(-1,3)
            mapped=np.array([decoder(z) for z in latent+delta]);change=np.max(abs(mapped-actual),axis=0)
            records.append(dict(proposal=p,scale=scale,correct_metric_norm=float(np.sqrt(x@metric@x)),
                maximum_actual_saved_action_change=change.tolist(),correct_half_unit_bound=(.5*units).tolist(),
                correct_support_passes=bool(np.all(change<=.5*units+1e-10)),
                correct_radius_passes=bool(x@metric@x<=.1**2+1e-12)))
    retained=read(accepted/'results.json')['full_candidate']
    result=dict(parent_sha256=read(new/'results.json')['parent_sha256'],latent_control_bitwise_identity=identical,
        old_nominal_map_max_error=nominal_error.tolist(),records=records,
        actual_small_model_still_complete=retained['summary']['successes']==64,
        actual_small_model_physical_shape_checks_preserved=retained['checks']['eligible'],
        intended_physical_geometry_comparison_valid=False,old_evidence_preserved=True,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_physical_geometry_decoder_error_audit',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('DECODER_ERROR',nominal_error.tolist(),'LATENT_IDENTITY',all(identical.values()),flush=True)
    for r in records:print('CORRECT_SCALE_AUDIT',r,flush=True)


if __name__=='__main__':main()
