"""Freeze shared paths once, before any mixed-curriculum model is executed."""
import traceback
import numpy as np
from study_io import BASE,ROOT,register,write_json,receipt

if __name__=='__main__':
    refs=sorted((ROOT/'论文项目/revision_20260913_preview_efficiency_v2/geometry_design_v1/references').glob('*.csv'))
    stage=register('mixed_curriculum_preparation_v2',[__file__,BASE/'mixed_curriculum.py',
        BASE/'mixed_curriculum_protocol_v2.md',BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',
        BASE/'geometry_spec.py',BASE/'inspect_geometry_semantics.py',*refs],
        dict(training_paths=48,validation_paths=12,zigzag_generation_attempts=256,
             original_reference_initial_observations=3,physics_ticks=0,models=0),
        'Prepare one shared frozen development curriculum and inspect bounded long-path observations, without execution.')
    try:
        from mixed_curriculum import make_paths,environment_factory
        train,valid=make_paths()
        write_json(stage/'paths.json',dict(training=[r.tolist() for r in train],
            validation=[r.tolist() for r in valid],split='Development curriculum and validation only'))
        observations=[]
        for path in refs:
            reference=np.loadtxt(path,delimiter=',',skiprows=1)
            env=environment_factory(reference)
            obs=env.observation()
            assert obs.shape==(72,) and np.isfinite(obs).all()
            assert np.abs(obs[11:15]).max()<=1.
            observations.append(dict(path=path.stem,observation_dim=len(obs),
                bounded_distant_geometry=obs[11:15].tolist(),max_abs_observation=float(np.abs(obs).max())))
        write_json(stage/'initial_observation_inventory.json',observations)
        receipt(stage,dict(status='shared_curriculum_and_bounded_observations_prepared',
            training_paths=len(train),validation_paths=len(valid),physics_ticks=0,models=0,
            environment_constructions_for_observation=3,performance_certificate=False))
        print({'training_paths':len(train),'validation_paths':len(valid),'observations':observations},flush=True)
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc()))
        raise
