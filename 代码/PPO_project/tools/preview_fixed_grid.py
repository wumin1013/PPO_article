"""Additional fixed-distance validation probes; these are not trained policies."""
import json
from pathlib import Path
import sys
import time

import yaml
import preview_research_v2 as v2


def main():
    destination = v2.ROOT/'论文项目/revision_20260908/v2/teacher_probes/fixed_intermediate_grid'
    protocol = dict(distances_mm=[2.6,3.0,3.4], split='validation',
        angular_action=0., feed_mm_s=5., trained=False, source=v2.source_state(),
        script_sha256=v2.backend.sha(__file__), manifest_sha256=v2.backend.sha(v2.DATA/'manifest.json'))
    if (destination/'protocol.json').exists():
        if json.loads((destination/'protocol.json').read_text(encoding='utf-8')) != protocol:
            raise ValueError('Preserve earlier grid protocol')
    else: v2.backend.write_json(destination/'protocol.json',protocol)
    for distance in protocol['distances_mm']:
        for path in v2.manifest()['paths']:
            if path['split'] != 'validation': continue
            run=destination/f'D{distance:g}'/path['id']
            if (run/'result.json').exists(): continue
            run.mkdir(parents=True,exist_ok=True)
            cfg=v2.config_for('pretrain_mid',101)
            cfg['reward_weights']['lookahead']['distance']=distance
            cfg['revision']['diagnostic']=dict(method='constant fixed preview', distance_mm=distance, trained=False)
            (run/'config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
            reference=v2.path_points(path['id'])
            env,obs=v2.make_env(cfg,path['id'],reference=reference)
            rows=[v2.backend.sample_row(env,{},0.,[0.,.05,.5])]
            done=False; began=time.perf_counter()
            while not done:
                obs,reward,done,info=env.step([0.,.05])
                rows.append(v2.backend.sample_row(env,info,reward,[0.,.05,.5]))
            v2.backend.save_trace(run/'trace.npz',rows)
            result=v2.backend.evaluate_trajectory(reference,[[r['x_mm'],r['y_mm']] for r in rows],
                [r['timestamp_s'] for r in rows],tolerance_mm=env.half_epsilon,limits=cfg['kinematic_constraints'],
                termination_reason=info['done_reason'],claimed_success=info['done_reason']=='success',
                initial_velocity=[0.,0.],initial_acceleration=[0.,0.],initial_heading=rows[0]['heading_rad'],
                initial_angular_velocity=0.,initial_angular_acceleration=0.)
            result.update(path_id=path['id'],split='validation',distance_mm=distance,trained=False,
                protocol_sha256=v2.backend.sha(destination/'protocol.json'),config_sha256=v2.backend.sha(run/'config.yaml'),
                trace_sha256=v2.backend.sha(run/'trace.npz'),wall_time_s=time.perf_counter()-began,
                command='python '+' '.join(sys.argv))
            v2.backend.write_json(run/'result.json',result)
            print(json.dumps({k:result[k] for k in ('path_id','distance_mm','success','elapsed_time_s','failure_reasons')}),flush=True)


if __name__=='__main__': main()
