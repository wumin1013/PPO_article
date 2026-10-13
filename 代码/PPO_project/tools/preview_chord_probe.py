"""Development-only chord-deviation preview selector; no trained-policy claims."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml
import preview_research_v2 as v2


class ChordPreview:
    """Largest candidate distance whose reference chord deviation fits a budget.

    Precomputation uses only the input polyline, not a path ID or rollout results.
    This geometric guide does not guarantee that a dynamic tracker follows the
    chord. All resulting commands still require independent trajectory checks.
    """
    def __init__(self, reference, deviation_budget_mm, grid_mm=.05):
        self.reference = np.asarray(reference, float)
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]
        self.grid = np.linspace(0., self.arc[-1], int(np.ceil(self.arc[-1]/grid_mm))+1)
        self.distances = np.arange(.8, 4.60001, .2)
        self.profile = []
        began = time.perf_counter()
        for s in self.grid:
            accepted = .8
            a = self.point(s)
            for distance in self.distances:
                end = min(s+distance, self.arc[-1])
                b = self.point(end)
                inside = self.reference[(self.arc > s) & (self.arc < end)]
                vector = b-a
                length2 = float(vector@vector)
                fraction = np.clip((inside-a)@vector/max(length2, 1e-20), 0., 1.)
                error = np.linalg.norm(inside-a-fraction[:, None]*vector, axis=1)
                if not len(error) or error.max() <= deviation_budget_mm:
                    accepted = float(distance)
            self.profile.append(accepted)
        self.preprocessing_s = time.perf_counter()-began

    def point(self, s):
        return np.array([np.interp(s, self.arc, self.reference[:, i]) for i in (0, 1)])

    def at(self, s):
        return float(np.interp(s, self.grid, self.profile))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction', type=float, choices=[.4, .7], required=True)
    args = parser.parse_args()
    destination = v2.ROOT/'论文项目/revision_20260908/v2/teacher_probes/chord_v1'/f'budget_{args.fraction:g}'
    state = v2.source_state()
    paths = [p for p in v2.manifest()['paths'] if p['split'] in ('train', 'validation')]
    for path in paths:
        path_id = path['id']
        run = destination/path_id
        if (run/'result.json').exists():
            print('Preserved', path_id, flush=True)
            continue
        run.mkdir(parents=True, exist_ok=True)
        reference = v2.path_points(path_id)
        cfg = v2.config_for('pretrain_mid', 101)
        cfg['revision']['diagnostic'] = dict(method='reference-chord-deviation geometry-only preview',
            deviation_fraction=args.fraction, candidate_distance_mm=[.8,4.6,.2],
            profile_grid_mm=.05, feed_target_mm_s=5., trained=False)
        (run/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        env, obs = v2.make_env(cfg, path_id, reference=reference)
        selector = ChordPreview(reference, env.half_epsilon*args.fraction)
        np.savez_compressed(run/'reference_profile.npz', arc_mm=selector.grid, distance_mm=selector.profile)
        rows = [v2.backend.sample_row(env, {}, 0., [0., .05, .5])]
        total_return, done, began = 0., False, time.perf_counter()
        while not done:
            s = float(env._project_onto_progress_path(env.current_position)[2])
            selected = selector.at(s)
            env.env.lookahead_dist = selected
            obs, reward, done, info = env.step([0., .05])
            row = v2.backend.sample_row(env, info, reward, [0., .05, .5])
            row.update(geometry_query_arc_mm=s, geometry_selected_mm=selected)
            rows.append(row)
            total_return += reward
        v2.backend.save_trace(run/'trace.npz', rows)
        result = v2.backend.evaluate_trajectory(reference, [[r['x_mm'],r['y_mm']] for r in rows],
            [r['timestamp_s'] for r in rows], tolerance_mm=env.half_epsilon,
            limits=cfg['kinematic_constraints'], termination_reason=info['done_reason'],
            claimed_success=info['done_reason']=='success', initial_velocity=[0.,0.],
            initial_acceleration=[0.,0.], initial_heading=rows[0]['heading_rad'],
            initial_angular_velocity=0., initial_angular_acceleration=0.)
        result.update(path_id=path_id, split=path['split'], deviation_fraction=args.fraction,
            method='geometry-only chord preview', trained=False, episode_return=total_return,
            source_state=state, probe_script_sha256=v2.backend.sha(__file__),
            config_sha256=v2.backend.sha(run/'config.yaml'),
            path_manifest_sha256=v2.backend.sha(v2.DATA/'manifest.json'),
            command='python '+' '.join(sys.argv), preprocessing_s=selector.preprocessing_s,
            rollout_wall_s=time.perf_counter()-began, final_online_coverage=info['ordered_coverage'])
        v2.backend.write_json(run/'result.json', result)
        print(json.dumps({k:result[k] for k in ('path_id','success','elapsed_time_s','failure_reasons')}), flush=True)


if __name__ == '__main__': main()
