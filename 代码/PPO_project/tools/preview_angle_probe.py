"""Registered angle/speed/distance capacity probe, not a learned policy result."""
from dataclasses import asdict
from pathlib import Path
import math
import sys

import numpy as np

import preview_macro_study as study
from terminal_braking import BrakingPolicy


def main():
    out=study.OUT.parent/'angle_capacity'
    protocol=dict(angles_deg=[45,90,135],feed_targets_mm_s=[5.,15.],distances_mm=[1.2,3.,4.],
        segment_lengths_mm=[8.,8.],turn_angle_definition='Change of travel direction, not interior angle',
        one_sided_tolerance_mm=1.25,dt_s=.001,brake=asdict(BrakingPolicy()),
        purpose='Synthetic development geometry intervention. No agent training, no per-test optimum or superiority claim.',
        source=study.source_state(),probe_sha256=study.old.sha(__file__),command='python '+' '.join(sys.argv))
    if (out/'protocol.json').exists():raise FileExistsError('Keep prior angle probe')
    study.old.write(out/'protocol.json',protocol)
    rows=[]
    for angle in protocol['angles_deg']:
        theta=math.radians(angle)
        reference=np.array([[0.,0.],[8.,0.],[8+8*math.cos(theta),8*math.sin(theta)]])
        for speed in protocol['feed_targets_mm_s']:
            for distance in protocol['distances_mm']:
                cfg=study.configuration('preview_time','adaptive',101)
                cfg['revision']['initial_preview_mm']=distance
                label=f'angle{angle}_v{speed:g}_D{distance:g}'
                env=study.MacroEnvironment(cfg,label,reference)
                while not env.done:env.step([0.,speed/cfg['kinematic_constraints']['MAX_VEL'],(distance-.8)/3.8])
                study.old.save_trace(out/f'{label}.npz',env.trace)
                row=dict(angle_deg=angle,feed_target_mm_s=speed,distance_mm=distance,**env.result,
                         trace_sha256=study.old.sha(out/f'{label}.npz'))
                rows.append(row)
                study.old.write(out/f'{label}.json',row)
                print(f'{label}: success={row["success"]}, T={row["elapsed_time_s"]:.3f}, error={row["contour_error_max_mm"]:.3f}',flush=True)
    study.old.write(out/'summary.json',dict(rows=rows,role=protocol['purpose']))


if __name__=='__main__':main()
