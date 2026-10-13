"""Scan saved successful paths for the guide's remaining lateral eligibility edge."""
import argparse
import time
import numpy as np
from geometry_spec import Corridor,GateTracker
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-adjacency',action='store_true')
    args=parser.parse_args()
    source=BASE/'canonical_gate_band_matched_probe_20260917'
    data=read(source/'results.json')
    if args.check_adjacency:
        previous=BASE/'canonical_gate_band_remaining_guide_support_audit_20260917'
        potential=read(previous/'results.json')['potential_hits']
        stage=register('canonical_gate_band_guide_adjacency_audit_20260917',
            [__file__,source/'results.json',previous/'results.json',previous/'receipt.json',BASE/'geometry_spec.py'],
            dict(new_actor_decisions=0,new_physics_ticks=0,saved_geometry_ticks=100000,wall_s=60),
            'Resolve the conservative all-gate scan using the exact original '
            'GateTracker on saved command endpoints. Retain only gate channels '
            'adjacent to the actual ordered frontier at each command start. '
            'This distinguishes real guide eligibility events from crossings '
            'of irrelevant infinite planes. Saved geometry only; no physics '
            'or state/guide/coverage changes. Report observed scope explicitly.')
        start=time.perf_counter();actual=[];count=0
        for candidate in data['candidates']:
            for r,trace in zip(candidate['results'],candidate['traces']):
                cases=[p for p in potential if p['arm']==candidate['arm'] and p['path']==r['original_path']]
                if not cases:continue
                lookup={}
                for p in cases:
                    for tick,distance in zip(p['ticks'],p['projection_max_distance_mm']):
                        lookup.setdefault(tick,[]).append((p['vertex'],distance))
                with np.load(BASE/trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
                corridor=Corridor(ref);tracker=GateTracker(corridor,ref[0])
                for tick in range(max(lookup)+1):
                    for vertex,distance in lookup.get(tick,[]):
                        adjacent=[corridor.gates[j].vertex_index for j in (tracker.frontier-1,tracker.frontier)
                                  if 0<=j<len(corridor.gates)]
                        if vertex in adjacent:actual.append(dict(arm=candidate['arm'],path=r['original_path'],
                            tick=tick,vertex=vertex,frontier=tracker.frontier,projection_distance_mm=distance))
                    tracker.update(rows[tick,:2]);count+=1
                    assert count<=100000 and time.perf_counter()-start<60
        write_json(stage/'results.json',dict(actual_guide_lateral_support_events=actual,
            conservative_cases=len(potential),saved_geometry_ticks=count,goal_complete=False))
        receipt(stage,dict(status='completed_exact_guide_adjacency_audit',events=len(actual),
            new_actor_decisions=0,new_physics_ticks=0,saved_geometry_ticks=count,wall_s=time.perf_counter()-start))
        print('ACTUAL_GUIDE_SUPPORT',len(actual),actual[:6],flush=True)
        return
    stage=register('canonical_gate_band_remaining_guide_support_audit_20260917',
        [__file__,source/'results.json',source/'receipt.json',BASE/'geometry_spec.py',
         BASE/'continuous_gate_guidance.py'],
        dict(saved_complete_paths=58,new_actor_decisions=0,new_physics_ticks=0,wall_s=30),
        'The state-side lateral switch was causally confirmed and removed, while '
        'the guide deliberately retains its old0.1mm band. Inspect every saved '
        'nonterminal1ms command-start position for a possible lateral support '
        'cutoff within that narrower guide band. Check every gate conservatively; '
        'multi-gate hits need later adjacency verification. An empty result rules '
        'out this switch on these observed paths only, not all future geometries '
        'or perturbations. No new physics or additional runtime changes.')
    start=time.perf_counter();records=[];ticks=0
    for candidate in data['candidates']:
        for r,trace in zip(candidate['results'],candidate['traces']):
            with np.load(BASE/trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
            corridor=Corridor(ref);positions=np.vstack([ref[0],rows[:-1,:2]])
            active=rows[:,12]==0;ticks+=int(active.sum())
            for gate in corridor.gates:
                h=(positions-gate.point)@gate.normal;indices=np.flatnonzero(active&(abs(h)<.05))
                q=positions[indices]-h[indices,None]*gate.normal
                distance=np.zeros(len(indices))
                for i in (gate.vertex_index-1,gate.vertex_index):
                    along=np.clip((q-corridor.starts[i])@corridor.tangents[i],0.,corridor.lengths[i])
                    distance=np.maximum(distance,np.linalg.norm(q-corridor.starts[i]-along[:,None]*corridor.tangents[i],axis=1))
                bad=np.flatnonzero(distance>1.25+1e-9)
                if len(bad):records.append(dict(arm=candidate['arm'],path=r['original_path'],
                    vertex=gate.vertex_index,ticks=indices[bad].tolist(),
                    projection_max_distance_mm=distance[bad].tolist(),
                    isolated_corner=len(ref)==3))
    result=dict(nonterminal_command_starts=ticks,potential_hits=records,
        runtime_changed=False,new_physics_ticks=0,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_saved_guide_lateral_support_audit',potential_cases=len(records),
        command_starts=ticks,new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('GUIDE_SUPPORT',len(records),records[:6],flush=True)


if __name__=='__main__':main()
