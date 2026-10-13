"""Saved-data coverage and credit audit; no simulation, training or test access."""
import itertools
import time
import numpy as np
from complete_corner_metrics import complete_corner_metrics,coverage_costs
from bounded_shape_return import terminal_shape_cost
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def inspect_trace(file):
    with np.load(file) as data:
        reference,rows,actions=[data[k] for k in ('reference','rows','actions')]
        if len(reference)!=3:return None
        _,old=terminal_shape_cost(reference,rows,actions,discrete_first_label=False)
        if 'normalized_energy' not in old:return None
        full=complete_corner_metrics(reference,rows)
        coarse=complete_corner_metrics(reference,rows,.2)
        return dict(original=old,complete=full,coverage_costs=coverage_costs(old,full),
                    spatial_02=dict(reverse_deg=coarse['full_spatial_reverse_deg'],
                                    normalized_energy=coarse['full_normalized_energy']),
                    physical_ticks_reused=len(rows))


def main():
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    pool=BASE/'local_geometry_full_training_audit_20260916'
    stage=register('complete_two_edge_shape_audit_20260916',
        [__file__,BASE/'complete_corner_metrics.py',BASE/'bounded_shape_return.py',
         BASE/'isolated_corner_shape.py',BASE/'geometric_turn_metrics.py',source/'receipt.json',
         source/'episode_results.json',source/'paired_groups.json',pool/'receipt.json'],
        dict(new_decisions=0,new_physics_ticks=0,max_saved_traces=1000,wall_s=90),
        'Audit current selected policy and completed stochastic training trajectories. Diagnostic2x2 factors: '
        'local versus complete energy/reversal range; original inbound-only side cost versus both full edge '
        'half-planes. Keep the original weights,0.05mm and5deg allowances, bounded cost, time and failure '
        'returns. Quantify old omissions and changed pair rankings without claiming learning causation. '
        'Adopt a mechanism hypothesis only with concrete XY omissions or score conflicts; no training follows '
        'automatically. Spatial0.2mm check distinguishes interpolation sensitivity. Do not overwrite old metrics.')
    started=time.monotonic();records=[]
    for group,folder in [('training',pool/'training_pool'),('development',source/'validation_18645')]:
        for trace in sorted(folder.glob('*.npz'),key=lambda p:int(p.stem)):
            row=inspect_trace(trace)
            if row:
                meta=read(trace.with_suffix('.json'))
                row.update(split=group,path=int(trace.stem),trace=trace.relative_to(BASE).as_posix(),
                           success=meta['success'],full_time_s=meta['completion_time_s'])
                records.append(row)
    episodes=read(source/'episode_results.json')['episodes'];episode_records=[]
    for meta in episodes:
        assert len(records)+len(episode_records)<1000 and time.monotonic()-started<90
        row=inspect_trace(source/'episodes'/meta['trace'])
        if row:
            row.update(episode=meta['episode'],path=meta['path'],success=meta['success'],
                       base_return=meta['base_return'],full_time_s=meta['full_time_s'])
            episode_records.append(row)
    by_episode={r['episode']:r for r in episode_records};pairs=[]
    for group in read(source/'paired_groups.json')['groups']:
        ids=group['episodes']
        if not all(i in by_episode for i in ids):continue
        for ia,ib in itertools.combinations(ids,2):
            a,b=by_episode[ia],by_episode[ib]
            if not (a['success'] and b['success']):continue
            differences={k:(a['base_return']-a['coverage_costs'][k]['cost'])-(b['base_return']-b['coverage_costs'][k]['cost']) for k in a['coverage_costs']}
            crossing_delta=a['complete']['out_edge_outward_mm']-b['complete']['out_edge_outward_mm']
            pairs.append(dict(episodes=[ia,ib],path=a['path'],return_differences=differences,
                 exit_excursion_delta_mm=crossing_delta,
                 original_prefers_more_exit_crossing=bool(differences['window_incoming']*crossing_delta>0 and abs(crossing_delta)>.05)))
    selected=[r for r in records if r['split']=='training'];successful=[r for r in episode_records if r['success']]
    keys=['window_incoming','window_two_edges','full_incoming','full_two_edges']
    summary=dict(saved_traces=len(records)+len(episode_records),
        selected_training_single_corners=len(selected),
        selected_training_exit_excursion_gt05=sum(r['complete']['out_edge_outward_mm']>.05 for r in selected),
        selected_training_in_excursion_gt05=sum(r['complete']['in_edge_outward_mm']>.05 for r in selected),
        selected_training_heading_overshoot_gt2=sum(r['complete']['heading_overshoot_deg']>2 for r in selected),
        original_prefers_more_exit_crossing_pairs=sum(p['original_prefers_more_exit_crossing'] for p in pairs),
        successful_pairs=len(pairs),
        ranking_changes={k:sum(p['return_differences'][k]*p['return_differences']['window_incoming']<0 for p in pairs) for k in keys[1:]},
        saturation_gt99={k:sum(r['coverage_costs'][k]['cost']>.99 for r in successful) for k in keys},
        successful_stochastic_single_corners=len(successful),
        selected_development=[dict(path=r['path'],turn=r['complete']['turn_deg'],
            out_mm=r['complete']['out_edge_outward_mm'],overshoot=r['complete']['heading_overshoot_deg'],
            reverse=r['complete']['full_physical_reverse_deg']) for r in records if r['split']=='development'],
        no_training_or_dynamic_feasibility_claim=True)
    write_json(stage/'results.json',dict(summary=summary,selected=records,episodes=episode_records,pairs=pairs))
    receipt(stage,dict(status='completed_saved_data_coverage_audit',wall_s=time.monotonic()-started,
                       new_decisions=0,new_physics_ticks=0,summary=summary))
    print(summary)


if __name__=='__main__':main()
