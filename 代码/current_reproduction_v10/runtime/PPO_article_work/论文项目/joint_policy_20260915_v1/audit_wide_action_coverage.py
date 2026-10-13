"""Count actual joint-action coverage near acute training corners; no rollouts."""
import numpy as np
from pathlib import Path
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    folders=['spatial_bending_learning_20260916','dynamic_state_residual_learning_20260916','preview_state_residual_learning_20260916']
    stage=register('wide_joint_action_coverage_audit_20260916',[__file__,BASE/'wide_turn_reachability_probe_20260916/results.json']+
        [BASE/f/n for f in folders for n in ('receipt.json','training_results.json')],
        dict(physics_ticks=0,model_calls=0,training_updates=0,wall_s=120),
        'Count actually executed joint actions 4-12mm before 140/155deg training turns. Probe-inspired envelope is diagnostic only, not necessary/sufficient acceptance or an online control rule. Include unsuccessful candidates and missing-region coverage.')
    catalogs={};records=[];summaries=[]
    for folder_name in folders:
        folder=BASE/folder_name;rows=read(folder/'training_results.json')['candidates']
        catalogs[str(folder/'training_results.json')]={r['tag']:r for r in rows}
        local=[];all_actions=[]
        for candidate in rows:
            for record in candidate['results']:
                if record['path'] not in (12,13,14,15):continue
                origin=folder;row=candidate
                while not (origin/'training_traces'/record['trace']).exists():
                    source=Path(row['reused_from'])
                    if str(source) not in catalogs:catalogs[str(source)]={r['tag']:r for r in read(source)['candidates']}
                    row=catalogs[str(source)][row.get('physical_source_tag',row['tag'])];origin=source.parent
                trace=origin/'training_traces'/record['trace']
                with np.load(trace) as data:ref,physical,actions=data['reference'],data['rows'],data['actions']
                decision_ticks=np.arange(len(actions))*20
                xy=np.vstack([ref[0],physical[:,:2]])[decision_ticks]
                incoming=(ref[1]-ref[0])/np.linalg.norm(ref[1]-ref[0])
                remaining=-(xy-ref[1])@incoming
                apex=int(np.argmin(np.linalg.norm(physical[:,:2]-ref[1],axis=1)))
                region=(remaining>=4)&(remaining<=12)&(decision_ticks<=apex)
                broad=(actions[:,2]>=6)&(actions[:,1]>=20)&(actions[:,1]<=55)&(np.abs(actions[:,0])<=np.deg2rad(10))&region
                longest=0;run=0
                for flag in broad:
                    run=run+1 if flag else 0;longest=max(longest,run)
                picked=actions[region]
                if len(picked):all_actions.append(picked)
                local.append(dict(candidate=candidate['tag'],path=record['path'],success=record['success'],
                    region_decisions=int(region.sum()),long_preview_decisions=int(((actions[:,2]>=6)&region).sum()),
                    coordinated_region_decisions=int(broad.sum()),maximum_consecutive_coordinated_decisions=longest,
                    trace_sha256=sha(trace),trace=str(trace),
                    actual_shape=record['shape'],actual_spatial=record['spatial']))
        values=np.vstack(all_actions)
        summary=dict(stage=folder_name,training_corner_records=len(local),records_reaching_region=sum(r['region_decisions']>0 for r in local),
            region_decisions=sum(r['region_decisions'] for r in local),
            records_any_long_preview=sum(r['long_preview_decisions']>0 for r in local),
            records_any_coordinated=sum(r['coordinated_region_decisions']>0 for r in local),
            records_coordinated_ge60ms=sum(r['maximum_consecutive_coordinated_decisions']>=3 for r in local),
            successful_records_coordinated_ge60ms=sum(r['maximum_consecutive_coordinated_decisions']>=3 and r['success'] for r in local),
            preview_quantiles_mm=np.quantile(values[:,2],[0,.5,.9,.99,1]).tolist(),
            feed_quantiles_mm_s=np.quantile(values[:,1],[0,.5,.9,.99,1]).tolist())
        summaries.append(summary);records.append(dict(stage=folder_name,records=local));print(summary,flush=True)
    write_json(stage/'results.json',dict(summary=summaries,records=records,physics_ticks=0,model_calls=0,goal_complete=False,
        boundary='Coverage in a finite probe-inspired action envelope is not proof of a necessary control law or policy capability. No new physics or weight updates.'))
    receipt(stage,dict(status='completed',physics_ticks=0,model_calls=0,training_updates=0,results_sha256=sha(stage/'results.json')))


if __name__=='__main__':main()
