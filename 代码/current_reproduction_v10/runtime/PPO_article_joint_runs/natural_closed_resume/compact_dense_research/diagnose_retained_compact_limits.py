"""Forward-only diagnostics on saved trajectories and offline teacher states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from compact_control import CompactPolicy

ROOT=Path(__file__).resolve().parent
HIST=ROOT.parent/'all34_dominance_research/joint_transit_retained_seed922222_full'
R4=ROOT/'direct14_saturation_aware_fit_seed922302_round4'
R5=ROOT/'direct14_speed_deficit_ppo_seed922302_round5'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    paths={'historical':HIST/'full_metrics.json', 'round4_6000':R4/'eval_6000_original_metrics.json',
        'round4_12000':R4/'eval_12000_original_metrics.json','round5_24':R5/'eval_0024_original_metrics.json'}
    metrics={k:read(p) for k,p in paths.items()}
    h=metrics['historical']; decomposition={}
    for name,m in metrics.items():
        total_deficit=100*m['complete_recorded_time_s']-m['travel_distance_mm']
        hdef=100*h['complete_recorded_time_s']-h['travel_distance_mm']
        decomposition[name]=dict(time_s=m['complete_recorded_time_s'],distance_mm=m['travel_distance_mm'],
            total_net_speed_deficit_mm=total_deficit,corner_deficit_mm=m['corner_total_speed_deficit_mm'],
            noncorner_deficit_mm=total_deficit-m['corner_total_speed_deficit_mm'],
            delta_time_vs_historical_s=m['complete_recorded_time_s']-h['complete_recorded_time_s'],
            distance_contribution_s=(m['travel_distance_mm']-h['travel_distance_mm'])/100,
            deficit_contribution_s=(total_deficit-hdef)/100)
    dataset=R4/'supervision.npz'
    with np.load(dataset) as data:
        x=torch.as_tensor(data['observation']);y=data['original_target'];weight=data['weight']
    counts=read(R4/'dataset.json')['sources']
    original_count=counts[0]['samples'];assert counts[0]['case']=='original'
    model_stats={}
    for name,path in {'round4_6000':R4/'fit_6000.pt','round4_12000':R4/'fit_12000.pt',
                      'round5_24':R5/'checkpoint_0024.pt'}.items():
        policy=CompactPolicy(14,3,922302);policy.load_state_dict(torch.load(path,weights_only=False)['policy'])
        with torch.no_grad():pred=policy.actor(x).numpy()
        stats={}
        for case,sl in [('original',slice(0,original_count)),('dense',slice(original_count,None))]:
            yy=y[sl];pp=pred[sl];sat=yy[:,1]>=1-1e-6
            err=(np.clip(pp,-1,1)-np.clip(yy,-1,1))*np.array([90.,50.,5.6])
            stats[case]=dict(samples=len(yy),mean_abs_action_error_deg_mm_s_mm=np.abs(err).mean(0).tolist(),
                max_abs_action_error_deg_mm_s_mm=np.abs(err).max(0).tolist(),saturated_teacher_samples=int(sat.sum()),
                predicted_saturated_count=int((pp[sat,1]>=1-1e-6).sum()),
                predicted_saturated_feed_min=float((50*(1+np.clip(pp[sat,1],-1,1))).min()),
                predicted_saturated_feed_mean=float((50*(1+np.clip(pp[sat,1],-1,1))).mean()),
                undersaturated_count_below99=int((pp[sat,1]<.98).sum()))
        model_stats[name]=stats
    report=dict(kind='Saved-data forward-only diagnosis; no new training/simulation',
        time_decomposition=decomposition,teacher_state_action_fit=model_stats,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [dataset,*paths.values()]})
    target=ROOT/'retained_compact_limits_diagnostic.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
