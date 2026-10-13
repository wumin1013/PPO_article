"""Compare complete paired motion with the existing fixed 84-corner windows."""
import argparse
import csv
import json
from pathlib import Path
from analyze_dense_trace import analyze, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--main-trace',type=Path,required=True)
    parser.add_argument('--main-result',type=Path,required=True)
    parser.add_argument('--two-trace',type=Path,required=True)
    parser.add_argument('--two-result',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=False)
    main=analyze(args.main_trace,args.main_result)
    two=analyze(args.two_trace,args.two_result)
    for name,data in [('main',main),('two_action',two)]:
        (args.out/f'{name}_metrics.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    corners=[]
    for a,b in zip(main['corners'],two['corners']):
        assert a['id']==b['id'] and a['angle_deg']==b['angle_deg']
        corners.append(dict(id=a['id'],angle_deg=a['angle_deg'],
            main_time_s=a['duration_s'],two_time_s=b['duration_s'],
            saved_time_s=b['duration_s']-a['duration_s'],
            main_speed_deficit_mm=a['speed_deficit_mm'],two_speed_deficit_mm=b['speed_deficit_mm'],
            saved_deficit_mm=b['speed_deficit_mm']-a['speed_deficit_mm'],
            main_swing_mm=a['outgoing_swing_in_window_mm'],two_swing_mm=b['outgoing_swing_in_window_mm']))
    with (args.out/'all84_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(corners[0]));writer.writeheader();writer.writerows(corners)
    partitions=[]
    for a,b in zip(main['time_partition'],two['time_partition']):
        assert a['label']==b['label']
        partitions.append(dict(label=a['label'],saved_time_s=b['time_s']-a['time_s'],
            length_contribution_s=(b['executed_length_mm']-a['executed_length_mm'])/100.,
            speed_deficit_contribution_s=(b['speed_deficit_mm']-a['speed_deficit_mm'])/100.))
    result=dict(main_time_s=main['complete_time_s'],two_time_s=two['complete_time_s'],
        saved_time_s=two['complete_time_s']-main['complete_time_s'],
        reduction_percent=100.*(1.-main['complete_time_s']/two['complete_time_s']),
        target_time_s=.95*two['complete_time_s'],remaining_gap_s=main['complete_time_s']-.95*two['complete_time_s'],
        length_contribution_s=(two['executed_length_mm']-main['executed_length_mm'])/100.,
        speed_deficit_contribution_s=(two['speed_deficit_mm']-main['speed_deficit_mm'])/100.,
        speed_deficit_not_worse_count=sum(c['saved_deficit_mm']>=-1e-7 for c in corners),
        largest_time_losses=sorted(corners,key=lambda c:c['saved_time_s'])[:10],
        largest_time_gains=sorted(corners,key=lambda c:c['saved_time_s'],reverse=True)[:10],
        partitions=partitions,main_trace_sha256=sha(args.main_trace),two_trace_sha256=sha(args.two_trace),
        ppo_updates=0,new_simulation_steps=0,goal_complete=False,
        scope='Fixed developed dense case; conditional residual comparison with shared backbone; no claim of independent full-method training')
    (args.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['largest_time_losses','largest_time_gains']},indent=2))


if __name__=='__main__':main()
