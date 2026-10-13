"""Replay actual completed trajectories to inspect state-kernel activation."""
import argparse
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features
from study_io import BASE,write_json
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--tags',nargs='+',required=True);parser.add_argument('--path',type=int,default=104)
    args=parser.parse_args();torch.set_num_threads(1);stage=BASE/args.stage
    md=torch.load(stage/'initial_policy.pt',map_location='cpu',weights_only=False)
    history=read(stage/'state_joint/results.json')['candidates'];out=[]
    for tag in args.tags:
        c=next(c for c in history if c['tag']==tag)
        md['coefficients']=np.array(c['coefficients']).reshape(-1,3)
        records=[r for r in c['records'] if r['path']==args.path]
        data=collect_data(md,md['parent'],records,include_latents=True)
        values=np.max(abs(features(md,data['observations'],data['innovations'])),axis=2)
        row=dict(tag=tag,path=args.path,decisions_ms=(data['decisions']*20).tolist(),kernel_activation=values.tolist())
        out.append(row);np.savez_compressed(stage/(tag+'_replay.npz'),**data)
        print('CLOSED_LOOP_SUPPORT',tag,[(int(d*20),v.tolist()) for d,v in zip(data['decisions'],values) if 260<=d*20<=360],flush=True)
    write_json(stage/'measured_kernel_support_audit.json',dict(records=out,
        actual_decoded_actions_replayed_exactly=True,new_physical_runs=0))


if __name__=='__main__':main()
