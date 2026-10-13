"""Render data-bound local mechanism review; no manuscript file is edited."""
import os
os.environ['MPLBACKEND']='Agg'
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from evaluate_historical_dense import ROOT,write,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'direct14_actual_arrival_fit_round55/mechanism_0001');parser.add_argument('--name',default='figure_review_v2');args=parser.parse_args()
    assert Path(args.name).name==args.name
    source=args.source.resolve()
    stage=source/args.name;stage.mkdir(exist_ok=False)
    shutil.copy2(__file__,stage/Path(__file__).name)
    contract=dict(conclusion='Local steering and preview requests precede executed turns;feed decreases at sharper corners while the obtuse corner maintains cruise.',
        evidence='Same14input candidate,existing135/90/45degree representatives,raw requests and physical outputs,bothdt.',
        geometry='User-prescribed six synchronized signal rows by three angle columns;local progress window only.',
        precedent='No external precedent claim. The user explicitly prescribed this geometry and restricts work to the PPO project;this research review follows that contract.',
        archetype='quantitative grid',backend='Python matplotlib',size_in=[6.6,8.1],exports=['pdf','svg','png'],
        integrity='No smoothing,synthetic traces or errorbars. Requests shown as held steps;executed signals show recorded physical samples.',
        limits='Descriptive mechanism evidence,not causal attribution or independent repeats. These are two numerical resolutions of one model.')
    write(stage/'figure_contract.json',contract)
    paths=[source/f'original_{key}_representative_rows.npz' for key in ['1ms','half_ms']]
    data=[]
    for path in paths:
        with np.load(path) as d:data.append({k:d[k].copy() for k in d.files})
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig,axes=plt.subplots(6,3,figsize=(6.6,8.1),sharex=True)
    fig.subplots_adjust(left=.13,right=.99,top=.92,bottom=.065,wspace=.30,hspace=.28)
    ids=['corner_01','corner_02','corner_06'];angles=[135,90,45]
    colors=['#27628E','#C57942'];styles=['-','--'];labels=['1 ms','0.5 ms']
    row_names=['Steering\n(deg)','Requested feed\n(mm s$^{-1}$)','Preview\n(mm)',
        'Heading change\n(deg)','Curvature magnitude\n(mm$^{-1}$)','Executed speed\n(mm s$^{-1}$)']
    kmax=max(float(abs(d[c+'_physical'][:,4]).max()) for d in data for c in ids)
    for col,(ident,angle) in enumerate(zip(ids,angles)):
        for j,d in enumerate(data):
            physical=d[ident+'_physical'];requests=d[ident+'_requests']
            for row in range(6):
                ax=axes[row,col]
                if row<3:
                    y=requests[:,row+1].copy()
                    if row==0:y=np.rad2deg(y)
                    ax.step(requests[:,0],y,where='post',color=colors[j],linestyle=styles[j],linewidth=.9,label=labels[j])
                else:
                    y=physical[:,row].copy()
                    if row==4:y=abs(y)
                    ax.plot(physical[:,0],y,color=colors[j],linestyle=styles[j],linewidth=.85)
        axes[0,col].set_title(f'({chr(97+col)})  {angle}'+r'$^\circ$',fontsize=10,fontweight='bold',pad=7)
        for row in range(6):
            ax=axes[row,col];ax.axvline(0,color='#888888',linestyle=':',linewidth=.65,zorder=0)
            ax.grid(axis='y',color='#E6E6E6',linewidth=.45);ax.set_xlim(-16,20)
            ax.tick_params(length=2.5,width=.6,pad=2)
            if col==0:ax.set_ylabel(row_names[row],labelpad=5)
        axes[0,col].set(ylim=(-15,15),yticks=[-10,0,10])
        axes[1,col].set(ylim=(44,105),yticks=[50,75,100])
        axes[2,col].set(ylim=(0,13),yticks=[0,6,12])
        turn=180-angle
        heading_max=max(float(d[ident+'_physical'][:,3].max()) for d in data)
        axes[3,col].set(ylim=(-5,max(turn+8,heading_max+4)),yticks=[0,turn/2,turn])
        axes[4,col].set_ylim(0,kmax*1.08)
        axes[5,col].set(ylim=(40,105),yticks=[50,75,100],xticks=[-15,0,15])
    fig.supxlabel('Reference progress relative to corner (mm)',y=.015,fontsize=8.5)
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',bbox_to_anchor=(.56,.983),ncol=2,handlelength=2.8,fontsize=8)
    outputs=[]
    for ext in ['pdf','svg','png']:
        path=stage/f'mechanism_review.{ext}';fig.savefig(path,dpi=220,facecolor='white');outputs.append(path)
    plt.close(fig)
    write(stage/'data_binding.json',dict(source_sha256={str(p):sha(p) for p in paths},
        audit_sha256=sha(source/'audit.json'),script_sha256=sha(__file__),
        outputs={p.name:sha(p) for p in outputs},representatives=ids,manuscript_and_original_three_figures_unchanged=True,
        submission_ready=False,visual_inspection_pending=True))
    print('MECHANISM_REVIEW',stage,flush=True)


if __name__=='__main__':main()
