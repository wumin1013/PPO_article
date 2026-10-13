"""Compact evidence figures; no failed run is represented as a fast completion."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
COLORS={'adaptive':'#246D91','fixed':'#AA6247'}


def save(fig,name):
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(folder/(name+'.'+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)


def main():
    result=json.loads((ROOT/'results'/'summary'/'all_results.json').read_text('utf-8'))
    labels=['Short blocks','Acute turns','Curvature','Line–arc','Closed pocket']
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),gridspec_kw={'width_ratios':[1.3,1]})
    for arm,offset in [('adaptive',-.13),('fixed',.13)]:
        for i,name in enumerate(result['test_paths']):
            rows=[r for r in result['records'] if r['path']==name and r['method']==arm]
            ok=[r for r in rows if r['success']]
            values=[r['time_s'] for r in ok]
            if values:
                spread=np.linspace(-.04,.04,len(values));axes[0].scatter(i+offset+spread,values,s=14,color=COLORS[arm],alpha=.8)
                axes[0].plot([i+offset-.07,i+offset+.07],[np.mean(values)]*2,color=COLORS[arm],lw=2)
            axes[1].plot(i+offset,len(ok)/len(rows) if rows else np.nan,'o',color=COLORS[arm],label=arm.capitalize() if i==0 else None)
            axes[1].annotate(f'{len(ok)}/{len(rows)}',(i+offset,len(ok)/len(rows) if rows else 0),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7,color=COLORS[arm])
    for ax in axes:ax.set_xticks(range(5),labels,rotation=28,ha='right');ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('Complete time among successes (s)');axes[1].set_ylabel('Success fraction');axes[1].set_ylim(-.04,1.2);axes[1].set_yticks(np.arange(0,1.01,.2));axes[1].legend(loc='lower center',fontsize=7)
    axes[0].set_title('a  Individual seeds and means',loc='left',fontsize=9);axes[1].set_title('b  All five attempts per program',loc='left',fontsize=9)
    fig.tight_layout();save(fig,'fresh_unseen_results')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.55))
    for arm in ['adaptive','fixed']:
        curves=[json.loads(p.read_text('utf-8')) for p in sorted((ROOT/'results'/'paired'/'fullrange').glob('seed_*/'+arm+'/training_curve.json'))]
        if not curves:continue
        x=np.array([r['decision'] for r in curves[0]])
        for ax,key in zip(axes,['mean_batch_reward','clip_fraction']):
            y=np.array([[r[key] for r in c] for c in curves])
            for v in y:ax.plot(x,v,color=COLORS[arm],lw=.6,alpha=.2)
            ax.plot(x,y.mean(0),color=COLORS[arm],lw=1.5,label=arm.capitalize())
    axes[0].set_ylabel('Mean rollout reward');axes[1].set_ylabel('PPO clip fraction')
    for ax in axes:ax.set_xlabel('Environment decisions');ax.grid(alpha=.15);ax.legend(fontsize=7)
    axes[0].set_title('a  Five independent training runs',loc='left',fontsize=9);axes[1].set_title('b  Empirical update behavior',loc='left',fontsize=9)
    fig.tight_layout();save(fig,'fresh_training_curves')


if __name__=='__main__':main()
