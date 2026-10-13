"""Render every preregistered complete final program, without selecting outcomes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
NAMES=['Dense folds','Chamfer chain','Rounded triangle','Double bend','Inward channels']

def main():
    folder=ROOT/'prospective/final_v3'
    registration=json.loads((folder/'registration.json').read_text('utf-8'))
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,
                        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.9),layout='constrained')
    receipts=[]
    for i,(row,name,ax) in enumerate(zip(registration['paths'],NAMES,axes.flat)):
        path=folder/row['file']
        if not path.exists():path=folder/'paths'/row['file']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==row['sha256']
        p=np.loadtxt(path,delimiter=',',skiprows=1)
        ax.plot(*p.T,c='#355e80',lw=1.1)
        ax.scatter(*p[0],c='#222222',s=14,zorder=4,label='Start')
        if not row['closed']:ax.scatter(*p[-1],c='#a56c38',s=18,marker='s',zorder=4,label='End')
        ax.set_aspect('equal',adjustable='datalim');ax.margins(.15)
        ax.set_xlabel('$X$ (mm)');ax.set_ylabel('$Y$ (mm)')
        ax.set_title(f'{chr(97+i)}  {name}',loc='left',fontsize=9)
        ax.text(.03,.96,f"{row['length_mm']:.1f} mm",transform=ax.transAxes,va='top',fontsize=7,
                bbox={'facecolor':'white','edgecolor':'none','pad':1,'alpha':.85})
        receipts.append({'id':row['id'],'sha256':digest,'length_mm':row['length_mm']})
    ax=axes.flat[-1];ax.axis('off')
    ax.plot([.10,.23],[.67,.67],color='#355e80',lw=1.2,transform=ax.transAxes)
    ax.text(.28,.67,'Reference path',va='center',transform=ax.transAxes)
    ax.scatter([.16],[.53],c='#222222',s=16,transform=ax.transAxes)
    ax.text(.28,.53,'Start',va='center',transform=ax.transAxes)
    ax.scatter([.16],[.39],c='#a56c38',s=20,marker='s',transform=ax.transAxes)
    ax.text(.28,.39,'End',va='center',transform=ax.transAxes)
    ax.text(.10,.21,'Equal $X/Y$ scale within each panel',fontsize=8,transform=ax.transAxes)
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/'shared_manuscript'/('programs_v3.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    (ROOT/'audit/programs_figure_v3.json').write_text(json.dumps({'sources':receipts,'subset_selection':'none'},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
