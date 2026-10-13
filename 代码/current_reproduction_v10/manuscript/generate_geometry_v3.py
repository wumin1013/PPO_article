"""Exact Euclidean tube schematic; illustrative coordinates, not run evidence."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'shared_manuscript'

def main():
    plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,
                         'svg.fonttype':'none','mathtext.fontset':'stix'})
    reference=np.array([[0.,0.],[16.,0.],[28.,12.]])
    radius=2.
    tube=LineString(reference).buffer(radius,quad_segs=128,cap_style='round',join_style='round')
    fig,ax=plt.subplots(figsize=(7.1,3.15))
    boundary=np.asarray(tube.exterior.coords)
    ax.fill(boundary[:,0],boundary[:,1],color='#e3ecf4',zorder=0)
    ax.plot(boundary[:,0],boundary[:,1],color='#628aa4',lw=1.,ls='--')
    ax.plot(*reference.T,color='#253b50',lw=1.8,marker='o',ms=4.)
    for point,label,offset in zip(reference,['$p_1$','$p_2$','$p_3$'],[(0,-16),(0,-16),(0,12)]):
        ax.annotate(label,point,xytext=offset,textcoords='offset points',ha='center')
    current=np.array([12.,1.2])
    ax.scatter(*current,s=28,c='#a56c38',zorder=4)
    ax.plot([12,12],[0,1.2],color='#a56c38',lw=1.)
    ax.annotate('$x_k$',current,xytext=(-2,10),textcoords='offset points',ha='center')
    ax.annotate('$d(x_k,\\Gamma)$',(12,.6),xytext=(7,5),fontsize=10,
                arrowprops={'arrowstyle':'-','lw':.6,'color':'#555'})
    center=np.array([23.,7.]);normal=np.array([-1.,1.])/np.sqrt(2.)
    left,right=center+radius*normal,center-radius*normal
    ax.annotate('',xy=left,xytext=right,arrowprops={'arrowstyle':'<->','color':'#42576a','lw':1.})
    ax.text(25,6.5,'$\\varepsilon$',color='#253b50',fontsize=13)
    ax.annotate('Reference $\\Gamma$',(5,0),xytext=(1,4.8),
                arrowprops={'arrowstyle':'-','color':'#555','lw':.7})
    ax.annotate('Euclidean tolerance band $\\Omega$',(16.6,-1.8),xytext=(17.5,-5.),
                ha='center',arrowprops={'arrowstyle':'-','color':'#555','lw':.7})
    ax.text(1,-4.8,'$d(x,\\Gamma)\\leq\\varepsilon/2$',fontsize=11)
    ax.set_aspect('equal');ax.set_xlim(-3,31);ax.set_ylim(-6,15);ax.axis('off')
    fig.subplots_adjust(0,0,1,1)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/('Figure_2.'+ext),dpi=200,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
    (ROOT/'audit/geometry_figure_v3.json').write_text(json.dumps({
        'kind':'mathematical schematic, not experimental data',
        'reference':reference.tolist(),'illustrative_radius':radius,
        'geometry':'finite-segment Euclidean tube with round caps and joins',
        'change':'replace mitered conceptual polygon by actual distance-defined corridor',
        'backend':'Python matplotlib/shapely','outputs':['Figure_2.pdf','Figure_2.svg','Figure_2.png']
    },indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
