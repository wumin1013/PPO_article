"""Retain the submitted figure grammar and correct the method semantics."""
import importlib.util
import json
import math
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,Polygon
from shapely.geometry import LineString
from study_io import BASE,ROOT,register,receipt,write_json,sha
from geometry_spec import Corridor

BLUE='#234e75'
LIGHT='#f4f8fc'
GREEN='#eaf5e8'
GOLD='#fff3d6'
PURPLE='#eee5f8'
RED='#cf243b'
mpl.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none',
    'mathtext.fontset':'stix','pdf.fonttype':42,'axes.linewidth':.7})


def box(ax,x,y,w,h,title,body='',color=LIGHT,edge=BLUE,fontsize=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.15',
        fc=color,ec=edge,lw=1.05))
    ax.text(x+w/2,y+h-.22,title,ha='center',va='top',weight='bold',fontsize=fontsize)
    if body:
        ax.text(x+w/2,y+h/2-.12,body,ha='center',va='center',fontsize=fontsize-.2,
            linespacing=1.22)


def arrow(ax,start,end,*,dashed=False,color=BLUE,lw=1.5):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',
        color=color,lw=lw,linestyle=(0,(5,4)) if dashed else '-',shrinkA=0,shrinkB=0))


def network(ax,left,bottom,width,height,*,labels=False):
    counts=[4,4,4,3]
    levels=[]
    for row,count in enumerate(counts):
        yy=bottom+height*(1-row/3)
        xx=np.linspace(left,left+width,count)
        levels.append([(x,yy) for x in xx])
    for previous,following in zip(levels[:-1],levels[1:]):
        for a in previous:
            for b in following:ax.plot([a[0],b[0]],[a[1],b[1]],color=BLUE,lw=.35,alpha=.55,zorder=1)
    for row,points in enumerate(levels):
        for col,(x,y) in enumerate(points):
            ax.add_patch(Circle((x,y),.12,ec=BLUE,fc=['#82d5c5','#79a8d0','#79a8d0','#efb649'][row],lw=.8,zorder=2))
            if labels and row==3:
                ax.text(x,y-.23,[r'$\delta_m$',r'$f_m$',r'$D_m$'][col],ha='center',va='top',fontsize=10)


def reference(ax,x,y,w,h,smoothed=False,sampled=False):
    points=np.array([[0.,.65],[.28,.22],[.52,.86],[.69,.12],[1.,.55]])
    xy=points*[w,h]+[x,y]
    if not sampled:
        ax.plot(*xy.T,color='black',lw=1.25,marker='o' if not smoothed else None,ms=2.8)
    if smoothed:
        output=[xy[0]]
        for i in range(1,len(xy)-1):
            before=xy[i]*.78+xy[i-1]*.22
            after=xy[i]*.78+xy[i+1]*.22
            output.append(before)
            for t in np.linspace(0,1,25):output.append((1-t)**2*before+2*t*(1-t)*xy[i]+t*t*after)
        output.append(xy[-1])
        if sampled:
            coordinates=np.asarray(output)
            arcs=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(coordinates,axis=0),axis=1))]
            sample=np.linspace(0.,arcs[-1],90)
            commands=np.c_[np.interp(sample,arcs,coordinates[:,0]),np.interp(sample,arcs,coordinates[:,1])]
            ax.plot(*commands.T,color=RED,lw=0,marker='.',ms=1.2)
        else:
            ax.plot(*np.asarray(output).T,color=RED,lw=1.4)


def figure_one(folder):
    fig,ax=plt.subplots(figsize=(7.15,5.45))
    ax.set(xlim=(0,14.4),ylim=(0,10.6));ax.axis('off')
    ax.text(.12,10.28,'(a) Conventional two-step planning',color=RED,fontsize=11.2)
    ax.text(7.25,10.28,'(b) J-NNC closed-loop planning',color=RED,fontsize=11.2)
    for x in [.5,7.65]:
        box(ax,x,8.3,5.6,1.45,'G-code input')
        reference(ax,x+.38,8.48,4.86,.72)
    box(ax,.5,6.12,5.6,1.5,'Path smoothing')
    reference(ax,.88,6.28,4.86,.84,True)
    box(ax,.5,3.97,5.6,1.48,'Feedrate planning')
    xx=np.linspace(.95,5.65,180)
    t=np.linspace(0,1,len(xx))
    yy=4.19+.55*np.minimum(np.minimum(t*8,(1-t)*8),1)
    yy-=.13*np.exp(-((t-.3)/.035)**2)+.17*np.exp(-((t-.57)/.035)**2)+.14*np.exp(-((t-.77)/.035)**2)
    ax.plot(xx,yy,color=RED,lw=1.5)
    arrow(ax,(.9,4.14),(5.7,4.14),color='black',lw=.8)
    box(ax,.5,1.82,5.6,1.48,'Interpolation')
    reference(ax,.88,1.98,4.86,.8,True,True)
    box(ax,.5,.15,5.6,1.0,'Position commands')
    for y0,y1 in [(8.3,7.64),(6.12,5.47),(3.97,3.32),(1.82,1.17)]:arrow(ax,(3.3,y0),(3.3,y1))
    box(ax,7.65,4.15,5.6,3.47,'PPO policy',color=GOLD)
    network(ax,8.85,5.,3.2,1.92,labels=True)
    for x,label in zip([8.85,10.45,12.05],['Steering','Feedrate','Preview']):
        ax.text(x,4.34,label,ha='center',fontsize=8.4)
    box(ax,7.65,1.82,5.6,1.65,'Kinematic constraint module',
        'Planar V/A/J projection\nAdmissible acceleration release',color=GREEN)
    box(ax,7.65,.15,5.6,1.0,'Position commands')
    arrow(ax,(10.45,8.3),(10.45,7.64));arrow(ax,(10.45,4.15),(10.45,3.49));arrow(ax,(10.45,1.82),(10.45,1.17))
    ax.plot([13.25,13.75,13.75,13.25],[.65,.65,6.1,6.1],color=BLUE,lw=1.0,ls=(0,(4,3)))
    arrow(ax,(13.75,6.1),(13.25,6.1),dashed=True,lw=1.)
    ax.text(14.04,3.5,'Motion feedback',rotation=90,va='center',ha='center',fontsize=8)
    fig.subplots_adjust(0,.01,1,.99)
    fig.savefig(folder/'Figure_1.svg');plt.close(fig)


def figure_two(folder):
    epsilon=1.25
    theta=math.radians(130.)
    ref=np.array([[-16.,0.],[0.,0.],[13*math.cos(theta),13*math.sin(theta)]])
    domain=LineString(ref).buffer(epsilon,quad_segs=128)
    boundary=np.asarray(domain.exterior.coords)
    radius=.90*epsilon/(1-math.cos(theta/2))
    offset=radius*math.tan(theta/2)
    center=np.array([-offset,radius])
    alpha=np.linspace(-math.pi/2,-math.pi/2+theta,161)
    arc=center+radius*np.c_[np.cos(alpha),np.sin(alpha)]
    path=np.vstack([ref[0],arc,ref[-1]])
    corridor=Corridor(ref)
    assert all(corridor.segment_inside(a,b) for a,b in zip(path[:-1],path[1:]))
    fig,ax=plt.subplots(figsize=(6.7,4.5))
    ax.fill(*boundary.T,fc='#d4e4f2',zorder=0)
    ax.plot(*boundary.T,color='#3477a4',lw=1.2,ls=(0,(5,3)))
    ax.plot(*ref.T,'ko-',lw=1.55,ms=4,zorder=3)
    ax.plot(*path.T,color='#70ae42',lw=2,ls=(0,(1.2,1.2)),zorder=4)
    for i,point in enumerate(ref):
        shift=np.array([-.85,.25]) if i==2 else np.array([.2,-.60])
        ax.text(*(point+shift),rf'$p_{i}$',fontsize=12)
    gate=corridor.gates[0]
    axis=np.array([-gate.normal[1],gate.normal[0]])
    extent=epsilon/math.cos(theta/2)
    ends=np.array([-axis*epsilon,axis*extent])
    assert all(corridor.gate_contains(gate,point) for point in np.linspace(*ends,33))
    ax.plot(*ends.T,color='#b77aa8',lw=1.0,ls=(0,(3,3)),zorder=2)
    ax.text(-1.8,-1.43,r'$G_1$',color='#86577d',fontsize=11)
    point=arc[64]
    ax.plot(*point,'o',color='#70ae42',ms=4.8,zorder=5)
    ax.annotate(r'Current position $x_m$',xy=point,xytext=(-10.8,4.0),
        arrowprops=dict(arrowstyle='->',color='#568930',lw=1.),fontsize=10)
    ax.annotate('Reference tool path',xy=(-12.,0.),xytext=(-15.,2.5),
        arrowprops=dict(arrowstyle='->',color='black',lw=1.),fontsize=10)
    ax.annotate(r'Geometric feasible region $\Omega$',xy=(-11.,-.6),xytext=(-15.5,-3.35),
        arrowprops=dict(arrowstyle='->',color='#3477a4',lw=1.),fontsize=10)
    ax.annotate('',xy=(-14.,epsilon),xytext=(-14.,0),arrowprops=dict(arrowstyle='<->',color=RED,lw=1.))
    ax.text(-13.68,.46,r'$\varepsilon$',fontsize=13,color=RED)
    ax.annotate('Inward transition',xy=arc[105],xytext=(-3.1,5.0),
        arrowprops=dict(arrowstyle='->',color='#568930',lw=1.),fontsize=10)
    ax.set_aspect('equal');ax.axis('off')
    ax.set(xlim=(-18.5,3.2),ylim=(-4.3,11.9))
    fig.subplots_adjust(0,0,1,1)
    fig.savefig(folder/'Figure_2.svg');plt.close(fig)
    write_json(folder/'geometry_illustration.json',dict(reference=ref.tolist(),epsilon=epsilon,
        turn_degrees=130,schematic_radius=radius,sample_command_segments_checked=len(path)-1,
        every_segment_inside=True,not_a_learned_or_kinematically_audited_motion=True,
        boundary_round_approximation_sagitta_mm=epsilon*(1-math.cos(math.pi/512))))


def figure_three(folder):
    fig,ax=plt.subplots(figsize=(7.2,4.25))
    ax.set(xlim=(0,15.4),ylim=(0,8.6));ax.axis('off')
    box(ax,.65,6.05,2.25,1.65,'G-code input','Reference path\nContour tolerance',fontsize=8.2)
    box(ax,3.35,5.8,2.6,2.15,'Geometry encoding','Ordered projection\nFixed multi-scale\nforward queries',fontsize=8.2)
    box(ax,6.4,6.05,2.05,1.65,'State',r'$s_m\in\mathbb{R}^{72}$'+'\nMotion and\ngeometry',fontsize=8.2)
    box(ax,8.9,5.8,2.65,2.15,'PPO policy',color=GOLD,fontsize=8.4)
    network(ax,9.48,6.2,1.45,1.05)
    box(ax,12.05,6.05,2.75,1.65,'Joint actions',r'$\delta_m,\ f_m,\ D_m$'+'\nDecision: 20 ms',fontsize=8.4)
    for start,end in [(2.9,3.35),(5.95,6.4),(8.45,8.9),(11.55,12.05)]:arrow(ax,(start,6.85),(end,6.85),lw=1.4)
    box(ax,11.15,4.1,3.65,1.1,'Guidance and tracking','Interpolation: 1 ms',fontsize=8.2)
    arrow(ax,(13.42,6.05),(13.42,5.22),lw=1.4)
    box(ax,8.9,1.65,5.9,1.88,'Vector KCM',color=GREEN,fontsize=9)
    for x,label in [(9.15,'Candidate\nprojection'),(11.0,'Continuation\ncheck'),(12.85,'Verified\nrelease step')]:
        ax.add_patch(FancyBboxPatch((x,2.22),1.62,.66,boxstyle='round,pad=.02,rounding_size=.1',fc=GOLD,ec=BLUE,lw=.75))
        ax.text(x+.81,2.55,label,ha='center',va='center',fontsize=7.5)
    ax.text(11.85,1.90,r'$\|\mathbf{v}\|\leq V_{\max},\quad\|\mathbf{a}\|\leq A_{\max},\quad\|\mathbf{j}\|\leq J_{\max}$',ha='center',fontsize=10)
    arrow(ax,(13.42,4.1),(13.42,3.55),lw=1.4)
    box(ax,6.2,1.95,2.05,1.4,'Position\ncommand',r'$x_{k+1}$',fontsize=8)
    box(ax,3.35,1.9,2.2,1.5,'Execution update','Position and motion\nTraversal progress',fontsize=8.2)
    arrow(ax,(8.9,2.65),(8.27,2.65),lw=1.4);arrow(ax,(6.2,2.65),(5.57,2.65),lw=1.4)
    arrow(ax,(4.45,3.42),(4.45,5.78),lw=1.4)
    ax.text(4.63,4.60,'State feedback',fontsize=8,rotation=90,va='center')
    box(ax,2.3,.15,3.7,1.10,'Reward','Time and progress potential\nFailure penalty',color=PURPLE,edge='#745597',fontsize=8)
    arrow(ax,(4.45,1.9),(4.45,1.27),dashed=True,lw=1.15)
    ax.plot([2.27,.15,.15,10.22],[.70,.70,8.35,8.35],color=BLUE,lw=1.15,ls=(0,(5,4)))
    arrow(ax,(10.22,8.35),(10.22,7.97),dashed=True,lw=1.15)
    ax.text(5.5,8.46,'PPO parameter update',ha='center',va='bottom',fontsize=8.1,color=BLUE)
    arrow(ax,(7.15,.65),(8.0,.65),lw=1.3);ax.text(8.15,.65,'Execution',va='center',fontsize=8)
    arrow(ax,(10.3,.65),(11.15,.65),dashed=True,lw=1.15);ax.text(11.3,.65,'Training feedback',va='center',fontsize=8)
    fig.subplots_adjust(0,0,1,.98)
    fig.savefig(folder/'Figure_3.svg');plt.close(fig)


if __name__=='__main__':
    originals=ROOT/'论文项目/submission_package_EAAI/05_figures'
    converter_path=ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources=[__file__,BASE/'method_figure_contract_v1.md',BASE/'geometry_spec.py',BASE/'joint_runtime.py',
        BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',
        BASE/'manuscript_joint_v1/methodology.tex',converter_path]
    sources += [originals/f'Figure_{i}.pdf' for i in [1,2,3]]
    stage=register('method_figures_v2',sources,dict(figures=3,svg_conversions=3,
        geometric_illustration_segments=162,gate_illustration_point_checks=33,models=0,physics_ticks=0),
        'Retain the submitted method-figure structures, correcting action, vector-limit and tolerance semantics; final PDF visual QA remains separate.')
    figure_one(stage);figure_two(stage);figure_three(stage)
    spec=importlib.util.spec_from_file_location('method_svg_converter',converter_path)
    converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    output_root=Path('D:/PPO_article_joint_runs/method_figures_v2')
    output_root.mkdir(parents=True,exist_ok=False)
    outputs=[]
    for index in [1,2,3]:
        svg=stage/f'Figure_{index}.svg'
        directory=output_root/f'Figure_{index}'
        converter.convert(svg,directory)
        pdf=directory/f'Figure_{index}.pdf'
        shutil.copy2(pdf,stage/pdf.name)
        outputs.append(dict(figure=index,svg_sha256=sha(svg),pdf_sha256=sha(pdf),
            preview=str(directory/'preview.png'),conversion=str(directory/'conversion.json')))
    receipt(stage,dict(status='three_method_svgs_converted_pending_full_visual_QA',
        outputs=outputs,models=0,physics_ticks=0,old_figures_preserved=True,final_manuscript_QA=False))
    print(outputs)
