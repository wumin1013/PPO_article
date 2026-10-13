"""Render the eight registered examples from saved commands, with explicit failures."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from study_io import BASE,sha,write_json

OUT=BASE/'PPO_article_manuscript';FIG=OUT/'figures'
BLUE='#0072B2';GREY='#666666';RED='#B23A48'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(FIG/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def boundary(ax,reference):
    region=LineString(reference).buffer(1.25,resolution=48)
    for poly in [region] if region.geom_type=='Polygon' else region.geoms:
        xy=np.asarray(poly.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='#EDF3F7',zorder=0)
        ax.plot(xy[:,0],xy[:,1],color='#829EAF',lw=.6,ls='--')
        for ring in poly.interiors:
            xy=np.asarray(ring.coords);ax.fill(xy[:,0],xy[:,1],color='white',zorder=0)
            ax.plot(xy[:,0],xy[:,1],color='#829EAF',lw=.6,ls='--')
    ax.plot(reference[:,0],reference[:,1],color=GREY,lw=.8,ls=':',label='Reference')


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,
        'path.simplify':False,'legend.frameon':False})
    audit=read(BASE/'paper_evidence_review/candidate_audit.json')
    current=next(c for c in audit['candidates'] if c['label']=='worst_guard')
    records={r['path']:r for r in current['records']}
    sources=[];notes=[]
    for number,index in enumerate([48,3,4,11,14,204,32,'long'],1):
        if index=='long':
            path=BASE/'paper_evidence_review/public_long/trace.npz';success=False;label='Public mixed path: failed prefix'
            record=read(BASE/'paper_evidence_review/public_long/evaluation.json');center_index=8
        else:
            record=records[index];path=BASE/record['trace'];success=True;label=f'Training path {index}'
            assert sha(path)==record['trace_sha256'];center_index=1
        z=np.load(path);ref=z['reference'];rows=z['rows'];xy=np.vstack([ref[0],rows[:,:2]])
        t=np.arange(1,len(rows)+1)*.001;color=BLUE if success else RED
        fig=plt.figure(figsize=(7.15,5.7));grid=fig.add_gridspec(3,2,height_ratios=[1.15,1,1],hspace=.56,wspace=.30)
        axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1]),fig.add_subplot(grid[2,:])]
        for ax in axes[:2]:
            boundary(ax,ref);ax.plot(xy[:,0],xy[:,1],color=color,lw=1.1,label='Learned policy' if success else 'Failed prefix')
            ax.set_aspect('equal',adjustable='box');ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)')
        center=ref[min(center_index,len(ref)-1)] if len(ref)>2 else .5*(ref[0]+ref[-1])
        axes[1].set_xlim(center[0]-10,center[0]+10);axes[1].set_ylim(center[1]-10,center[1]+10)
        axes[0].plot(*ref[0],marker='^',ms=4,color='black');axes[0].plot(*xy[-1],marker='o' if success else 'x',ms=5,color=color)
        for ax,cols,limit,ylabel in zip(axes[2:],[(2,4),(4,6),(6,8)],[100,2000,62500],
            [r'$\|\mathbf{v}\|/100$',r'$\|\mathbf{a}\|/2000$',r'$\|\mathbf{j}\|/62500$']):
            values=np.linalg.norm(rows[:,cols[0]:cols[1]],axis=1)/limit
            ax.plot(t,values,color=color,lw=.65);ax.axhline(1,color=GREY,lw=.65,ls='--')
            ax.set_ylim(-.025,max(1.06,float(values.max())+.04));ax.set_xlim(0,t[-1])
            ax.set_xlabel('Time (s)');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
        for ax,title in zip(axes,['a  Full trajectory','b  Fixed 20 mm window','c  Velocity','d  Acceleration','e  Jerk']):
            ax.set_title(title,loc='left')
        status=f'Complete, T = {t[-1]:.3f} s' if success else f'FAILURE at {t[-1]:.3f} s; complete time undefined'
        fig.text(.5,.99,status,ha='center',va='top',fontsize=9,color=color)
        name=f'case_{number}';save(fig,name)
        source=dict(figure=name,case=index,trace=str(path.relative_to(BASE)),trace_sha256=sha(path),
            success=success,complete_time_s=t[-1] if success else None,
            all_1ms_samples_used=True,local_window_mm=20,window_center=center.tolist(),
            physics_limits=[100,2000,62500],new_actor_calls=0,new_simulation_ticks=0)
        sources.append(source)
        notes.append(f'## {name}: {label}\n\n'+
            ('完整轨迹包含加速、弯部运动、制动与停稳。' if success else '全路径参考与失败前缀明确分开；未完成覆盖和制动，前缀速度不能用于完整效率。')+
            '\n\n(a)完整XY；(b)预定20 mm局部窗与1.25 mm欧氏走廊；(c–e)全部1 ms平面向量V/A/J，相同硬上限虚线。'+
            f' 轨迹SHA256：`{source["trace_sha256"]}`。统计单位为该完整路径；不把采样点当独立样本。\n')
    # Detailed action/shape traces for two predeclared illustrative cases.
    fig,axes=plt.subplots(5,2,figsize=(7.15,8.0),sharex='col',layout='constrained')
    for col,index in enumerate([3,204]):
        z=np.load(BASE/records[index]['trace']);r=z['rows'];a=z['actions']
        tt=np.arange(len(a))*.020;t=np.arange(1,len(r)+1)*.001
        for k,(unit,label) in enumerate([(180/np.pi,'Steering (deg)'),(1.,'Requested feed (mm/s)'),(1.,'Preview (mm)')]):
            axes[k,col].step(tt,a[:,k]*unit,where='post',color=BLUE,lw=.8)
            axes[k,col].set_ylabel(label);axes[k,col].grid(alpha=.12)
        ref=z['reference'];u=ref[1]-ref[0];u=u/np.linalg.norm(u)
        normal=np.array([-u[1],u[0]]);v=r[:,2:4];acc=r[:,4:6];speed=np.linalg.norm(v,axis=1)
        heading=np.rad2deg(np.unwrap(np.arctan2(v@normal,v@u)));heading[speed<=5]=np.nan
        curvature=np.full(len(r),np.nan);moving=speed>5
        curvature[moving]=(v[moving,0]*acc[moving,1]-v[moving,1]*acc[moving,0])/speed[moving]**3
        axes[3,col].plot(t,heading,color=BLUE,lw=.8);axes[3,col].set_ylabel('Heading (deg)')
        axes[4,col].plot(t,curvature,color=BLUE,lw=.8);axes[4,col].set_ylabel(r'Curvature (mm$^{-1}$)')
        axes[4,col].set_xlabel('Time (s)')
        axes[0,col].set_title(f'Path {index}: '+('+35 deg' if index==3 else '-155 deg, perturbed'))
        for ax in axes[:,col]:ax.axhline(0,color=GREY,lw=.4);ax.set_xlim(0,t[-1])
    save(fig,'actions_and_shape')
    notes.append('## actions_and_shape\n\n对应既定+35°和−155°扰动案例；动作按20 ms阶梯绘制，航向/曲率用原1 ms速度和加速度计算，速度≤5 mm/s时方向量留空。两端未拼接，不平滑或抹除反向修正。\n')
    write_json(OUT/'figure_sources.json',dict(cases=sources,
        action_figure_paths=[3,204],action_figure_sources=[records[i]['trace_sha256'] for i in [3,204]],
        visual_precedent='Previously user-designated five-panel full XY/local XY/V/A/J layout; inspected existing rendered S2 layer. This is not claimed as a verified published top-journal precedent.',
        backend='Python matplotlib only',global_dataset_statistics_not_inferred_from_examples=True))
    (OUT/'逐图说明.md').write_text('# PPO_article逐图说明\n\n'+'\n'.join(notes),encoding='utf-8')
    print('FIGURES_COMPLETE',len(sources)+1,flush=True)


if __name__=='__main__':main()
