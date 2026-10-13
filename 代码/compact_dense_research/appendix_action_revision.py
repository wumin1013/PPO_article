"""Add one compact appendix trace and clarify the contour-deviation table header."""
import json
import hashlib
import shutil
import sys
import difflib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BASE=Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1')
OLD=BASE/'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'
PREV=OLD/'academic_figures_20260924'
OUT=OLD/'appendix_action_20260924'
WORK=ROOT/'lowdim_original_manuscript_work/appendix_action_revision'
FIG=OUT/'figures'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def source_text(old,lang):
    if lang=='en':
        assert old.count('Max. distance (mm)')==1
        t=old.replace('Max. distance (mm)',r'\shortstack{Maximum contour\\deviation (mm)}')
        t=t.replace('Maximum distance covers complete command intervals.',
                    'Maximum contour deviation covers complete command intervals.')
        caption=(r'Action--motion correspondence for J-NNC at the same $90^\circ$/C02 corner as Fig.~\ref{fig:full_utilization}. '
            r'The left column shows the decoded steering correction, feedrate request and Adaptive Preview distance, held between 20 ms policy updates; '
            r'the right column shows executed heading, speed and signed curvature reconstructed from the 1 ms position record. '
            r'All panels share time relative to the first execution sample whose ordered reference progress reaches the corner ($t=0$). Heading is measured from the incoming tangent; '
            r'steering, heading and curvature are positive toward the inside of the turn. Horizontal dotted lines mark the outgoing $90^\circ$ heading and the 99 mm/s cruise threshold. Signals are shown without smoothing. '
            r'The panels align the three requests with their joint motion response.')
    else:
        assert old.count('最大距离（mm）')==1
        t=old.replace('最大距离（mm）','最大轮廓偏差（mm）')
        t=t.replace('最大距离覆盖完整运动指令区间。','最大轮廓偏差覆盖完整运动指令区间。')
        caption=(r'J-NNC在正文图~\ref{fig:full_utilization}相同$90^\circ$/C02角点的动作与运动对应关系。'
            r'左列为解码后的转向修正、进给请求和自适应前瞻距离，在20 ms策略更新之间保持；'
            r'右列为根据1 ms位置记录重建的实际航向、速度和带符号曲率。'
            r'所有子图共用时间轴，以有序参考进度首次达到角点的执行采样时刻为$t=0$。航向以入弯切向为基准，转向、航向和曲率均以内侧为正。'
            r'水平点线分别标出出弯$90^\circ$航向和99 mm/s巡航阈值。所有信号均未平滑处理，各子图对应三项请求共同产生的运动响应。')
    block='\n'.join([r'\begin{figure}[!htbp]',r'\centering',r'\renewcommand{\thefigure}{B.\arabic{figure}}',
        r'\setcounter{figure}{0}',r'\includegraphics[width=0.96\linewidth]{Figure_B1.pdf}',
        r'\caption{'+caption+'}',r'\label{fig:action_correspondence}',r'\end{figure}',r'\clearpage',''])
    marker=r'\setlength{\bibsep}'
    assert t.count(marker)==1
    return t.replace(marker,block+'\n'+marker),block


def prepare():
    OUT.mkdir(exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'delivery_manifest.json').exists(), 'Preserve any existing delivery before revising.'
    if not FIG.exists():shutil.copytree(PREV/'figures',FIG)
    contract=dict(core_conclusion='Recorded three-action requests and the executed C02 transition share one time base.',
        scope='One compact appendix figure; the nine main figures and all numeric results remain unchanged.',
        evidence_chain=['20ms decoded steering/feed/preview requests','1ms XY-derived heading/speed/curvature'],
        geometry='3 rows by 2 columns, requests left and executed response right; one prescribed C02 corner',
        archetype='quantitative grid',backend='Python/matplotlib',width_inches=7.25,height_inches=4.7,
        visual_precedent='User-directed compact reduction of the existing, data-bound action-mechanism figure; preserve the established manuscript visual grammar.',
        export=['PDF','SVG with editable text','PNG','CSV'],no_smoothing=True,
        scope_boundary='Temporal correspondence of joint requests and response, not an isolated-action causal effect.')
    write(OUT/'appendix_figure_contract.json',contract)
    b=read(OLD/'paper_result_binding.json');r=b['methods']['J-NNC']['original'][0]
    case=b['cases']['original']
    for p,digest in [(r['trace'],r['trace_sha256']),(r['metrics'],r['metrics_sha256']),
                     (case['geometry'],case['geometry_sha256']),(case['reference'],case['reference_sha256'])]:
        assert sha(p)==digest
    z=np.load(r['trace']);dt=float(z['dt_s']);assert dt==.001
    ref=np.loadtxt(case['reference'],delimiter=',',skiprows=1)
    xy=z['rows'][:,:2];v=np.diff(np.vstack([ref[0],xy]),axis=0)/dt
    a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
    times=(np.arange(len(xy))+1)*dt
    geometry=read(case['geometry']);c=next(q for q in geometry['sharp_corners'] if q['id']=='corner_02')
    metric=read(r['metrics']);metric=metric.get('metrics',metric)
    cm=next(q for q in metric['corners'] if q['id']=='corner_02')
    cache_path=ROOT/'lowdim_original_manuscript_work/academic_figure_revision/plot_cache'/f'original_Joint_{r["trace_sha256"][:12]}.npz'
    cache=np.load(cache_path);assert np.array_equal(cache['rows'][:,:2],xy)
    tc=times[np.flatnonzero(cache['progress']>=c['reference_s_mm'])[0]]
    i=c['reference_vertex_index'];tangent=ref[i]-ref[i-1];tangent=tangent/np.linalg.norm(tangent)
    turn_sign=np.sign(c['signed_heading_change_deg']);normal=turn_sign*np.array([-tangent[1],tangent[0]])
    keep=abs(times-tc)<=.35;vl=v[keep];al=a[keep];speed=np.linalg.norm(vl,axis=1)
    heading=np.rad2deg(np.unwrap(np.arctan2(vl@normal,vl@tangent)))
    curvature=turn_sign*(vl[:,0]*al[:,1]-vl[:,1]*al[:,0])/speed**3
    ta=z['decision_times_s']-tc;actions=z['actions'];assert actions.shape[1]==3
    ia=max(0,np.searchsorted(ta,-.35,side='right')-1);ib=min(len(ta),np.searchsorted(ta,.35,side='right')+1)
    at=ta[ia:ib];ap=actions[ia:ib].copy();ap[:,0]=turn_sign*np.rad2deg(ap[:,0])
    tt=times[keep]-tc
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8.6,
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(3,2,figsize=(7.25,4.7),sharex=True)
    color='#17689A'
    requests=[(ap[:,0],'Steering correction (°)'),(ap[:,1],'Feedrate request (mm/s)'),(ap[:,2],'Preview distance (mm)')]
    actual=[(heading,'Executed heading (°)'),(speed,'Executed speed (mm/s)'),(curvature,r'Signed curvature (mm$^{-1}$)')]
    for row in range(3):
        ax[row,0].step(at,requests[row][0],where='post',color=color,lw=1.15)
        ax[row,1].plot(tt,actual[row][0],color=color,lw=1.25)
        for col,records in [(0,requests),(1,actual)]:
            p=ax[row,col];p.set_title(f'({chr(97+row*2+col)})  {records[row][1]}',loc='left',pad=5)
            p.axvline(0,color='#8998A3',ls=':',lw=.75);p.grid(axis='y',color='#E4E9ED',lw=.45)
            p.set(xlim=(-.35,.35),xticks=[-.3,-.15,0,.15,.3]);p.margins(y=.15)
    ax[1,0].set_ylim(0,105);ax[1,1].set_ylim(0,105)
    ax[1,1].axhline(99,color='#8F999F',ls=':',lw=.6)
    ax[0,1].axhline(90,color='#8F999F',ls=':',lw=.6)
    for p in ax[-1]:p.set_xlabel('Time from corner passage (s)')
    fig.text(.265,.978,'Policy requests · 20 ms',ha='center',weight='bold',color=color,fontsize=9)
    fig.text(.775,.978,'Executed motion · 1 ms',ha='center',weight='bold',color=color,fontsize=9)
    fig.subplots_adjust(left=.075,right=.985,top=.89,bottom=.115,hspace=.67,wspace=.30)
    for ext in ['pdf','svg','png']:fig.savefig(FIG/f'Figure_B1.{ext}',dpi=230,bbox_inches='tight')
    plt.close(fig)
    np.savetxt(FIG/'Figure_B1_requests.csv',np.column_stack([at,ap]),delimiter=',',comments='',
        header='time_from_corner_passage_s,steering_correction_inward_deg,feedrate_request_mm_s,preview_distance_mm')
    np.savetxt(FIG/'Figure_B1_motion.csv',np.column_stack([tt,heading,speed,curvature]),delimiter=',',comments='',
        header='time_from_corner_passage_s,heading_inward_deg,speed_mm_s,curvature_inward_per_mm')
    write(OUT/'appendix_action_binding.json',dict(source_result_binding=str(OLD/'paper_result_binding.json'),
        source_result_binding_sha256=sha(OLD/'paper_result_binding.json'),trace=r['trace'],trace_sha256=r['trace_sha256'],
        metrics=r['metrics'],metrics_sha256=r['metrics_sha256'],geometry=case['geometry'],geometry_sha256=case['geometry_sha256'],
        reference=case['reference'],reference_sha256=case['reference_sha256'],model_sha256=b['selected_checkpoint_sha256'],
        model_seed=923202,corner='C02',crossing_time_s=tc,display_window_s=[-.35,.35],request_hold_s=.02,
        time_origin='First execution sample whose ordered reference progress reaches the C02 reference vertex.',
        projection_cache=str(cache_path),projection_cache_sha256=sha(cache_path),
        execution_interval_s=dt,requests='Recorded physical decoded actions; steering converted from radians to inward-positive degrees.',
        physical_quantities='Velocity and acceleration from XY finite differences with initial rest; heading relative to incoming tangent; curvature=cross(v,a)/speed^3 with inward sign.',
        smoothing=False,outputs={p.name:sha(p) for p in FIG.glob('Figure_B1.*')},
        csv_outputs={p.name:sha(p) for p in FIG.glob('Figure_B1_*.csv')},plot_script_sha256=sha(__file__)))
    for lang,name in [('en','manuscript'),('zh','manuscript_zh')]:
        dest=OUT/f'latex_source_{lang}'
        if not dest.exists():shutil.copytree(PREV/f'latex_source_{lang}',dest)
        old=(PREV/f'latex_source_{lang}'/(name+'.tex')).read_text(encoding='utf-8')
        new,block=source_text(old,lang)
        (dest/(name+'.tex')).write_text(new,encoding='utf-8')
        shutil.copy2(FIG/'Figure_B1.pdf',dest/'figures_lowdim_revision/Figure_B1.pdf')
        (OUT/f'changes_{lang}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
            fromfile='previous/'+name+'.tex',tofile='current/'+name+'.tex')),encoding='utf-8')
    print('Prepared compact appendix Figure B.1 and bilingual table-header-only edits.')


def build():
    import build_academic_revision as compiler
    compiler.OUT=OUT;compiler.WORK=WORK
    sys.argv=['build','final'];compiler.main()


if __name__=='__main__':
    if sys.argv[-1]=='build':build()
    else:prepare()
