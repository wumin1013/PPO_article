"""Figure8: developed-controller ablation and full dense-case loss accumulation."""
from pathlib import Path
src=Path(__file__).with_name('draw_completed_paired_figure8.py')
code=src.read_text(encoding='utf-8')
a=code.index("code=code[:start]+'''")
b=code.index("''' +code[end:]",a)+len("''' +code[end:]")
replacement="""code=code[:start]+'''    curve_data={}
    for name,key in [('Two-action','Fixed'),('J-NNC','Joint')]:
        item=data[name]
        speed=np.linalg.norm(item['rows'][:,2:4],axis=1)
        deficit=np.cumsum(np.maximum(100.-speed,0.))*.001
        ax.plot(item['progress'],deficit,color=common.COLORS[key],ls=common.STYLES[key],lw=1.3,label=name)
        ax.text(1448,deficit[-1],f'{deficit[-1]:.1f}',color=common.COLORS[key],fontsize=7,va='center')
        curve_data[name+'_progress_mm']=item['progress']
        curve_data[name+'_deficit_mm']=deficit
    np.savez_compressed(output/'Figure_8_dense_deficit.npz',**curve_data)
    ax.set(xlim=(0,1660),ylim=(0,240),xticks=[0,400,800,1200,1600],xlabel='Reference progress (mm)',ylabel='Cumulative speed deficit (mm)')
    ax.grid(color='#E7EAED',lw=.4)
    ax.legend(fontsize=7,loc='upper left')
    common.panel(ax,'b','Speed loss over all 84 corners')
''' +code[end:]"""
code=code[:a]+replacement+code[b:]
code=code.replace("out=delivery/'paired_ablation'","out=delivery/'dense_ablation'")
code=code.replace("conclusion='Developed controllers differ in local trajectory;equal-opportunity PPO intervention has small mixed gains.'", "conclusion='Three-action developed controller reduces complete dense speed deficit while using different local transitions.'")
code=code.replace("panels='a densecase,b matched refinement,c/d developed-controller local shapes;roles explicitly separate'", "panels='a densecase,b complete loss over84corners,c/d local shapes;matched PPO intervention reported separately intext'")
code=code.replace("paired_panel='Three common-start pairs;equal48PPOupdates;fixed0/24/48selection;not the separately developed final-controller comparison.'", "dense_loss_panel='Complete nominal dense execution includingbrake/rest;all84corners;no smoothing'")
exec(compile(code,str(src),'exec'))
