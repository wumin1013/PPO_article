"""Edit the author's original vector PDFs in place on preserved work copies."""
import sys, json, hashlib, shutil
from pathlib import Path
sys.path.append('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import pypdfium2 as pdfium
import pypdfium2.raw as raw
from pypdf import PdfReader, PdfWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT=Path(__file__).resolve().parent
OLD=Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1/paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision')
PREV=OLD/'figure_revision_20260924'
OUT=OLD/'academic_figures_20260924'
FIG=OUT/'figures'
WORK=ROOT/'lowdim_original_manuscript_work/academic_figure_revision'
for p in [OUT,FIG,WORK]:p.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
BLUE='#205E89';GRAY='#48525A';PALE='#EAF2F7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rgba(hex):return tuple(int(hex[i:i+2],16) for i in (1,3,5))+(255,)
def recolor(o,color,fill=True,stroke=True):
    if fill:raw.FPDFPageObj_SetFillColor(o,*rgba(color))
    if stroke:raw.FPDFPageObj_SetStrokeColor(o,*rgba(color))
def canvas(size):
    fig=plt.figure(figsize=(size[0]/72,size[1]/72));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,size[0]),ylim=(0,size[1]));ax.axis('off')
    return fig,ax
def txt(ax,x,y,s,fs=9,**kw):
    return ax.text(x,y,s,fontsize=fs,ha=kw.pop('ha','center'),va='center',color=kw.pop('color',GRAY),**kw)
def node(ax,x,y,s,r=8.5,fs=7.5,fc=PALE):
    ax.add_patch(Circle((x,y),r,facecolor=fc,edgecolor=BLUE,lw=.65,zorder=3));txt(ax,x,y,s,fs,zorder=4)
def finish(n,doc,page,fig,removed):
    page.gen_content();clean=WORK/f'Figure_{n}_retained_vectors.pdf';doc.save(str(clean))
    overlay=WORK/f'Figure_{n}_new_labels.pdf'
    fig.savefig(overlay,transparent=True)
    fig.savefig(FIG/f'Figure_{n}_editable_labels.svg',transparent=True)
    plt.close(fig)
    reader=PdfReader(str(clean));reader.pages[0].merge_page(PdfReader(str(overlay)).pages[0])
    writer=PdfWriter();writer.add_page(reader.pages[0])
    with (FIG/f'Figure_{n}.pdf').open('wb') as f:writer.write(f)
    check=pdfium.PdfDocument(str(FIG/f'Figure_{n}.pdf'));pg=check[0]
    pg.render(scale=2).to_pil().save(FIG/f'Figure_{n}.png')
    text=pg.get_textpage().get_text_range()
    (WORK/f'Figure_{n}_revised_text.txt').write_text(text,encoding='utf-8')
    return dict(original=str(PREV/f'figures/Figure_{n}.pdf'),original_sha256=sha(PREV/f'figures/Figure_{n}.pdf'),
        output_sha256=sha(FIG/f'Figure_{n}.pdf'),removed_top_level_object_indices=removed,
        original_page_size_preserved=True,original_shapes_retained=True,
        labels_svg='Editable replacement labels and local diagrams only; original Illustrator source was not available.')
def one():
    doc=pdfium.PdfDocument(str(PREV/'figures/Figure_1.pdf'));p=doc[0]
    allobjs=list(p.get_objects());rm=set(range(49,57))|set(range(66,74))|set(range(83,107))|set(range(163,181))|{198,200,201,202,203,205,207,208,240}
    removed=[]
    for i,o in enumerate(allobjs):
        if o.level:continue
        if i in rm:p.remove_obj(o);removed.append(i)
        elif i in range(181,195):recolor(o,BLUE)
        elif i in [40,42,44,46]:recolor(o,'#E2F1F0',stroke=False)
        elif i in [57,59,61,63,74,76,78,80]:recolor(o,'#C9DCEB',stroke=False)
    f,ax=canvas(p.get_size())
    txt(ax,3,328,'(a) Two-step planning',11,ha='left')
    txt(ax,244,328,'(b) J-NNC joint motion planning',11,ha='left',color=BLUE,weight='bold')
    txt(ax,244,242,'b.2 Learned policy: static features + network',8,ha='left')
    for x,s in zip([337.8,372.1,405.6,456.4],[r'$s_1$',r'$s_2$',r'$s_3$',r'$s_{14}$']):txt(ax,x,221,s,8)
    for y,k in [(189.8,1),(157.7,2)]:
        for x,j in zip([337.8,372.1,405.6,456.4],['1','2','3','n']):txt(ax,x,y,rf'$h^{{({k})}}_{{{j}}}$',6.5)
    for x in [337.8,372.1,405.6,456.4]:
        for xx in [350,397,444]:ax.plot([x,xx],[148.1,135.1],lw=.35,color='#A7BAC8',zorder=1)
    for x,s in zip([350,397,444],[r'$u_\theta$',r'$u_v$',r'$u_L$']):node(ax,x,126.7,s,8.8,8,fc='#F5E5CC')
    for y,s in [(220,'14 inputs'),(189,'Hidden layer'),(157,'Hidden layer'),(126,'3 actions')]:txt(ax,291,y,s,7.8)
    txt(ax,3,100,'a.4 Interpolation',9,ha='left');txt(ax,98.5,54.8,'Interpolation points',5.4)
    txt(ax,244,101,'b.3 Kinematic constraint module',8.4,ha='left')
    for y in [45.4,73.7]:ax.add_patch(Rectangle((297.6,y),170.1,14.2,fc=PALE,ec='none'))
    txt(ax,382.6,80.6,'1. Planar V/A/J projection',7.7)
    txt(ax,382.6,52.5,'2. Feasible acceleration release',7.7)
    txt(ax,244,29.4,'b.4 Feasible motion command',8.4,ha='left')
    return finish(1,doc,p,f,removed)
def three():
    doc=pdfium.PdfDocument(str(PREV/'figures/Figure_3.pdf'));p=doc[0]
    objs=list(p.get_objects());removed=[]
    # Retain original boxes, inter-module execution arrows and reward-collection link.
    for i,o in enumerate(objs):
        if o.level:continue
        remove=(o.type in [1,5] or 80<=i<=235 or i in [37,38,41,42])
        if remove:p.remove_obj(o);removed.append(i)
    f,ax=canvas(p.get_size())
    txt(ax,84.5,518,'G-code input',14,weight='bold')
    txt(ax,84.5,482,r'Reference path $P_{\rm raw}$',11.8)
    txt(ax,84.5,459,r'Full band width $\varepsilon$',11.5)
    txt(ax,263,514,'Geometry encoding',14,weight='bold')
    for y,s in [(468,'Ordered projection'),(451,'Fixed forward geometry'),(430,'2, 6, 15 and 30 mm')]:txt(ax,263,y,s,12.5)
    txt(ax,447,519,r'Current state $s$',14,weight='bold')
    txt(ax,447,493,r'$s\in\mathbb{R}^{14}$',15)
    for y,s in [(468,'Tracking error'),(454,'Planar kinematics'),(440,'Forward geometry')]:txt(ax,447,y,s,12.5)
    txt(ax,631.5,537,'Learned three-action policy',11.5,weight='bold',color=BLUE)
    xx=[586,617,648,677];ys=[[510,486,462],[510,486,462],[510,486,462],[508,486,464]]
    for i in range(3):
        for y in ys[i]:
            for yy in ys[i+1]:ax.plot([xx[i],xx[i+1]],[y,yy],color='#95A4AF',lw=.55,zorder=1)
    for i,x in enumerate(xx):
        for j,y in enumerate(ys[i]):node(ax,x,y,(['', '', ''] if i<3 else [r'$u_\theta$',r'$u_v$',r'$u_L$'])[j],5.5,5.8,fc='#E4EDF4' if i==0 else '#FBF4E8')
    txt(ax,631.5,430,r'$u=f_{\theta}(s)$',14)
    txt(ax,816,512,'Action request',14,weight='bold')
    txt(ax,816,477,r'$u=[u_\theta,u_v,u_L]$',14)
    txt(ax,816,458,r'Decode $(\alpha,F,L)$',13)
    txt(ax,816,442,'Held for 20 ms',12.5)
    txt(ax,744,312,'KCM: planar kinematic constraints',14,weight='bold')
    txt(ax,624,268,'V/A/J\nprojection',12.5)
    txt(ax,739,268,'Acceleration\nrelease',12.5)
    txt(ax,860,268,'Next velocity\nselection',12.5)
    txt(ax,744,229,r'$\|v\|\leq V_{\max},\quad\|a\|\leq A_{\max},\quad\|j\|\leq J_{\max}$',15)
    txt(ax,447,284,'Feasible command',12.1,weight='bold')
    txt(ax,447,253,r'$v_{k+1}\in\mathbb{R}^{2}$',14)
    txt(ax,263,314,'Interpolation\nexecution and\nstate update',12.8,weight='bold',linespacing=1.15)
    for y,s in [(269,'Position'),(252,'Velocity / acceleration'),(235,'Ordered progress'),(216,'Contour distance')]:txt(ax,263,y,s,11.4)
    txt(ax,93,166,r'Reward $r$',14)
    for y,s in [(132,'Elapsed time'),(113,'Progress potential'),(94,'Failure penalty')]:txt(ax,93,y,s,12.5)
    # Route the parameter-update connector outside execution blocks, never through KCM.
    ax.plot([22,10,10,632],[121,121,563,563],color=BLUE,lw=1.5,ls=(0,(7,5)))
    ax.annotate('',xy=(632,552),xytext=(632,563),arrowprops=dict(arrowstyle='-|>',lw=1.5,color=BLUE))
    txt(ax,420,96,'PPO learning + offline refinement: policy parameters only',13,color=BLUE,weight='bold')
    txt(ax,201,37,'Online execution',11.5)
    txt(ax,493,37,'Training feedback / parameter update',11.5)
    return finish(3,doc,p,f,removed)
def main():
    report={1:one(),3:three()}
    shutil.copy2(PREV/'figures/Figure_2.pdf',FIG/'Figure_2.pdf')
    report[2]=dict(unchanged=True,sha256=sha(FIG/'Figure_2.pdf'))
    write(OUT/'original_vector_edit_audit.json',dict(figures=report,original_files_untouched=True,
        editing_method='Remove obsolete top-level PDF objects; retain original vector structure; merge vector text/local-node overlay. No raster covering or substitute historical diagram.',
        authorization='Latest user instruction permits reliable modification of existing exported PDFs.'))
    print('EDITED_ORIGINAL_VECTORS',str(FIG))
if __name__=='__main__':main()
