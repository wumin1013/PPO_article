"""Prepare a clean ESWA release and a bounded evidence/code retention plan."""
import os,sys,re,json,shutil,hashlib,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'release_cleanup_20260924'
STAGE=WORK/'staged'
SYNC=Path('D:/Nutstore/1/基于强化学习的运动规划方法')
BASE=Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1')
OLD=BASE/'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'
CURRENT=OLD/'appendix_action_20260924'
ORIGINAL=SYNC/'论文项目/submission_package_EAAI'
PKG=STAGE/'新EWSA发布包'
PAPER=STAGE/'论文'
DATA=STAGE/'论文用到的数据和模型'
CODE=STAGE/'代码'
POP=Path('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
WORK.mkdir(exist_ok=True)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def mapped(p):
    for root,label in [(Path('D:/PPO_article_work'),'project_work'),(Path('D:/PPO_article_joint_runs'),'joint_runs')]:
        try:return Path(label)/p.relative_to(root)
        except ValueError:pass
    raise ValueError(str(p))

def seed_evidence():
    supplement=ROOT/'experimental_supplement_20260923'
    seeds=[OLD/'paper_result_binding.json',OLD/'nnc_evidence_binding.json',OLD/'citation_audit.json',
        CURRENT/'appendix_action_binding.json',CURRENT/'content_audit.json',CURRENT/'final_figure_binding.json',
        supplement/'selected_main_923202_evidence/selected_acceptance_20260924.json',
        supplement/'three_selected_student_acceptance_v2.json',
        supplement/'common_two_action_expansion/completed_endpoint_selection.json',
        supplement/'matched_direction_initialization/seed_923202/three/registration.json',
        supplement/'matched_direction_initialization/seed_923202/three/fit_log.json',
        supplement/'matched_direction_initialization/protocol.json',supplement/'matched_direction_initialization/supervision.npz']
    chosen={};queue=[]
    def add(p,depth):
        p=Path(p)
        if not p.is_absolute():p=ROOT/p
        if not p.is_file():return
        try:dest=mapped(p)
        except ValueError:return
        if any(part in ['plot_cache','__pycache__','status_history','delivery_history'] for part in p.parts):return
        if p.suffix.lower() not in {'.json','.npz','.pt','.pth','.csv','.npy','.yaml','.yml','.py','.md'}:return
        key=str(p)
        if key in chosen:return
        chosen[key]=dict(source=key,relative=str(dest),bytes=p.stat().st_size,depth=depth)
        if p.suffix=='.json' and depth<2 and p.stat().st_size<3000000:queue.append((p,depth))
    for p in seeds:add(p,0)
    acceptance=read(supplement/'three_selected_student_acceptance_v2.json')
    for rec in acceptance['records']:
        model=Path(rec['checkpoint']);model=model if model.is_absolute() else ROOT/model
        add(model,0)
        for name in ['registration.json','result.json','updates.json','fit_log.json','protocol.json','receipt.json']:
            add(model.parent/name,0)
        evaldir=Path(rec['evaluation'])
        for p in evaldir.glob('*.npz'):add(p,0)
        for p in evaldir.glob('*.json'):add(p,0)
    def paths(obj):
        if isinstance(obj,dict):
            for k,v in obj.items():
                yield k
                yield from paths(v)
        elif isinstance(obj,list):
            for v in obj:yield from paths(v)
        elif isinstance(obj,str):yield obj
    while queue:
        p,depth=queue.pop(0)
        for value in paths(read(p)):
            if re.match(r'^[Dd]:[/\\]PPO_',value) or (value.endswith(('.pt','.npz')) and '\n' not in value):
                add(value,depth+1)
    write(WORK/'evidence_retention_plan.json',dict(records=list(chosen.values()),bytes=sum(x['bytes'] for x in chosen.values()),
        scope='Files bound to final figures/results, three selected models, paired comparison and final-stage training. Original non-synced research tree is retained.'))
    print('EVIDENCE_PLAN',len(chosen),'files',round(sum(x['bytes'] for x in chosen.values())/1024**2,2),'MiB',flush=True)

def prepare():
    for p in [STAGE,PAPER,DATA,CODE,PKG]:p.mkdir(exist_ok=True)
    before={str(p.relative_to(ORIGINAL)):sha(p) for p in ORIGINAL.rglob('*') if p.is_file()}
    write(WORK/'original_EAAI_hashes.json',before)
    if not (STAGE/'EAAI发布包').exists():shutil.copytree(ORIGINAL,STAGE/'EAAI发布包')
    folders=['01_manuscript_without_author_details','02_title_page_with_author_details','03_cover_letter',
        '04_declaration_of_competing_interests','05_figures','06_latex_source_files','07_highlights']
    for f in folders:(PKG/f).mkdir(exist_ok=True)
    for f in ['02_title_page_with_author_details','04_declaration_of_competing_interests']:
        for p in (ORIGINAL/f).iterdir():
            if p.is_file():shutil.copy2(p,PKG/f/p.name)
    figures=PAPER/'图件';figures.mkdir(exist_ok=True)
    for p in (CURRENT/'figures').iterdir():
        if p.suffix in ['.pdf','.svg','.png','.csv','.json']:shutil.copy2(p,figures/p.name)
    conversions=[]
    for i in [1,2,3]:
        pdf=figures/f'Figure_{i}.pdf';svg=figures/f'Figure_{i}.svg'
        temp_pdf=WORK/f'vector_{i}.pdf';temp_svg=WORK/f'vector_{i}.svg'
        shutil.copy2(pdf,temp_pdf)
        result=subprocess.run(['D:/texlive/2025/bin/windows/pdftocairo.exe','-svg',str(temp_pdf),str(temp_svg)],capture_output=True)
        assert result.returncode==0,result.stderr.decode('utf-8',errors='replace')
        shutil.copy2(temp_svg,svg)
        conversions.append(dict(figure=i,pdf_sha256=sha(pdf),svg_sha256=sha(svg),
            scope='Complete vector SVG derived from current PDF, with outlined glyphs; original Illustrator layer history is unavailable.'))
    write(WORK/'complete_svg_conversion.json',conversions)
    for lang,name in [('en','manuscript'),('zh','manuscript_zh')]:
        dest=(PKG/'06_latex_source_files/latex_source_anonymous') if lang=='en' else (PAPER/'中文源码')
        if not dest.exists():shutil.copytree(CURRENT/f'latex_source_{lang}',dest)
        text=(CURRENT/f'latex_source_{lang}'/(name+'.tex')).read_text(encoding='utf-8')
        assert text.count(r'\journal{Engineering Applications of Artificial Intelligence}')==1
        text=text.replace(r'\journal{Engineering Applications of Artificial Intelligence}',r'\journal{Expert Systems with Applications}')
        (dest/(name+'.tex')).write_text(text,encoding='utf-8')
    for p in figures.glob('*.pdf'):shutil.copy2(p,PKG/'05_figures'/p.name)
    (PKG/'05_figures/SVG_editable').mkdir(exist_ok=True)
    for p in figures.glob('*.svg'):shutil.copy2(p,PKG/'05_figures/SVG_editable'/p.name)
    seed_evidence()
    # Preserve only code/configuration, not run directories or render caches.
    roots=[(Path('D:/PPO_article_work/PPO_project'),'PPO_project'),(BASE,'joint_policy'),(ROOT,'compact_dense_research'),(ROOT.parent,'natural_closed_resume')]
    copied=[]
    for src,label in roots:
        if label=='PPO_project':
            candidates=list(src.glob('*.py'))+list(src.glob('*.md'))+list(src.glob('requirements*.txt'))
            for name in ['src','configs','scripts','tools','tests']:
                candidates += [p for p in (src/name).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.yml','.json','.toml','.ps1','.md','.txt'] and '__pycache__' not in p.parts]
        else:candidates=list(src.glob('*.py'))
        for p in candidates:
            dest=CODE/label/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            copied.append(dict(source=str(p),destination=str(dest.relative_to(STAGE)),sha256=sha(p)))
    write(CODE/'代码来源索引.json',dict(records=copied,active_work_directory='D:/PPO_article_work',branch='codex/complex-path-adaptive-preview'))
    (CODE/'运行说明.md').write_text('''# 代码与运行环境

保留当前PPO代码、配置及论文对应研究/绘图脚本。实际研究目录仍为D:/PPO_article_work，执行前使用conda activate PPO。
本目录是用于同步保存的代码副本。部分历史研究脚本保留原D盘绝对路径；不要对其批量替换后启动新训练。
整理后的数据和模型位置见相邻“论文用到的数据和模型/来源与整理路径.json”，原D盘研究记录全部保留。
当前主模型为seed923202的fit_12000.pt，论文全部主图使用该模型；不会因目录整理重新训练。
''',encoding='utf-8')
    print('CODE_COPIED',len(copied),flush=True)

def copy_evidence():
    plan=read(WORK/'evidence_retention_plan.json');records=[]
    for r in plan['records']:
        p=Path(r['source']);dest=DATA/r['relative'];dest.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(p);shutil.copy2(p,dest);assert sha(dest)==digest
        records.append({**r,'sha256':digest})
    write(DATA/'来源与整理路径.json',dict(records=records,bytes=sum(r['bytes'] for r in records),
        original_research_retained=True,active_original_paths_unchanged=True))
    (DATA/'README.md').write_text('''# 论文所用数据、模型与训练依据

包含最终论文绑定的参考路径、1 ms/0.5 ms轨迹与结果，三个选定主模型、两动作与NNC对照依据，最终离线拟合监督数据、配置及配对实验依据。
“来源与整理路径.json”记录原始路径、当前相对路径和SHA-256。JSON内部原路径保留以便追溯；原D盘研究目录仍可直接使用。
开发全过程与旧实验留在非同步D盘；本目录不是完整历史镜像，也不改变论文对重复性或训练流程的表述。
''',encoding='utf-8')
    print('EVIDENCE_COPIED',len(records),flush=True)

if __name__=='__main__':
    copy_evidence() if sys.argv[-1]=='copy-evidence' else prepare()
