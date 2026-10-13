import sys,os,subprocess,shutil,re,zipfile,hashlib,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from prepare_eswa_release import ROOT,WORK,STAGE,PKG,PAPER,DATA,CODE,CURRENT,OLD,SYNC,read,write,sha,POP
from PIL import Image,ImageChops

def pdf_compare():
    results={}
    for lang in ['en','zh']:
        old=CURRENT/f'manuscript_revised_{lang}.pdf';vis=WORK/f'previous_visual_{lang}';vis.mkdir(exist_ok=True)
        subprocess.run([str(POP/'pdftoppm.exe'),'-r','85','-png',str(old),str(vis/'page')],check=True,capture_output=True)
        a=sorted(vis.glob('page-*.png'));b=sorted((WORK/f'eswa_visual_{lang}').glob('page-*.png'));assert len(a)==len(b)
        result=[]
        for i,(p,q) in enumerate(zip(a,b),1):
            box=ImageChops.difference(Image.open(p).convert('RGB'),Image.open(q).convert('RGB')).getbbox()
            result.append(dict(page=i,changed_bbox=box))
        results[lang]=result
        print(lang,'changed pages',[r for r in result if r['changed_bbox']],flush=True)
    write(WORK/'journal_only_pixel_comparison.json',results)

def inventory_sync():
    prefix='\\\\?\\'+str(SYNC)
    files=[];errors=[];links=[]
    def error(e):errors.append(str(e))
    for parent,dirs,names in os.walk(prefix,onerror=error,followlinks=False):
        for n in dirs:
            p=os.path.join(parent,n)
            if os.path.islink(p):links.append(p)
        for n in names:
            p=os.path.join(parent,n)
            try:
                relative=os.path.relpath(p,prefix)
                files.append(dict(relative=relative,bytes=os.stat(p).st_size,sha256=sha(p)))
            except Exception as e:errors.append(str(e))
    write(WORK/'sync_full_inventory_hashed.json',dict(files=files,errors=errors,links=links,bytes=sum(r['bytes'] for r in files)))
    assert not errors,errors
    assert not links,links
    print('SYNC INVENTORY',len(files),round(sum(r['bytes'] for r in files)/1024**2,1),'MiB',flush=True)

def package():
    # Preserve original EAAI package byte-for-byte.
    original=read(WORK/'original_EAAI_hashes.json')
    assert {str(p.relative_to(STAGE/'EAAI发布包')):sha(p) for p in (STAGE/'EAAI发布包').rglob('*') if p.is_file()}==original
    src=PKG/'06_latex_source_files/latex_source_anonymous'
    tex=(src/'manuscript.tex').read_text('utf-8')
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex)
    assert len(figures)==11
    for f in figures:
        assert sha(src/'figures_lowdim_revision'/f)==sha(PKG/'05_figures'/f)==sha(PAPER/'图件'/f)
    with zipfile.ZipFile(PKG/'06_latex_source_files/latex_source_anonymous.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(src.rglob('*')):
            if p.is_file():z.write(p,Path('latex_source_anonymous')/p.relative_to(src))
    # Decompress the delivered source archive and verify all bytes against the independently compiled ASCII source.
    unz=WORK/'source_zip_validation';unz.mkdir(exist_ok=True)
    with zipfile.ZipFile(PKG/'06_latex_source_files/latex_source_anonymous.zip') as z:
        assert z.testzip() is None;z.extractall(unz)
    for p in src.rglob('*'):
        if p.is_file():assert sha(p)==sha(unz/'latex_source_anonymous'/p.relative_to(src))==sha(WORK/'src_en'/p.relative_to(src))
    mapping=['Figure_1: architecture comparison','Figure_2: reference path and tolerance corridor','Figure_3: closed-loop framework',
        'Figure_4: long-route geometry and angle coverage','Figure_5: NNC constraint violations','Figure_6: aligned local V/A/J and speed deficit',
        'Figure_7: representative local trajectories','Figure_8: all-corner loss and complete-route cruising','Figure_9: dense case and two-action comparison',
        'Figure_B1: compact action-motion correspondence','Figure_C1: complete V/A/J and terminal braking']
    (PKG/'05_figures/figure_mapping.txt').write_text('\n'.join(mapping)+'\n',encoding='utf-8')
    (PKG/'README_投稿文件说明.md').write_text('''# ESWA投稿文件包

目标期刊：Expert Systems with Applications（ESWA）。文件夹按作者指定名称“新EWSA发布包”保存。

1. 01：23页匿名英文稿PDF。
2. 02：原作者页DOCX，作者顺序、单位、基金与贡献保持原文。
3. 03：ESWA投稿信DOCX及PDF。
4. 04：原利益冲突声明DOCX。
5. 05：9幅正文图、附图B.1/C.1的矢量PDF与SVG。
6. 06：匿名LaTeX源码目录及ZIP，含参考文献、样式和全部引用图件。
7. 07：5条Highlights，DOCX/TXT内容一致，每条不超过85字符；附PDF预览。

使用XeLaTeX与BibTeX编译，推荐TeX Live 2025。字体依赖SimSun；将源码放入较短路径可避免Windows路径限制。已从独立工作副本成功编译，源码ZIP已解压核对。

中文21页稿和给老师的新旧稿改进清单位于相邻“论文”目录，不属于匿名投稿附件。
图1—3的完整SVG由当前矢量PDF导出，文字为轮廓，不保留原Illustrator图层和文字编辑历史。图4—9及附图来自真实数据绘图。

提交前作者事项：原利益冲突声明仍有通讯作者确认提示；请确认后移除该提示。投稿信沿用原稿“未同时投往他刊”的声明，实际提交时须确保符合当前投稿状态。本文档是文件整理说明，不代表已向期刊提交，也不代表实时核验了投稿网站最新字段要求。
''',encoding='utf-8')
    (PKG/'submission_checklist.md').write_text('''# 文件核验清单

- [x] 匿名英文稿23页，PDF作者字段为空，正文无作者姓名与邮箱。
- [x] 正文保留9幅图和3张结果表，含2幅附图。
- [x] 引用沿用已核验的24项；新包只更改期刊名称，正文数值未改。
- [x] PDF、单独图件和LaTeX引用图件SHA-256一致。
- [x] 中英文分别编译成功，无未定义引用、缺字或溢出版面警告。
- [x] 投稿信为Expert Systems with Applications；Highlights为5条，每条≤85字符。
- [x] 作者页和利益声明保持原作者文件。
- [ ] 通讯作者确认利益冲突声明并移除文件中的TODO提示。
- [ ] 作者在实际提交时确认投稿状态及期刊系统字段。
''',encoding='utf-8')
    (PAPER/'README.md').write_text('''# 当前论文文件

英文稿_ESWA.pdf：23页投稿正文，基于唯一18页原稿修订。
中文稿.pdf：21页中文对照阅读稿。
给辅导老师/新旧稿改进清单.pdf：4页对比清单，并有可编辑DOCX与Markdown。
图件：9幅主图与附图B.1/C.1、矢量源及绑定数据；图1—3SVG为PDF转矢量轮廓版本。
中文源码：中文LaTeX源码。英文源码在相邻“新EWSA发布包/06_latex_source_files”。

旧18页稿仍在相邻“EAAI发布包/01_manuscript_without_author_details”，原EAAI包内容未修改。
训练与逐轮过程保留非同步D盘，论文正文不放过程日志。
''',encoding='utf-8')
    # Keep provenance and editing guidance with paper material, outside submission source.
    qa=PAPER/'核验与修改说明';qa.mkdir(exist_ok=True)
    for p in [CURRENT/'delivery_manifest.json',CURRENT/'content_audit.json',CURRENT/'final_figure_binding.json',
              WORK/'complete_svg_conversion.json',WORK/'journal_only_pixel_comparison.json',WORK/'eswa_build_inventory.json']:
        if p.is_file():shutil.copy2(p,qa/p.name)
    for p in CURRENT.glob('*.md'):shutil.copy2(p,qa/p.name)
    (qa/'本次发布说明.md').write_text('原delivery_manifest及图件绑定保留作来源记录；其旧路径指向未改动的D盘历史交付。当前ESWA发布位置与校验值见新包release_manifest.json。原图1/3修订已经纳入当前PDF；本次新增完整SVG不保留AI图层。\n',encoding='utf-8')
    records=[dict(relative=str(p.relative_to(PKG)),bytes=p.stat().st_size,sha256=sha(p)) for p in PKG.rglob('*') if p.is_file() and p.name!='release_manifest.json']
    write(PKG/'release_manifest.json',dict(journal='Expert Systems with Applications',paper_pages=23,chinese_pages=21,main_figures=9,appendix_figures=2,
        base_manuscript_pages=18,selected_seed=923202,checkpoint_sha256=read(OLD/'paper_result_binding.json')['selected_checkpoint_sha256'],
        previous_delivery=str(CURRENT/'delivery_manifest.json'),body_change_this_release='Journal name only.',files=records))
    print('PACKAGE VERIFIED',len(records),'files',flush=True)

if __name__=='__main__':{'compare':pdf_compare,'inventory':inventory_sync,'package':package}[sys.argv[-1]]()
