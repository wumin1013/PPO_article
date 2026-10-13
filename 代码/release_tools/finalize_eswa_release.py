"""Assemble the frozen ESWA delivery; no model execution or training."""
import os,sys,re,shutil,subprocess,json,zipfile,difflib
from pathlib import Path
from prepare_eswa_release import ROOT,WORK,STAGE,PKG,PAPER,DATA,CODE,CURRENT,OLD,read,write,sha,mapped,POP

def paths(x):
    if isinstance(x,dict):
        for k,v in x.items():
            yield k
            yield from paths(v)
    elif isinstance(x,list):
        for v in x:yield from paths(v)
    elif isinstance(x,str):yield x

def resolved(s):
    p=Path(s)
    return p if p.is_absolute() else ROOT/p

def evidence():
    full=WORK/'evidence_retention_plan_full_history.json'
    plan=read(full if full.exists() else WORK/'evidence_retention_plan.json')
    allowed=set()
    for p in [OLD/'paper_result_binding.json',OLD/'nnc_evidence_binding.json',CURRENT/'appendix_action_binding.json']:
        allowed.update(str(resolved(s)) for s in paths(read(p)) if '\n' not in s and len(s)<500)
    acc=read(ROOT/'experimental_supplement_20260923/three_selected_student_acceptance_v2.json')
    endpoints=read(ROOT/'experimental_supplement_20260923/common_two_action_expansion/completed_endpoint_selection.json')
    # Keep full traces for final selected students, selected paired endpoints and the fastest valid dense comparator.
    for r in acc['records']+endpoints['selected']+[endpoints['endpoint_records'][0]]:
        allowed.add(str(resolved(r['checkpoint'])))
        d=Path(r.get('evaluation',r.get('directory')))
        allowed.update(str(p) for p in d.glob('*.npz'))
    selected=[];excluded=[]
    for r in plan['records']:
        p=Path(r['source'])
        if p.suffix=='.npz' and str(p) not in allowed and 'eval_' in p.name:
            excluded.append(r);continue
        if p.suffix=='.npz' and p.name=='trace.npz' and str(p) not in allowed:
            excluded.append(r);continue
        category={'.npz':'数据','.pt':'模型','.pth':'模型','.csv':'数据','.npy':'数据'}.get(p.suffix,'核验与配置')
        r={**r,'original_relative':r['relative'],'relative':str(Path(category)/(sha(p)[:12]+'_'+p.name))}
        selected.append(r)
    # All small receipts, endpoint/failure summaries and checkpoints stay in the evidence copy.
    write(WORK/'evidence_retention_plan_full_history.json',plan)
    write(WORK/'evidence_retention_plan.json',dict(records=selected,bytes=sum(r['bytes'] for r in selected),
        scope='Final figure/result evidence, selected student models, paired endpoints, checkpoints and learning data; candidate traces remain in the unchanged D-drive research archive.'))
    write(WORK/'candidate_traces_retained_on_D.json',dict(records=excluded,deleted=False))
    import prepare_eswa_release as p
    p.copy_evidence()
    print('SELECTED',len(selected),'EXCLUDED_FROM_SYNC',len(excluded),'MiB',round(sum(r['bytes'] for r in selected)/1024**2,1),flush=True)

def compile_papers():
    os.environ['PYTHONUTF8']='1'
    compile_script=Path('C:/Users/wumin/.codex/plugins/cache/openai-bundled/latex/0.2.7/scripts/compile_latex.py')
    from PIL import Image,ImageDraw
    inventory={}
    for lang,name,source in [('en','manuscript',PKG/'06_latex_source_files/latex_source_anonymous'),('zh','manuscript_zh',PAPER/'中文源码')]:
        old=(CURRENT/f'latex_source_{lang}'/(name+'.tex')).read_text('utf-8')
        new=(source/(name+'.tex')).read_text('utf-8')
        assert new==old.replace(r'\journal{Engineering Applications of Artificial Intelligence}',r'\journal{Expert Systems with Applications}')
        (WORK/f'journal_change_{lang}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))),encoding='utf-8')
        # Compile from a short ASCII working copy to avoid TeX/Windows codepage and MAX_PATH failures.
        local=WORK/f'src_{lang}'
        shutil.copytree(source,local,dirs_exist_ok=True)
        source=local
        bld=WORK/f'eswa_build_{lang}';bld.mkdir(exist_ok=True)
        cmd=[sys.executable,str(compile_script),str(source/(name+'.tex')),'--compiler','texlive','--engine','xelatex','--output-directory',str(bld),'--json']
        result=subprocess.run(cmd,capture_output=True)
        (WORK/f'eswa_compile_{lang}.txt').write_bytes(result.stdout+result.stderr)
        assert result.returncode==0,(result.stdout+result.stderr)[-4000:]
        pdf=bld/(name+'.pdf')
        dest=PAPER/('英文稿_ESWA.pdf' if lang=='en' else '中文稿.pdf');shutil.copy2(pdf,dest)
        if lang=='en':shutil.copy2(pdf,PKG/'01_manuscript_without_author_details/manuscript_without_author_details.pdf')
        vis=WORK/f'eswa_visual_{lang}';vis.mkdir(exist_ok=True)
        subprocess.run([str(POP/'pdftoppm.exe'),'-r','85','-png',str(pdf),str(vis/'page')],capture_output=True,check=True)
        info=subprocess.check_output([str(POP/'pdfinfo.exe'),str(pdf)]).decode('utf-8',errors='replace')
        page_count=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        subprocess.run(['D:/texlive/2025/bin/windows/pdftotext.exe','-layout',str(pdf),str(vis/'fulltext.txt')],check=True,capture_output=True)
        text=(vis/'fulltext.txt').read_text('utf-8')
        assert page_count==(23 if lang=='en' else 21)
        if lang=='en':
            assert not any(s in text for s in ['huzifang','Huicheng Zhou','Min Wu','Zifang Hu'])
            assert not re.search(r'^Author:\s*\S',info,re.M)
        log=pdf.with_suffix('.log').read_text('utf-8',errors='replace')
        issues=re.findall(r'^.*(?:Overfull|undefined references|undefined citations|Missing character:).*$',log,re.M|re.I)
        assert not issues,issues
        pages=sorted(vis.glob('page-*.png'))
        contacts=[]
        for start in range(0,len(pages),6):
            canvas=Image.new('RGB',(1530,2200),'#eeeeee');draw=ImageDraw.Draw(canvas)
            for j,p in enumerate(pages[start:start+6]):
                im=Image.open(p).convert('RGB');im.thumbnail((495,690));x=j%3*510+(510-im.width)//2;y=j//3*1100+28
                canvas.paste(im,(x,y));draw.text((j%3*510+12,j//3*1100+8),f'{lang} {start+j+1}',fill='black')
            dest=vis/f'contact-{start//6+1}.png';canvas.save(dest);contacts.append(str(dest))
        inventory[lang]=dict(pdf=str(pdf),sha256=sha(pdf),pages=page_count,compile_issues=issues,contacts=contacts)
        print('COMPILED',lang,page_count,flush=True)
    write(WORK/'eswa_build_inventory.json',inventory)

if __name__=='__main__':
    {'evidence':evidence,'compile':compile_papers}[sys.argv[-1]]()
