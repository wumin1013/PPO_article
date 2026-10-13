"""Render every PDF page and retain text/bounding-box checks for visual QA."""
import re, subprocess
from xml.etree import ElementTree as ET
from PIL import Image, ImageOps, ImageDraw
from study_io import BASE, sha, write_json

paper=BASE/'manuscript_joint_v1'
build=paper/'build_natural'
pdf=build/'PPO_article.pdf'
qa=paper/'natural_pdf_qa'/sha(pdf)[:12]
qa.mkdir(exist_ok=True,parents=True)
relative_qa=str(qa.relative_to(paper)).replace('\\','/')
for args,output in ((['pdfinfo','build_natural/PPO_article.pdf'],'pdfinfo.txt'),(['pdffonts','build_natural/PPO_article.pdf'],'pdffonts.txt')):
    r=subprocess.run(args,cwd=paper,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        text=r.stdout.decode('utf-8')
    except UnicodeDecodeError:
        text=r.stdout.decode('gb18030')
    (qa/output).write_text(text,encoding='utf-8')
subprocess.run(['pdftotext','-layout','build_natural/PPO_article.pdf',relative_qa+'/manuscript_text.txt'],cwd=paper,check=True)
subprocess.run(['pdftotext','-bbox','build_natural/PPO_article.pdf',relative_qa+'/manuscript_bbox.html'],cwd=paper,check=True)
subprocess.run(['pdftoppm','-r','110','-png','build_natural/PPO_article.pdf',relative_qa+'/page'],cwd=paper,check=True)
xml=(qa/'manuscript_bbox.html').read_text(encoding='utf-8')
invalid_xml_controls=re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',xml)
tree=ET.ElementTree(ET.fromstring(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',' ',xml)))
pages=tree.findall('.//{*}page')
records=[]
for i,page in enumerate(pages,1):
    words=page.findall('.//{*}word')
    body=[w for w in words if float(w.attrib['yMin'])>63 and float(w.attrib['yMax'])<785]
    bad=[w.text for w in words if float(w.attrib['xMin'])<20 or float(w.attrib['xMax'])>float(page.attrib['width'])-20 or float(w.attrib['yMin'])<10 or float(w.attrib['yMax'])>float(page.attrib['height'])-10]
    records.append(dict(page=i,word_count=len(body),first_words=' '.join(w.text or '' for w in body[:18]),
        last_words=' '.join(w.text or '' for w in body[-18:]),outside_page_safe_region=bad))
    print(i,len(body),records[-1]['first_words'],'...',records[-1]['last_words'])
for start in range(1,len(pages)+1,2):
    ids=list(range(start,min(start+2,len(pages)+1)))
    pictures=[]
    for i in ids:
        src=qa/f'page-{i:02d}.png'
        im=Image.open(src).convert('RGB')
        im.thumbnail((820,1160))
        im=ImageOps.expand(im,border=(10,30,10,10),fill='white')
        ImageDraw.Draw(im).text((12,8),f'Page {i}',fill='black')
        pictures.append(im)
    canvas=Image.new('RGB',(sum(x.width for x in pictures),max(x.height for x in pictures)), '#dddddd')
    x=0
    for im in pictures:canvas.paste(im,(x,0));x+=im.width
    canvas.save(qa/f'contact_{start:02d}_{ids[-1]:02d}.png')
log=(build/'PPO_article.log').read_bytes().decode('utf-8',errors='replace')
issues=[line for line in log.splitlines() if re.search('Warning|Overfull|Underfull|undefined|Missing character',line)]
record=dict(pdf_sha256=sha(pdf),page_count=len(pages),page_count_requirement_pass=20<=len(pages)<=25,
    poppler_xml_control_glyph_count=len(invalid_xml_controls),
    rendered_pages=list(range(1,len(pages)+1)),page_records=records,latex_issues=issues,
    visual_review_complete=False)
write_json(qa/'structural_audit.json',record)
write_json(paper/'natural_pdf_qa/latest.json',dict(directory=str(qa.relative_to(BASE)),pdf_sha256=sha(pdf),page_count=len(pages)))
print('PAGES',len(pages),'LATEX_ISSUES',issues)
print('QA_DIRECTORY',qa)
