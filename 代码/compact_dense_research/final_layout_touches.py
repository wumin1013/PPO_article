from pathlib import Path
import json, hashlib, shutil
from datetime import datetime, timezone

ROOT=Path(r'D:\PPO_article_joint_runs\natural_closed_resume\compact_dense_research')
WORK=ROOT/'lowdim_original_manuscript_work'
NEW=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1\paper_evidence_review\verified_results\lowdim_original_revision\selected923202_revision')
EN=Path(r'D:\PPO_article_work\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous\manuscript.tex')
ZH=NEW/'latex_source_zh/manuscript_zh.tex'
history=WORK/'editorial_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
history.mkdir(parents=True)
for path in [EN,ZH,NEW/'修改说明.tex',NEW/'图件修改建议与实验外修改说明.md',NEW/'translation_alignment.json']:
    shutil.copy2(path,history/path.name)
for path in [EN,ZH]:
    text=path.read_text(encoding='utf-8')
    text=text.replace(r'\bibliographystyle{unsrt}',r'\setlength{\bibsep}{3pt plus 0.3pt}'+'\n'+r'\bibliographystyle{unsrt}')
    if path==ZH:
        text=text.replace(r'\renewcommand{\abstractname}{摘要}',r'\abstracttitle{摘要}'+'\n'+r'\keywordtitle{关键词}')
    path.write_text(text,encoding='utf-8')
path=NEW/'修改说明.tex'
text=path.read_text(encoding='utf-8').replace('array,amsmath,hyperref','array,amsmath,amssymb,hyperref')
text=text.replace('原位置 & 建议替换或调整 & 原因\\\\\n\\midrule','原位置 & 建议替换或调整 & 原因\\\\\n\\midrule\\endhead')
text=text.replace('原位置 & 建议文字\\\\\n\\midrule','原位置 & 建议文字\\\\\n\\midrule\\endhead')
text=text.replace('s $\\in$ R$^{14}$',r'$s\in\mathbb{R}^{14}$')
text=text.replace('$\\Vert$v$\\Vert$$_2$ $\\leq$ Vmax；$\\Vert$a$\\Vert$$_2$ $\\leq$ Amax；$\\Vert$j$\\Vert$$_2$ $\\leq$ Jmax',r'$\|v\|_2\leq V_{\max}$；$\|a\|_2\leq A_{\max}$；$\|j\|_2\leq J_{\max}$')
text=text.replace('Safe Command中的$\\omega$t、vt & Planar Velocity vk+1',r'Safe Command中的$\omega_t$、$v_t$ & Planar Velocity $v_{k+1}$')
text=text.replace('u$\\theta$、uv、uL',r'$u_\theta$、$u_v$、$u_L$')
text=text.replace('正文五章和实验三个小节均保留。','正文五章和实验三个小节均保留。参考文献仅压缩条目间距，以避免末条独占一页。')
path.write_text(text,encoding='utf-8')
path=NEW/'图件修改建议与实验外修改说明.md'
text=path.read_text(encoding='utf-8').replace('正文五章和实验三个小节均保留。','正文五章和实验三个小节均保留。参考文献仅压缩条目间距，以避免末条独占一页。')
path.write_text(text,encoding='utf-8')
path=NEW/'translation_alignment.json'
data=json.loads(path.read_text(encoding='utf-8'))
data['source_sha256']=hashlib.sha256(EN.read_bytes()).hexdigest()
data['translated_sha256']=hashlib.sha256(ZH.read_bytes()).hexdigest()
path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Layout changes saved; prior sources at',history)
