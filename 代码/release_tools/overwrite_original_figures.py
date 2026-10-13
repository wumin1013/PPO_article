"""Apply the user's explicit original-figure overwrite request, preserving a hash-bound backup."""
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from prepare_eswa_release import WORK,PAPER,SYNC,read,write,sha

backup=WORK/'original_master_backup';backup.mkdir(exist_ok=True)
records=[]
names={1:'图1 传统串行式运动规划与J-NNC端到端规划的架构对比.pdf',2:'图2参考路径与轮廓允差带构成的几何可行域.pdf',3:'图3 总体流程图.pdf'}
for i in [1,2,3]:
    targets=[Path('D:/PPO_article_work/论文项目/figures')/f'fig{i}.pdf',SYNC/'论文项目/figures'/f'fig{i}.pdf',
        Path('D:/PPO_article_work/论文项目/强化学习论文绘图')/f'图{i}'/names[i]]
    for j,target in enumerate(targets):
        assert target.is_file(),target
        saved=backup/f'figure_{i}_source_{j}.pdf'
        assert not saved.exists(),'Do not overwrite a historical backup'
        before=sha(target);shutil.copy2(target,saved);assert sha(saved)==before
        source=PAPER/'图件'/f'Figure_{i}.pdf';shutil.copy2(source,target)
        records.append(dict(target=str(target),before_sha256=before,after_sha256=sha(target),backup=str(saved)))
        svg=target.with_suffix('.svg')
        if svg.exists():
            saved_svg=backup/f'figure_{i}_source_{j}.svg';shutil.copy2(svg,saved_svg)
        shutil.copy2(PAPER/'图件'/f'Figure_{i}.svg',svg)
        records.append(dict(target=str(svg),after_sha256=sha(svg),kind='Complete SVG derived from PDF; original AI layer history unavailable.'))
write(WORK/'original_figure_overwrite_receipt.json',dict(records=records,old_EAAI_package_untouched=True))
shutil.copy2(WORK/'original_figure_overwrite_receipt.json',PAPER/'核验与修改说明/original_figure_overwrite_receipt.json')
print('Original PDF masters updated; complete outline SVGs placed alongside. Old package unchanged.')
