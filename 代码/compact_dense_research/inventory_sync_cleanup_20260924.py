"""Read-only inventory for the user-authorized PPO sync-folder consolidation."""
from pathlib import Path
import os,json,collections
ROOT=Path('D:/Nutstore/1/基于强化学习的运动规划方法')
WORK=Path('D:/PPO_article_joint_runs/natural_closed_resume/compact_dense_research/release_cleanup_20260924')
WORK.mkdir(exist_ok=True)
records=[];errors=[];summary=collections.defaultdict(lambda:dict(files=0,bytes=0,offline=0,reparse=0))
for directory,dirs,files in os.walk(ROOT,followlinks=False,onerror=lambda e:errors.append(str(e))):
    for name in files:
        p=Path(directory)/name
        try:
            st=p.stat();rel=p.relative_to(ROOT);attrs=getattr(st,'st_file_attributes',0)
            r=dict(relative=str(rel),size=st.st_size,attributes=attrs)
            records.append(r);s=summary[rel.parts[0]];s['files']+=1;s['bytes']+=st.st_size
            s['offline']+=bool(attrs&4096);s['reparse']+=bool(attrs&1024)
        except OSError as e:errors.append(str(e))
(WORK/'sync_inventory_before.json').write_text(json.dumps(dict(root=str(ROOT),records=records,errors=errors),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(summary=summary,files=len(records),bytes=sum(r['size'] for r in records),errors=errors[:10]),ensure_ascii=False))
