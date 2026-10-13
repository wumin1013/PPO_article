"""Compile the local manuscript with existing TeX Live, keeping UTF-8 logs."""
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from study_io import BASE, sha, write_json

paper=BASE/'manuscript_joint_v1'
out=paper/'build_natural'
out.mkdir(exist_ok=True)
cmd=[r'D:\texlive\2025\bin\windows\latexmk.exe','-norc','-pdf','-interaction=nonstopmode',
     '-halt-on-error','-synctex=1','-outdir=build_natural','PPO_article.tex']
r=subprocess.run(cmd,cwd=paper,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
try:
    log=r.stdout.decode('utf-8')
except UnicodeDecodeError:
    log=r.stdout.decode('gb18030')
(out/'compile_console_utf8.log').write_text(log,encoding='utf-8')
print(log[-5000:])
if r.returncode:
    raise SystemExit(r.returncode)
write_json(out/'compile_receipt.json',dict(compiled_utc=datetime.now(timezone.utc).isoformat(),
    toolchain='existing TeX Live 2025, latexmk -norc',exit_code=r.returncode,
    pdf_sha256=sha(out/'PPO_article.pdf'),console_sha256=sha(out/'compile_console_utf8.log')))
