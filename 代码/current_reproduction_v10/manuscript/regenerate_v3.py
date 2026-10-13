"""Regenerate numerical paper content and figures from all retained v3 outcomes."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STEPS=['analyze_final_v3.py','generate_geometry_v3.py','generate_programs_v3.py',
       'plot_final_v3.py','generate_development_v3.py','write_evidence_text_v3.py',
       'revise_v3_source.py']

if __name__=='__main__':
    for name in STEPS:
        print('REGENERATE',name,flush=True)
        subprocess.run([sys.executable,str(ROOT/'manuscript'/name)],cwd=ROOT,check=True)
    print('Scientific source regenerated. Render both journals with journal_materials/render_journal_source.py.')
