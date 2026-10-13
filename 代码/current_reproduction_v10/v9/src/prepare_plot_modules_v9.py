"""One-time extraction of existing reviewed plot styles into versioned v9 modules."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
OUT=ROOT/'src'
sources={}
def read(name):
    p=BASE/'manuscript'/name
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p.read_text('utf-8')

main=read('generate_frozen_main_v6.py')
main=main.replace('Regenerate frozen-controller manuscript figures and tables from full traces.',
    'Generate v9 full-command plots and tables; style derived from generate_frozen_main_v6.py.')
main=main.replace("ROOT=Path(__file__).resolve().parents[1]\nsys.path.insert(0,str(ROOT/'src'))",'')
main=main[:main.index("if __name__=='__main__':")]
main=main.replace("'path.simplify':False", "'path.simplify':False,'svg.hashsalt':'jnn-v9'")
(OUT/'plot_main_v9.py').write_text(main,encoding='utf-8')

f8=read('generate_main_bound_v5.py')
header='''"""Plot v9 Figure 8 using the original geometry-defined windows and complete final commands."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from generate_development_v3 import local_frame
METHODS=('JNN','S2','S2R','SCMO')
def read(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
'''
style=f8[f8.index('LABELS ='):f8.index('def generate():')]
style=style.replace("np.load(ROOT / run['trace'])", "np.load(Path(run['trace']))")
body=f8[f8.index('def generate():'):f8.index('    def value(case,method')]
body=body.replace('def generate():\n    manifest, records = verify()\n    OUT.mkdir(parents=True, exist_ok=True)',
    'def generate(records, case_definitions, OUT):\n    OUT.mkdir(parents=True, exist_ok=True)')
body=body.replace("np.loadtxt(ROOT / record['reference'], delimiter=',', skiprows=1)",
    "np.loadtxt(case_definitions[case]['reference'], delimiter=',', skiprows=1)")
body=body.replace("read(ROOT / record['geometry'])", "read(case_definitions[case]['geometry'])")
body=body.replace('figure8_main_v5.', 'figure8.')
body=body.replace("'path.simplify':False", "'path.simplify':False,'svg.hashsalt':'jnn-v9'")
body=body.replace("lw=1.1)", "lw=1.1,rasterized=True)")
body+='''    receipt=dict(source_traces={key:r['trace_sha256'] for key,r in records.items()},
        windows={case:corner for case,(_,corner) in CASES.items()},
        selection='first geometric-window visit; complete-path deficit',
        outputs={p.name:sha(p) for p in OUT.glob('figure8*') if p.is_file() and p.name!='figure8_receipt.json'})
    (OUT/'figure8_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n',encoding='utf-8')
'''
(OUT/'plot_figure8_v9.py').write_text(header+style+body,encoding='utf-8')

f9=read('generate_scaled_comparison_v8.py')
header='''"""Plot complete v9 enlarged-program commands; layout inherited from the v8 figure contract."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_main_v9 import COLOR,LABEL,STYLE
CASES=('piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10')
TITLES=('Curvature coupon','Neck contour')
METHODS=('JNN','FIXED','S2','S2R','SCMO','COPP','NNC')
PRIMARY_DT=.001
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def generate(references,plots,audit,sources,output):
    output.mkdir(parents=True,exist_ok=True)
    receipt=dict(validated_cells=len(audit),input_sha256=sources,primary_dt_s=PRIMARY_DT,
        smoothing=False,decimation=False,temporal_cropping=False)
'''
body=f9[f9.index('    plt.rcParams.update'):f9.index('    contract_copy =')]
body=body.replace('figure9_scaled_v8','figure9_v9')
body=body.replace('"path.simplify": False', '"path.simplify": False,"svg.hashsalt":"jnn-v9"')
body+='''    receipt['outputs_sha256']={p.name:sha(p) for p in outputs}
    (output/'figure9_v9_receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n',encoding='utf-8')
'''
(OUT/'plot_figure9_v9.py').write_text(header+body,encoding='utf-8')
(OUT/'plot_source_attribution.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
