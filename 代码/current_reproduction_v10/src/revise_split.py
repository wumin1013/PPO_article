"""Before training/test execution, separate the related semicircular programs."""
import json
import shutil
import numpy as np
from common import ROOT, sha, write_json, utc

path=ROOT/'data'/'manifest.json'
old=ROOT/'data'/'manifest_v1.json'
if old.exists():raise FileExistsError(old)
shutil.copy2(path,old)
m=json.loads(path.read_text('utf-8'))
for row in m['paths']:
    if row['id']=='train_semicircle':row['split']='development';row['group']='semicircular_program_development_only'
t=np.linspace(0,1,42)
control=np.array([[0,0],[25,30],[45,-30],[70,0]],float)
points=(1-t[:,None])**3*control[0]+3*(1-t[:,None])**2*t[:,None]*control[1]+3*(1-t[:,None])*t[:,None]**2*control[2]+t[:,None]**3*control[3]
file=ROOT/'data'/'train_bezier.csv'
np.savetxt(file,points,delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12g')
m['paths'].append(dict(id='train_bezier',split='train',group='cubic_s_bend_program',origin='synthetic',generator='revise_split.py',sha256=sha(file),vertices=len(points),length_mm=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),closed=False))
m['amendment']=dict(utc=utc(),prior_sha256=sha(old),reason='Semicircular training and mixed line/semicircle test may be related variants. Move the former to development before network training or test execution. Add a noncircular cubic training program. Test bytes unchanged.',test_runs_before_amendment=0,learned_models_before_amendment=0)
write_json(path,m)
