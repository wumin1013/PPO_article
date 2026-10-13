"""Plot nine synthetic paths only after the complete selected final-study freeze.

No module import reads held-out geometry. The CLI checks every frozen final
model and config before reading the manifest or reference CSVs. Output follows
the existing manuscript's Times font and editable vector figure conventions.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import revision_plot as common
import preview_unified_baselines as gate
import preview_saved_artifacts as archived

FAMILIES=('mixed_lengths','dense_corners','line_curve')
SPLITS=('train','validation','test')
SPLIT_LABEL={'train':'Training','validation':'Validation','test':'Held-out test'}


def select_nine_paths(manifest):
    rows=[row for row in manifest['paths'] if row['split'] in SPLITS]
    expected={(family,split) for family in FAMILIES for split in SPLITS}
    pairs=[(row['family'],row['split']) for row in rows]
    if len(pairs)!=9 or set(pairs)!=expected:
        raise ValueError('Require exactly one path for each of the three families and three registered splits')
    return sorted(rows,key=lambda row:(FAMILIES.index(row['family']),SPLITS.index(row['split'])))


def checked_points(row,data_dir):
    if row['source']['kind']!='synthetic' or row['coordinate_unit']!='mm' or row['plane']!='XY':
        raise ValueError('This figure contract requires synthetic XY paths in millimetres')
    path=data_dir/row['reference_file']
    if gate.sha(path)!=row['reference_sha256']:raise ValueError(f'Reference CSV hash mismatch: {row["id"]}')
    points=np.loadtxt(path,delimiter=',',skiprows=1)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError('Need finite Nx2 reference CSV')
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    if len(points)-1!=row['statistics']['segment_count'] or not np.isclose(length,row['statistics']['path_length_mm'],rtol=0,atol=1e-7):
        raise ValueError('Manifest geometry statistics do not match the reference CSV')
    return path,points


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=['direct-fine','direct-explore'],default='direct-explore')
    args=parser.parse_args()
    candidate=gate.final_study(args.study)
    frozen,verified_runs=archived.final_saved(candidate,gate.v2.DATA/'manifest.json')
    manifest_path=gate.v2.DATA/'manifest.json'
    manifest=gate.read(manifest_path)
    rows=select_nine_paths(manifest)
    out=candidate.OUT/'report/final_paths'
    out.mkdir(parents=True,exist_ok=True)
    sources={manifest_path.relative_to(common.ROOT).as_posix():gate.sha(manifest_path),
        (candidate.OUT/'freeze.json').relative_to(common.ROOT).as_posix():gate.sha(candidate.OUT/'freeze.json'),
        (candidate.OUT/'protocol.json').relative_to(common.ROOT).as_posix():gate.sha(candidate.OUT/'protocol.json')}
    for item in verified_runs.values():
        for name in ['source.zip','provenance.json','training_summary.json','checkpoint.pt','config.yaml']:
            path=item['run']/name
            sources[path.relative_to(common.ROOT).as_posix()]=gate.sha(path)
    common.style()
    fig,axes=plt.subplots(3,3,figsize=(7.2,6.4),gridspec_kw={'height_ratios':[1,.55,1]},constrained_layout=True)
    csv_rows=[]
    for index,row in enumerate(rows):
        file,points=checked_points(row,gate.v2.DATA)
        sources[file.relative_to(common.ROOT).as_posix()]=gate.sha(file)
        stats=row['statistics'];i,j=divmod(index,3);ax=axes[i,j]
        ax.plot(points[:,0],points[:,1],color='#1f77b4',lw=1.1)
        ax.scatter(points[0,0],points[0,1],s=13,color='#60666c',marker='o',zorder=3)
        ax.scatter(points[-1,0],points[-1,1],s=13,color='#d78332',marker='s',zorder=3)
        ax.set_aspect('equal',adjustable='box')
        ax.set(xlabel='x (mm)',ylabel='y (mm)')
        ax.set_title(f"{SPLIT_LABEL[row['split']]}: {common.FAMILY_LABEL[row['family']]}\n{stats['path_length_mm']:.2f} mm; {stats['segment_count']} segments",fontsize=7.5)
        ax.grid(alpha=.15,ls=':');ax.margins(.10)
        if row['family']=='dense_corners':
            ax.set_yticks([0])  # Equal-aspect long paths leave little vertical label space.
        ax.annotate(chr(97+index),xy=(-.14,1),xycoords='axes fraction',
                    xytext=(0,14),textcoords='offset points',weight='bold')
        quantiles=stats['segment_length_quantiles_mm']
        csv_rows.append(dict(path_id=row['id'],family=row['family'],split=row['split'],source_kind='synthetic',
            path_length_mm=stats['path_length_mm'],segment_count=stats['segment_count'],
            minimum_segment_mm=quantiles['min'],median_segment_mm=quantiles['median'],maximum_segment_mm=quantiles['max'],
            coordinate_unit='mm',plane='XY',reference_file=row['reference_file'],reference_sha256=row['reference_sha256'],
            discretization_bound_mm=row['source'].get('analytic_to_polyline_deviation_bound_mm')))
    fig.suptitle('Synthetic planar benchmark paths',fontsize=10)
    fig.legend(handles=[Line2D([],[],color='#60666c',marker='o',ls='',label='Start'),
        Line2D([],[],color='#d78332',marker='s',ls='',label='End')],loc='outside lower center',ncol=2,fontsize=7)
    common.save(fig,'final_paths')
    with (out/'geometry_summary.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    images={f'figures/generated/revision_final_paths.{ext}':gate.sha(common.FIGURES/f'revision_final_paths.{ext}') for ext in ['pdf','svg','png']}
    gate.write(out/'sources.json',dict(study=args.study,synthetic=True,contains_real_cam_data=False,
        frozen_formal_model_count=len(frozen['models']),source_hashes=sources,generator_sha256=gate.sha(Path(__file__)),
        archived_verifier_sha256=gate.sha(Path(archived.__file__)),
        source_rows=rows,geometry_summary=csv_rows,image_hashes=images,
        note='Train/validation plus newly held-out geometries only; previously exposed V1 test paths are development and omitted',
        command=f'python PPO_project/tools/preview_final_paths_figure.py --study {args.study}'))
    tex=common.figure('final_paths',
        'Synthetic planar reference paths in the final registered split. Columns show training, validation, and held-out test paths; rows show mixed segment lengths, dense corners, and mixed lines/curves. Each panel gives polyline length and linear segment count. Axes are in millimetres with equal geometric scaling within each panel. Dots and squares indicate the start and end. These paths are generated benchmarks, not industrial CAM exports. Geometric discretization error is distinct from the planning tolerance.',
        'fig:final_synthetic_paths')
    (common.GENERATED/'final_paths.tex').write_text('\n'.join(tex),encoding='utf-8')
    print(out/'sources.json')


if __name__=='__main__':main()
