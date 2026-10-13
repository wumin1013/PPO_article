"""Generate every changed quantitative paper artifact from the complete v9 grid.

Only file paths are remapped when using a separately extracted evidence bundle.
Scientific records are never edited, omitted or replaced by successful subsets.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

OLD_ROOT=Path(r'D:\PPO_article_work\JNN_efficiency_20261012_v9')
OLD_BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
METHODS=('JNN','FIXED','S2','S2R','SCMO','COPP','NNC')
CASES=('original','dense','piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10')
DT=(.001,.0005)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def put(out,name,text): (out/name).write_text(text,encoding='utf-8')
def macros(out,name,values):
    put(out,name,''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in values.items()))
def time_text(v,d=3):return '--' if v is None else f'{v:.{d}f}'

def generate(root,base,out):
    root=root.resolve();base=base.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(base/'src'),str(base/'manuscript')]
    from plot_main_v9 import generate as main_plots, LABEL
    from plot_figure8_v9 import generate as figure8
    from plot_figure9_v9 import generate as figure9
    def resolve(value):
        path=Path(value)
        for old,new in ((OLD_ROOT,root),(OLD_BASE,base)):
            try:return new/path.relative_to(old)
            except ValueError:pass
        for allowed in (root,base):
            if path.is_relative_to(allowed):return path
        raise ValueError('Evidence path outside the two explicit roots: '+str(path))
    comparison_path=root/'canonical/final_comparison.json'
    comparison_hash=sha(comparison_path)
    data=read(comparison_path);rows=data['rows']
    assert len(rows)==56 and {(r['case'],r['method'],r['dt_s']) for r in rows}=={
        (c,m,d) for c in CASES for m in METHODS for d in DT}
    definitions=data['cases']
    if isinstance(definitions,list):definitions={x.get('id',x.get('case')):x for x in definitions}
    refs={};sources={};local_definitions={}
    for case in CASES:
        definition=definitions[case];local_definitions[case]={}
        for field in ('reference','geometry'):
            path=resolve(definition[field]);assert sha(path)==definition[field+'_sha256']
            local_definitions[case][field]=str(path);sources[str(path)]=sha(path)
        refs[case]=np.loadtxt(local_definitions[case]['reference'],delimiter=',',skiprows=1)
    plots={};lookup={};audit=[]
    for row in rows:
        r=dict(row);m=dict(r['metrics']);key=f"{r['case']}_{r['method']}_dt{round(r['dt_s']*1e6)}us"
        r['id']=key;lookup[key]=r
        assert m['success']==(m['complete_time_s'] is not None)
        for field in ('record','trace'):
            if r.get(field):
                p=resolve(r[field]);assert sha(p)==r[field+'_sha256'],key
                sources[str(p)]=sha(p);r[field]=str(p)
        s=dict(positions=None,time_s=None,speed_mm_s=None,distance_mm=None,deficit_mm=None,success=m['success'])
        if r.get('trace'):
            with np.load(r['trace'],allow_pickle=False) as z:p=z['positions'].copy();ref=z['reference']
            assert np.array_equal(ref,refs[r['case']]),key
            assert p.shape[1]==2 and np.isfinite(p).all();s['positions']=p
            dt=r['dt_s'];n=len(p)-1
            if n:
                v=np.diff(p,axis=0)/dt;a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
                j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
                norms=np.column_stack([np.linalg.norm(x,axis=1) for x in (v,a,j)])
                peaks=norms.max(0);length=float(norms[:,0].sum()*dt);deficit=float((100-norms[:,0]).sum()*dt)
                assert np.allclose(peaks,m['vector_peaks'],rtol=1e-8,atol=2e-4),key
                assert abs(length-m['executed_length_mm'])<1e-6
                assert abs(n*dt-m['elapsed_prefix_s'])<1e-8
                if m['success']:
                    assert m['kinematic_pass'] and m['rest_pass'] and m['geometry']['ordered_complete'] and m['geometry']['corridor_inside']
                s.update(time_s=np.arange(1,n+1)*dt,speed_mm_s=norms[:,0],
                    distance_mm=np.r_[0,np.cumsum(norms[:,0])*dt],deficit_mm=np.r_[0,np.cumsum(100-norms[:,0])*dt])
                m.update(executed_length_mm=length,speed_deficit_mm=deficit,mean_speed_mm_s=length/(n*dt))
        else:assert not m['success']
        r['metrics']=m;r['tolerance_mm']=m.get('tolerance_mm')
        audit.append(dict(id=key,case=r['case'],method=r['method'],dt_s=r['dt_s'],
            trace_sha256=r.get('trace_sha256'),success=m['success'],complete_time_s=m['complete_time_s']))
        if r['dt_s']==.001:plots[(r['case'],r['method'])]=s
    f8rows={k:r for k,r in lookup.items() if r['case'] in CASES[:2] and r['method'] in ('JNN','S2','S2R','SCMO') and r['dt_s']==.001}
    figure8(f8rows,local_definitions,out)
    # Reuse the reviewed five full-command panels and original-program macro rules.
    manifest=dict(files={},main_model=str(base/'baseline_records/supplementary_reproduction/models/main.pt'),
        model_sha256='c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764',
        figure8_sha256=sha(out/'figure8.pdf'),runs=[r for r in lookup.values() if r['case'] in CASES[:2]])
    adapter=out/'main_plot_input';adapter.mkdir(exist_ok=True);write(adapter/'manifest.json',manifest)
    main_plots(adapter,out)
    scaled_plots={k:v for k,v in plots.items() if k[0] in CASES[2:]}
    figure9({c:refs[c] for c in CASES[2:]},scaled_plots,[a for a in audit if a['case'] in CASES[2:]],sources,out)
    def metric(case,method,dt=.001):return lookup[f'{case}_{method}_dt{round(dt*1e6)}us']['metrics']
    values={};changes=[]
    for case,stem in zip(CASES[2:],('ScaledCoupon','ScaledNeck')):
        j=metric(case,'JNN');values[stem+'ReferenceLength']=f'{np.linalg.norm(np.diff(refs[case],axis=0),axis=1).sum():.2f}'
        values[stem+'Allowance']=f"{j['tolerance_mm']:.4f}"
        for method,suffix in zip(METHODS[:5],('JNN','Fixed','SQ','SR','SCMO')):
            m=metric(case,method);assert m['success']
            values[stem+suffix+'Time']=f"{m['complete_time_s']:.3f}"
            values[stem+suffix+'Length']=f"{m['executed_length_mm']:.2f}"
            values[stem+suffix+'Deficit']=f"{m['speed_deficit_mm']:.2f}"
            values[stem+suffix+'Saving']=f"{100*(m['complete_time_s']-j['complete_time_s'])/m['complete_time_s']:.2f}"
        times=[metric(case,m)['complete_time_s'] for m in METHODS[:4]]
        assert times[0]<min(times[1:])
        values[stem+'TimingSpan']=f'{1000*(max(times)-min(times)):.1f}'
        values[stem+'SRDifference']=f"{1000*(metric(case,'S2R')['complete_time_s']-j['complete_time_s']):.1f}"
        values[stem+'SRFineDifference']=f"{1000*(metric(case,'S2R',.0005)['complete_time_s']-metric(case,'JNN',.0005)['complete_time_s']):.1f}"
        for method in METHODS:
            a,b=[metric(case,method,d)['complete_time_s'] for d in DT]
            if a is not None and b is not None:changes.append(abs(1000*(b-a)))
    values['ScaledMaxResolutionChange']=f'{max(changes):.1f}'
    # Costs retain their measured scopes rather than treating actor time as total latency.
    actor=[];refine=[];total=[];cost_rows=[]
    for case in CASES[2:]:
        for dt in DT:
            r=lookup[f'{case}_JNN_dt{round(dt*1e6)}us'];record=read(r['record']);cost=record['cost']
            original=read(resolve(record['source_record']));calls=original.get('actor_calls',original.get('policy_calls'))
            assert calls
            actor.append(1000*cost['original_planning_cost']['actor_wall_s']/calls)
            refine.append(cost['refinement_wall_s']);total.append(cost['original_run_wall_s']+cost['refinement_wall_s'])
            cost_rows.append(dict(case=case,dt_s=dt,actor_mean_ms=actor[-1],refinement_wall_s=refine[-1],complete_planning_wall_s=total[-1]))
    for stem,array in [('ScaledActorMean',actor),('BoundaryRefinementWall',refine),('CompletePlanningWall',total)]:
        values[stem+'Min']=f'{min(array):.2f}';values[stem+'Max']=f'{max(array):.2f}'
    traditional=data['cost_summary']['scaled_successful_traditional']['rows']
    assert len(traditional)==12
    pipeline=[r['combined_recorded_pipeline_wall_s'] for r in traditional]
    values['ScaledPipelineWallMin']=f'{min(pipeline):.2f}'
    values['ScaledPipelineWallMax']=f'{max(pipeline):.2f}'
    macros(out,'scaled_numbers.tex',values);write(out/'planning_cost_data.json',cost_rows)
    table=[r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Complete execution time on the enlarged synthetic programs. All numerical entries satisfy the common command-interval geometry, ordered traversal, planar vector limits and final rest. A dash denotes an incomplete execution or a planning failure.}',
        r'\label{tab:representative_comparison}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r' & \multicolumn{2}{c}{Curvature coupon (s)} & \multicolumn{2}{c}{Neck contour (s)} \\',
        r'Method & 1 ms & 0.5 ms & 1 ms & 0.5 ms \\',r'\midrule']
    for method in METHODS:
        table.append(LABEL[method]+' & '+' & '.join(time_text(metric(c,method,d)['complete_time_s'],3 if d==.001 else 4) for c in CASES[2:] for d in DT)+r' \\')
    put(out,'comparison_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n')
    put(out,'scaled_motion_results.tex',r'''J-NNC has the shortest complete duration among the feasible methods on both enlarged programs. On the curvature coupon, it requires \ScaledCouponJNNTime{} s, compared with \ScaledCouponFixedTime{}, \ScaledCouponSQTime{} and \ScaledCouponSRTime{} s for fixed preview, S2-Q and S2-R. Its lead over S2-R is \ScaledCouponSRDifference{} ms at 1 ms resolution and \ScaledCouponSRFineDifference{} ms at 0.5 ms. The trajectories remain close to the speed limit over most of this gradually turning path. Their small timing differences reflect the combined contributions of path length and the start and stop portions.

On the neck contour, J-NNC completes in \ScaledNeckJNNTime{} s, compared with \ScaledNeckSRTime{} s for S2-R and \ScaledNeckSQTime{} s for S2-Q. The corresponding time reductions are \ScaledNeckSRSaving\% and \ScaledNeckSQSaving\%. J-NNC and S2-R execute \ScaledNeckJNNLength{} and \ScaledNeckSRLength{} mm, with speed deficits of \ScaledNeckJNNDeficit{} and \ScaledNeckSRDeficit{} mm. Figure~\ref{fig:scaled_motion} shows the complete command geometry and speed histories. Equation~\eqref{eq:speed_deficit} accounts for the timing differences using both executed length and accumulated deficit.
''')
    put(out,'scaled_figure.tex',r'''\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figure9_v9.pdf}
\caption{'''+(out/'figure9_v9_caption.tex').read_text('utf-8').strip()+r'''}
\label{fig:scaled_motion}
\end{figure}
''')
    genpath=root/'boundary_refinement/generalization_summary.json';general=read(genpath)['rows'];assert len(general)==10
    gl={(r['case'],r['dt_s']):r for r in general};names={
        'hilbert_cell_sweep':'Hilbert sweep','harmonic_radial_contour':'Harmonic contour',
        'piecewise_curvature_coupon':'Curvature coupon','asymmetric_neck_contour':'Neck contour','annular_sector_raster':'Annular raster'}
    for r in general:
        for field in ('record','trace'):assert sha(resolve(r[field]))==r[field+'_sha256']
    counts=[sum(r['metrics']['success'] for r in general if r['dt_s']==d) for d in DT]
    cost=read(base/'audit/frozen_main_costs_v6.json')
    macros(out,'generated_frozen_macros.tex',dict(FrozenTrainingStageWall=f"{cost['historical_stage_wall_sum_s']:.1f}",
        FrozenFinalFitWall=f"{cost['final_action_fit_wall_s']:.1f}",FrozenTransferNominalComplete=counts[0],
        FrozenTransferFineComplete=counts[1],FrozenTransferCases=5,FrozenTransferCells=10,FrozenTransferCompleteCells=sum(counts),
        FrozenTransferPairedComplete=sum(all(gl[(c,d)]['metrics']['success'] for d in DT) for c in names)))
    table=[r'\begin{table}[tb]',r'\centering\small',r'\caption{Complete J-NNC commands on five synthetic programs with fixed policy weights and boundary-search settings. Times include braking and final rest. A dash denotes an incomplete execution; ordered-gate counts retain every program.}',
        r'\label{tab:frozen_generalization}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Program & $T_{1\,\mathrm{ms}}$ (s) & Gates & $T_{0.5\,\mathrm{ms}}$ (s) & Gates\\',r'\midrule']
    for case,label in names.items():
        cells=[]
        for dt in DT:
            m=gl[(case,dt)]['metrics'];g=m['geometry'];cells.extend([time_text(m['complete_time_s'],4),f"{g['gates_passed']}/{g['gate_count']}"])
        table.append(label+' & '+' & '.join(cells)+r'\\')
    put(out,'generalization_table.tex','\n'.join(table+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n')
    assert counts==[2,3]
    put(out,'generalization_results.tex',r'''The curvature coupon and neck contour complete at both resolutions, and the harmonic contour completes at 0.5 ms. Table~\ref{tab:frozen_generalization} includes all five programs. The remaining executions end before complete ordered traversal and receive no completion time. The resulting completion counts are \FrozenTransferNominalComplete/\FrozenTransferCases{} at 1 ms and \FrozenTransferFineComplete/\FrozenTransferCases{} at 0.5 ms.
''')
    put(out,'v9_formal_binding.tex','% Quantitative artifacts generated from the complete v9 inventory.\n% final_comparison SHA256: '+sha(comparison_path)+'\n')
    science=[dict(id=r['id'],case=r['case'],method=r['method'],dt_s=r['dt_s'],metrics={k:r['metrics'].get(k) for k in (
        'success','complete_time_s','elapsed_prefix_s','vector_peaks','executed_length_mm','speed_deficit_mm','max_polyline_deviation_mm','geometry')}) for r in lookup.values()]
    write(out/'scientific_values.json',science)
    products=[p for p in out.iterdir() if p.is_file() and (p.suffix in ('.png','.npz') or p.name in (
        'numbers.tex','scaled_numbers.tex','generated_frozen_macros.tex','complete_table.tex','comparison_table.tex',
        'generalization_table.tex','resolution_table.tex','scientific_values.json','planning_cost_data.json'))]
    assert sha(comparison_path)==comparison_hash,'Comparison changed during generation'
    write(out/'v9_generation_receipt.json',dict(comparison_sha256=comparison_hash,generalization_sha256=sha(genpath),
        runs=56,generalization_runs=10,successful_runs=sum(r['metrics']['success'] for r in rows),
        generator_sha256=sha(__file__),outputs={p.name:sha(p) for p in products},source_traces={r['id']:r.get('trace_sha256') for r in lookup.values()}))
    print(json.dumps(dict(generated=True,runs=56,generalization=counts)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=OLD_ROOT)
    p.add_argument('--base',type=Path,default=OLD_BASE);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.root,a.base,a.output)
