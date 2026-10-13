"""Independently audit the complete 56+10 cell matrix in an extracted workspace."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
from archive_tools import digest, read, write_new
from portable_paths import install_guard


def audit_matrix(root,out,jobs):
    v9=root/'v9';here=v9/'portable_reproduction'
    comparison_path=v9/'canonical/final_comparison.json'
    general_path=v9/'boundary_refinement/generalization_summary.json'
    comparison=read(comparison_path);general=read(general_path)
    assert len(comparison['rows'])==56 and len(general['rows'])==10
    grid=comparison['grid']
    assert {(r['case'],r['method'],r['dt_s']) for r in comparison['rows']}=={
        (c,m,d) for c in grid['cases'] for m in grid['methods'] for d in grid['dt_s']}
    assert len({(r['case'],r['dt_s']) for r in general['rows']})==10
    definitions=comparison['cases']
    if isinstance(definitions,list):
        definitions={c.get('id',c.get('case')):c for c in definitions}
    specs=[]
    for group,rows in (('comparison',comparison['rows']),('generalization',general['rows'])):
        for row in rows:
            identity=f"{group}_{row['case']}_{row['method']}_dt{round(row['dt_s']*1e6)}us"
            spec=dict(id=identity,group=group,row=row,definition=definitions.get(row['case']))
            specfile=out/'specs'/(identity+'.json');write_new(specfile,spec);specs.append((identity,specfile))
    def run_cell(spec):
        identity,specfile=spec
        log=out/'logs'/(identity+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        resultfile=out/'metrics'/(identity+'.json')
        with log.open('x',encoding='utf-8') as stream:
            process=subprocess.run([sys.executable,'-B','-X','utf8',str(here/'audit_trace_v9.py'),
                '--workspace',str(root),'--spec',str(specfile),'--output',str(resultfile)],
                cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(log.read_text('utf-8'))
        result=read(resultfile)
        assert result['guard']['guard_active'] and not result['guard']['blocked_original_accesses']
        return result
    audits=[]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending={pool.submit(run_cell,s):s[0] for s in specs}
        for future in as_completed(pending):
            result=future.result();audits.append(result)
            print(f"Audited {len(audits)}/66: {result['id']} {result['status']}",flush=True)
    write_new(out/'full_trace_audit.json',dict(comparison_sha256=digest(comparison_path),
        generalization_sha256=digest(general_path),expected_cells=66,audited_cells=len(audits),
        full_trace_cells=sum(r['trace_present'] for r in audits),
        retained_planning_failure_cells=sum(not r['trace_present'] for r in audits),
        auditor_source_sha256={str(p.relative_to(root)):digest(p) for p in (
            here/'run_trace_matrix_v9.py',here/'audit_trace_v9.py',here/'portable_paths.py',
            root/'src/metrics_exact_index_v5.py',root/'src/runtime_bootstrap.py')},
        rows=sorted(audits,key=lambda r:r['id'])))
    return audits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args();root=a.workspace.resolve();out=a.output.resolve()
    assert out.is_relative_to(root) and not out.exists()
    guard=install_guard(root);out.mkdir(parents=True)
    audit_matrix(root,out,a.jobs)
    write_new(out/'matrix_guard.json',guard)


if __name__=='__main__':
    main()
