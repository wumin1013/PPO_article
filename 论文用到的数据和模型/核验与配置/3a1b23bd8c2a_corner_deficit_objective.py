"""Training-only cost for excess deficit in the already registered34windows."""
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,write,sha
from natural_case_metrics import ordered_projection

WEIGHT=20.


class CornerDeficitCost:
    def __init__(self,reference,geometry,*,dt=.001,metric_paths=None):
        assert dt in [.001,.0005]
        assert dt==.001 or metric_paths is not None
        self.dt=float(dt)
        self.reference=reference
        self.corners={c['id']:c for c in geometry['sharp_corners']}
        protocol_path=ROOT/'joint_lowdim_acceptance_20260922.json'
        protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
        self.source_sha256={str(protocol_path):sha(protocol_path)}
        sources=[]
        entries=protocol['original']['comparators'].values() if metric_paths is None else [
            dict(path=str(p),sha256=sha(p)) for p in metric_paths]
        for entry in entries:
            assert sha(entry['path'])==entry['sha256']
            self.source_sha256[entry['path']]=entry['sha256']
            payload=json.loads(Path(entry['path']).read_text(encoding='utf-8'))
            if 'metrics' in payload:
                assert sha(payload['source'])==payload['source_sha256']
                self.source_sha256[payload['source']]=payload['source_sha256']
                payload=payload['metrics']
            if 'dt_s' in payload:assert abs(payload['dt_s']-self.dt)<1e-12
            sources.append({c['id']:c['speed_deficit_mm'] for c in payload['corners']})
        assert len(self.corners)==34 and all(set(s)==set(self.corners) for s in sources)
        self.budgets={key:min(s[key] for s in sources)+1e-7 for key in self.corners}
        self.current=None

    def reset(self,corner_id):
        self.current=corner_id
        self.accumulated=0.
        self.finished=False

    def consume(self,rows):
        if self.current is None:
            raise RuntimeError('A training fragment must identify its registered corner')
        c=self.corners[self.current]
        lo=c['reference_s_mm']-c['window_before_mm']
        hi=c['reference_s_mm']+c['window_after_mm']
        progress=ordered_projection(self.reference,rows[:,:2])
        mask=(progress>=lo)&(progress<=hi)
        added=float(np.maximum(100.-np.linalg.norm(rows[mask,2:4],axis=1),0.).sum()*self.dt)
        limit=self.budgets[self.current]
        before=max(self.accumulated-limit,0.)
        self.accumulated+=added
        after=max(self.accumulated-limit,0.)
        self.finished=bool(self.finished or (progress>hi).any())
        return WEIGHT*(after-before)/100.,dict(corner=self.current,deficit_mm=self.accumulated,
            budget_mm=limit,window_passed=self.finished,added_deficit_mm=added)


def verify_saved_trace():
    """Check incremental costs against the independent stored full metrics."""
    from evaluate_historical_dense import BASE
    case=BASE/'paper_evidence_review/natural_closed_case'
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    stage=ROOT/'direct14_previewcue_ppo_seed922302_round13'
    metric=json.loads((stage/'eval_0024_original_metrics.json').read_text(encoding='utf-8'))
    with np.load(stage/'eval_0024_original.npz') as data:
        rows=data['rows']
    cost=CornerDeficitCost(reference,geometry)
    errors=[];penalty_errors=[];records=[]
    for c in metric['corners']:
        cost.reset(c['id']);total=0.
        window=rows[max(0,c['first_index']-25):c['last_index']+26]
        for start in range(0,len(window),20):
            penalty,_=cost.consume(window[start:start+20]);total+=penalty
        errors.append(abs(cost.accumulated-c['speed_deficit_mm']))
        expected=WEIGHT*max(c['speed_deficit_mm']-cost.budgets[c['id']],0.)/100.
        penalty_errors.append(abs(total-expected))
        assert cost.finished
        records.append(dict(id=c['id'],deficit_mm=cost.accumulated,budget_mm=cost.budgets[c['id']],penalty_s=total))
    assert max(errors)<1e-10 and max(penalty_errors)<1e-10
    return dict(corners=records,maximum_metric_error_mm=max(errors),maximum_telescoping_error_s=max(penalty_errors),
        source_sha256=cost.source_sha256,trace_sha256=sha(stage/'eval_0024_original.npz'),
        weight=WEIGHT,new_physics_ticks=0,new_training_updates=0,
        scope='Training-only incremental cost; no change to observations,execution or final acceptance')


if __name__=='__main__':
    report=verify_saved_trace()
    write(ROOT/'corner_deficit_objective_saved_trace_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['corners','source_sha256']}))
