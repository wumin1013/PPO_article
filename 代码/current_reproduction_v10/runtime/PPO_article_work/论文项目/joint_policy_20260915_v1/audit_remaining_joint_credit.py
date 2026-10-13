"""Locate remaining real guard failures and validate the last measured secants."""
import numpy as np
from duration_joint_metrics import guard_names
from study_io import BASE, write_json, sha
from search_complete_return import read


OLD_NAMES = ['incoming','outgoing','reverse','energy','peak','apex_speed',
             'time','cruise','first_inside','corridor','lateral_total']


def names_for(records):
    return ([f"{r['path']}:{name}" for r in records for name in OLD_NAMES] +
            [f"{r['path']}:{name}" for r in records for name in guard_names()])


def main():
    stage=BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920'
    history=read(stage/'state_joint/results.json')['candidates']
    index={c['tag']:c for c in history}
    anchor=index['geometry_secant_17'];names=names_for(anchor['records'])
    values=np.asarray(anchor['measured_inequalities'])
    failed=[dict(index=int(i),name=names[i],value=float(values[i]))
            for i in np.flatnonzero(values < -1e-8)]
    model=read(stage/'final_secant_credit.json');jac=np.asarray(model['jacobian'])
    gradient=np.asarray(model['loss_gradient'])
    response=read(BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920/joint_main_turn_credit_models.json')
    steps=np.repeat(np.asarray(response['steps'])[:,None],7,axis=1).reshape(-1)
    def coordinates(c):
        u=np.asarray(c['coefficients']).reshape(3,7,5,3)
        return u[:,:,[0,1,3,4],:].reshape(-1)/steps
    x=coordinates(anchor);a=np.asarray(anchor['learning_inequalities']);comparisons=[]
    for c in history[-7:]:
        delta=coordinates(c)-x
        pred=a+jac@delta;actual=np.asarray(c['learning_inequalities'])
        error=actual-pred
        comparisons.append(dict(tag=c['tag'],distance=float(np.linalg.norm(delta)),
            maximum_error=float(abs(error).max()),
            failed_rows=[dict(name=names[r['index']],predicted=float(pred[r['index']]),
                actual=float(actual[r['index']]),error=float(error[r['index']])) for r in failed],
            worst_errors=[dict(name=names[i],error=float(error[i]),actual=float(actual[i]),
                predicted=float(pred[i])) for i in np.argsort(abs(error))[-5:]]))
    deltas=[]
    for r,b in zip(anchor['records'],history[0]['records']):
        keys=['full_physical_reverse_deg','out_edge_outward_mm','reverse_lateral_significant_mm',
              'reverse_lateral_significant_duration_s','full_spatial_peak_curvature_mm_inv','apex_speed_mm_s']
        deltas.append(dict(path=r['path'],changes={k:r['shape'][k]-b['shape'][k] for k in keys}))
    output=dict(anchor=anchor['tag'],failures=failed,comparisons=comparisons,path_changes=deltas,
                new_physical_runs=0,source_sha256=sha(stage/'state_joint/results.json'))
    write_json(stage/'remaining_credit_audit.json',output)
    print('FAILURES',failed,flush=True)
    for c in comparisons:
        print('CREDIT',c['tag'],c['distance'],c['maximum_error'],c['worst_errors'],flush=True)
    print('CHANGES',deltas,flush=True)


if __name__=='__main__':main()
