"""Dense action-response Jacobians before nonlinear physical-shape aggregation."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from scipy.linalg import null_space
from local_suffix_credit_policy import LocalSuffixCreditPolicy
from learn_prefix_preserving_suffix_credit import prefix_indices
from learn_retained_joint_growth import growth_deficits
from learn_trace_resolved_joint_credit import signed_channels,trace_deficits
from learn_absolute_shape_joint_credit import absolute_shape_response


def branch_vector(record,baseline,retained):
    if not record['success']:return None
    previous=next(r for r in baseline['records'] if r['path']==record['path'])
    summary=dict(baseline['summary'])
    if record['path'] not in (48,57):
        summary['smooth_curved_cruise99_s']+=record['smooth_cruise99_s']-previous['smooth_cruise99_s']
    return growth_deficits(dict(records=[record if r['path']==record['path'] else r for r in baseline['records']],summary=summary),retained)


def build_model(baseline,metadata,source,sampled,queries,retained,amplitude,center_coefficients=None):
    p=LocalSuffixCreditPolicy(metadata['parent'],metadata['mean'],metadata['std'],metadata['centers'],metadata['radius'],precision='float64')
    support,protected=[],[]
    for r in baseline['records']:
        with np.load(source/r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
        support.append(obs);protected.append(obs[prefix_indices(ref,rows,obs)])
    support,protected=np.vstack(support),np.vstack(protected)
    qobs=np.array([q['observation'] for q in queries],np.float32)
    fq,fa,fp=[p.features(torch.tensor(x)).detach().numpy() for x in (qobs,support,protected)]
    transforms=[null_space(fp[:,:,h],rcond=1e-12) for h in range(3)]
    n=sum(t.shape[1] for t in transforms);qa=np.zeros((len(queries),3,n));aa=np.zeros((len(support),3,n))
    offset=0
    for h,t in enumerate(transforms):
        k=t.shape[1];qa[:,h,offset:offset+k]=fq[:,:,h]@t*p.output_scale[h].item()
        aa[:,h,offset:offset+k]=fa[:,:,h]@t*p.output_scale[h].item();offset+=k
    base=growth_deficits(baseline,retained);jac=np.zeros((len(base),n))
    paths=[r['path'] for r in baseline['records']];old={r['path']:r for r in baseline['records']}
    margins=np.array([r['corridor']['margin_lower_mm'] for r in baseline['records']]);mj=np.zeros((len(paths),n))
    for i,q in enumerate(queries):
        assert q['stratum_size']==1
        local_amplitude=np.asarray(q.get('amplitude',amplitude))
        b=margins[paths.index(q['path'])]
        for h in range(3):
            lo,hi=q['records'][2*h:2*h+2]
            lv,hv=branch_vector(lo,baseline,retained),branch_vector(hi,baseline,retained)
            difference=((hv-lv)/(2*local_amplitude[h]) if lv is not None and hv is not None else
                        (hv-base)/local_amplitude[h] if hv is not None else
                        (base-lv)/local_amplitude[h] if lv is not None else np.zeros_like(base))
            jac+=np.outer(difference,qa[i,h])
            dm=((hi['corridor']['margin_lower_mm']-lo['corridor']['margin_lower_mm'])/(2*local_amplitude[h]) if lv is not None and hv is not None else
                (hi['corridor']['margin_lower_mm']-b)/local_amplitude[h] if hv is not None else
                (b-lo['corridor']['margin_lower_mm'])/local_amplitude[h] if lv is not None else 0.)
            mj[paths.index(q['path'])]+=dm*qa[i,h]
    models={}
    for r in baseline['records']:
        if 'shape' not in r:continue
        with np.load(source/r['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        nominal=signed_channels(ref,rows);tick=np.arange(len(rows));response={k:[] for k in ('incoming','outgoing','omega','turn')};features=[]
        for i,q in enumerate(queries):
            if q['path']!=r['path']:continue
            local_amplitude=np.asarray(q.get('amplitude',amplitude))
            for h in range(3):
                sides=[]
                for record in q['records'][2*h:2*h+2]:
                    if record['success']:
                        with np.load(sampled/record['trace']) as z:branch_rows=z['rows'].copy()
                        sides.append(signed_channels(ref,branch_rows,nominal['grid'],tick))
                    else:sides.append(None)
                low,high=sides
                if low is None and high is None:continue
                for key in response:
                    d=((high[key]-low[key])/(2*local_amplitude[h]) if low is not None and high is not None else
                       (high[key]-nominal[key])/local_amplitude[h] if high is not None else
                       (nominal[key]-low[key])/local_amplitude[h])
                    response[key].append(d)
                features.append(qa[i,h])
        f=np.asarray(features)
        models[r['path']]=dict(base=nominal,jac={k:np.asarray(v).T@f for k,v in response.items()})
        print('DENSE_TRACE_MODEL',r['path'],'complete_action_columns',len(features),flush=True)
    return dict(policy=p,base=base,metric_jac=jac,models=models,transforms=transforms,protected=protected,
        anchors=(aa/np.array([.003,.025,.025])[None,:,None]).reshape(-1,n),margins=margins,margin_jac=mj,paths=paths,
        baseline=baseline,retained=retained,center_coefficients=center_coefficients)


def values(model,z,linear=False):
    if linear:return model['base']+model['metric_jac']@z,model['metric_jac']
    return trace_deficits(z,model['base'],model['metric_jac'],model['models'],model['baseline'],model['retained'])


def objective(model,z,linear=False):
    v,j=values(model,z,linear)
    v,j=absolute_shape_response(v,j,model['baseline']['records'],model['retained'])
    positive=np.maximum(v,0.)
    return float(positive@positive),2*j.T@positive


def coefficients(model,z):
    output=[];offset=0
    for t in model['transforms']:
        k=t.shape[1];output.append(t@z[offset:offset+k]);offset+=k
    result=np.column_stack(output)
    center=model.get('center_coefficients')
    return result if center is None else result+center


def coordinates(model,metadata):
    center=model.get('center_coefficients')
    residual=metadata['coefficients'] if center is None else metadata['coefficients']-center
    z=np.concatenate([t.T@residual[:,h] for h,t in enumerate(model['transforms'])])
    assert abs(coefficients(model,z)-metadata['coefficients']).max()<1e-9
    return z
