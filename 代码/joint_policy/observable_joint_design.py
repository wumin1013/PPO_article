"""Linear coefficient design for the existing causal three-action residual."""
import numpy as np
from observable_memory_joint_policy import dynamic_context, future_bend_context
from joint_teacher_state_policy import features, OUTPUT_SCALE


def design(md,replay):
    observations=replay['observations']
    f=features(md,observations,replay['innovations'])
    count=md['dynamic_context_count']
    context=np.tanh((dynamic_context(observations)[...,None,:count]-
        md.get('per_center_context_mean',md['dynamic_context_mean']))/
        md.get('per_center_context_scale',md['dynamic_context_scale']))
    context=np.broadcast_to(context,(*f.shape[:-1],count))
    v,a=context[...,0:1],context[...,1:2]
    basis=md.get('dynamic_context_basis','linear')
    if basis!='linear':context=np.concatenate([context,v*a]+([v*v,a*a] if basis=='quadratic' else []),axis=-1)
    geometry_count=md.get('geometry_context_terms',0)
    if geometry_count:
        geom=np.tanh((future_bend_context(observations)[...,None]-md['per_center_geometry_mean'])/
                     md['per_center_geometry_scale'])[...,None]
        context=np.concatenate([context]+[geom,v*geom,a*geom,geom*geom][:geometry_count],axis=-1)
    f=np.concatenate([f]+[f*context[...,k,None] for k in range(context.shape[-1])],axis=-2)
    decays=np.asarray(md['memory_decays'])[:,None,None]
    rows=[];previous_path=None
    for path,fi in zip(replay['path_ids'],f):
        if path!=previous_path:
            memory=np.zeros((len(decays),*fi.shape));previous=np.zeros_like(fi);previous_path=path
        memory=decays*memory+(1.-decays)*previous;previous=fi.copy()
        rows.append(np.concatenate([fi,*memory],axis=0))
    return np.asarray(rows)


def correction(md,features):
    coefficient=np.asarray(md['coefficients']).reshape(-1,3)
    count=features.shape[-2]//(len(md['memory_decays'])+1)
    value=(features[:,:count]*coefficient[:count]).sum(axis=-2)*OUTPUT_SCALE
    for i in range(len(md['memory_decays'])):
        start=(i+1)*count;stop=start+count
        if np.any(coefficient[start:stop]):
            value+=(features[:,start:stop]*coefficient[start:stop]).sum(axis=-2)*OUTPUT_SCALE
    return value
