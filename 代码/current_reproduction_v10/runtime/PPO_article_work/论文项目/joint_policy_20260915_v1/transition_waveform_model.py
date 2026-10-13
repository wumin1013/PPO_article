"""Chain closed-loop local responses and differentiate existing shape signals."""
import numpy as np
from closed_loop_transition_credit import ACTION_SCALE
from prepare_current_waveform_credit import wave_signals
from energy_waveform_credit import energy_signals


def chain_responses(local, directions=None):
    features=local['features'];dimension=768 if directions is None else len(directions)
    response=np.zeros((7,dimension));chunks=[]
    for k,sj in enumerate(local['state_jacobian']):
        if directions is None:
            direct=np.zeros((3,768))
            for h in range(3):direct[h,h*256:(h+1)*256]=features[k,h]/ACTION_SCALE[h]
        else:
            direct=np.einsum('hk,jhk->hj',features[k],directions.reshape(-1,3,256))/ACTION_SCALE[:,None]
        a,b=local['row_offsets'][k:k+2];rj=local['row_jacobian'][a:b]
        chunks.append(np.einsum('tcq,qd->tcd',rj[:,:,:7],response)+
                      np.einsum('tch,hd->tcd',rj[:,:,7:],direct))
        response=sj[:,:7]@response+sj[:,7:]@direct
    return np.concatenate(chunks)


def signal_jacobians(reference, rows, row_gradient):
    nominal,valid=wave_signals(reference,rows)
    density,density_valid=energy_signals(reference,rows)
    n,_,dimension=row_gradient.shape
    gradient=np.zeros((n,6,dimension));energy_gradient=np.zeros((n,2,dimension))
    u,v=np.diff(reference,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);sign=np.sign(angle)
    ni=sign*np.array([-u[1],u[0]]);no=sign*np.array([-v[1],v[0]])
    gradient[:,0]=np.einsum('c,tcd->td',ni,row_gradient[:,:2])
    gradient[:,1]=np.einsum('c,tcd->td',no,row_gradient[:,:2])
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    ids=np.flatnonzero(valid[:,2]);vx,vy,ax,ay=rows[ids,2:6].T
    dvx,dvy,dax,day=[row_gradient[ids,c] for c in range(2,6)]
    cross=vx*ay-vy*ax;dot=vx[:,None]*dvx+vy[:,None]*dvy
    dc=ay[:,None]*dvx+vx[:,None]*day-ax[:,None]*dvy-vy[:,None]*dax
    s=speed[ids,None]
    gradient[ids,2]=sign*180/np.pi*(vx[:,None]*dvy-vy[:,None]*dvx)/s**2
    gradient[ids,4]=sign*(dc/s**3-3*cross[:,None]*dot/s**5)
    gradient[ids,5]=sign*180/np.pi*(dc/s**2-2*cross[:,None]*dot/s**4)
    scale=np.sqrt(speed[ids]*.001*1.25/angle**2)
    energy_gradient[ids,1]=gradient[ids,4]*scale[:,None]+density[ids,1,None]*.5*dot/s**2
    xy=np.vstack([reference[0],rows[:,:2]])
    dxy=np.concatenate([np.zeros((1,2,dimension)),row_gradient[:,:2]])
    segments=np.diff(xy,axis=0);dsegments=np.diff(dxy,axis=0)
    length=np.linalg.norm(segments,axis=1)
    ids=np.flatnonzero(valid[:,3])
    a,b=segments[ids],segments[ids+1];da,db=dsegments[ids],dsegments[ids+1]
    la,lb=length[ids],length[ids+1];chord=a+b;lc=np.linalg.norm(chord,axis=1)
    aa=np.einsum('tc,tcd->td',a,da);bb=np.einsum('tc,tcd->td',b,db)
    cc=np.einsum('tc,tcd->td',chord,da+db)
    cross_derivative=(da[:,0]*b[:,1,None]+a[:,0,None]*db[:,1]-
                      da[:,1]*b[:,0,None]-a[:,1,None]*db[:,0])
    logarithm=aa/la[:,None]**2+bb/lb[:,None]**2+cc/lc[:,None]**2
    gradient[ids,3]=sign*2*cross_derivative/(la*lb*lc)[:,None]-nominal[ids,3,None]*logarithm
    measure=.5*(la+lb);dmeasure=.5*(aa/la[:,None]+bb/lb[:,None])
    scale=np.sqrt(measure*1.25/angle**2)
    energy_gradient[ids,0]=gradient[ids,3]*scale[:,None]+density[ids,0,None]*.5*dmeasure/measure[:,None]
    assert np.isfinite(gradient).all() and np.isfinite(energy_gradient).all()
    return nominal,valid,gradient,density,density_valid,energy_gradient
