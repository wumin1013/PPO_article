"""Experimental union of straight-release and constant-speed turn-release certificates.

Original physical limits and finite-disk intersection are retained. This module
does not modify the historical projector or certify geometric task completion.
"""
import math
import numpy as np
from joint_runtime import projector as ORIGINAL

DT=.001
ROTATION_JERK=62375.0
EQUAL_RADIUS_TOL=1e-10

def norm(x):return math.hypot(float(x[0]),float(x[1]))

def turn_certificate(v,a):
    radius=norm(v)
    if radius<1.-EQUAL_RADIUS_TOL or radius>100.+1e-10 or norm(a)>1998.+1e-7:return False
    previous=v-DT*a
    if abs(norm(previous)-radius)>EQUAL_RADIUS_TOL:return False
    return norm(a)**2/radius<ROTATION_JERK-1.

def turn_backup(v,a):
    """Finite monotone angular-step decay, with a conservative triangle bound."""
    if not turn_certificate(v,a):raise ValueError('No constant-speed turn-release certificate')
    radius=norm(v);previous=v-DT*a
    angle=math.atan2(previous[0]*v[1]-previous[1]*v[0],previous@v)
    room=ROTATION_JERK-norm(a)**2/radius
    decrement=.5*room*DT*DT/radius
    next_angle=math.copysign(max(abs(angle)-decrement,0.),angle)
    c,s=math.cos(next_angle),math.sin(next_angle)
    candidate=np.array([c*v[0]-s*v[1],s*v[0]+c*v[1]])
    candidate*=radius/norm(candidate)
    next_a=(candidate-v)/DT;jerk=(next_a-a)/DT
    if norm(next_a)>1998.+1e-7 or norm(jerk)>62437.5+1e-5 or not turn_certificate(candidate,next_a):
        raise ValueError('Turn-release numerical transition failed unchanged internal limits')
    return candidate,next_a,jerk

def straight_certificate(v,a):
    try:return max(ORIGINAL._state_residuals(v,a,DT)[0].values())<=1e-10
    except ValueError:return False

def admissible(v,a,candidate,centers,radii):
    next_a=(candidate-v)/DT;next_j=(next_a-a)/DT
    disks=np.linalg.norm(candidate[None,:]-centers,axis=1)-radii
    physical=bool(disks.max()<=1e-10 and norm(candidate)<=100.+1e-10 and
                  norm(next_a)<=1998.+1e-7 and norm(next_j)<=62437.5+1e-5)
    if not physical:return None
    straight=straight_certificate(candidate,next_a)
    turning=turn_certificate(candidate,next_a)
    if not (straight or turning):return None
    return dict(velocity=candidate,acceleration=next_a,jerk=next_j,straight=straight,turning=turning)

def circle_candidates(radius,request,centers,radii):
    angles=[math.atan2(request[1],request[0])]
    for center,r in zip(centers[1:],radii[1:]):
        distance=norm(center)
        if distance==0.:continue
        cosine=(radius*radius+distance*distance-r*r)/(2*radius*distance)
        if -1.<=cosine<=1.:
            offset=math.acos(np.clip(cosine,-1.,1.));base=math.atan2(center[1],center[0])
            angles.extend([base-offset,base+offset])
    return [radius*np.array([math.cos(t),math.sin(t)]) for t in angles]

class TurnAwareProjector:
    def __init__(self,env):
        self.env=env;self.counts=dict(calls=0,circle=0,projected=0,backup=0,straight_backup=0,turn_backup=0)

    def enforce_kinematic_constraints(self,velocity,acceleration,desired_jerk,dt=DT):
        if dt!=DT:raise ValueError('Experimental projector is currently certified at 1ms only')
        v=np.asarray(velocity,float);a=np.asarray(acceleration,float);d=np.asarray(desired_jerk,float)
        if not np.isfinite(np.r_[v,a,d]).all():raise ValueError('Nonfinite KCM input')
        self.counts['calls']+=1
        old_ok=straight_certificate(v,a);turn_ok=turn_certificate(v,a)
        if not (old_ok or turn_ok):raise ValueError('Input lacks both recursive certificates')
        centers=np.array([np.zeros(2),v,v+DT*a]);radii=np.array([100.,1998.*DT,62437.5*DT*DT])
        if old_ok:
            magnitude=norm(a);ba=a*(max(magnitude-62437.5*DT,0.)/magnitude) if magnitude else np.zeros(2)
            backup=admissible(v,a,v+DT*ba,centers,radii);backup_type='straight_backup'
        else:
            bv,_,_=turn_backup(v,a);backup=admissible(v,a,bv,centers,radii);backup_type='turn_backup'
        if backup is None:raise ValueError('Inductive backup failed numerical certification')
        self.counts[backup_type]+=1
        clipped=ORIGINAL._clip(a+DT*ORIGINAL._clip(d,62437.5),1998.)
        request=v+DT*clipped;radius=norm(v);chosen=None;mode='backup'
        # A speed-preserving angular projection is permitted only when already
        # at the near-limit requested cruise, never while accelerating to it.
        if self.env.held[1]>=99.99999999 and radius>=99.99999995:
            candidates=[]
            for point in circle_candidates(radius,request,centers,radii):
                q=admissible(v,a,point,centers,radii)
                if q is not None:candidates.append(q)
            if candidates:
                chosen=min(candidates,key=lambda q:float(np.sum((q['velocity']-request)**2)));mode='circle'
        if chosen is None:
            projection=ORIGINAL._disks.project_disks(request,centers,radii,tolerance=1e-10)
            target=np.asarray(projection['selected']['point']) if projection['selected'] is not None else None
            if target is not None:
                chosen=admissible(v,a,target,centers,radii)
                if chosen is not None:mode='projected'
            if chosen is None and target is not None:
                # Every point is checked against the union; no convexity claim.
                for i in range(1,25):
                    fraction=.5**i;q=admissible(v,a,backup['velocity']+fraction*(target-backup['velocity']),centers,radii)
                    if q is not None:chosen=q;mode='projected';break
        if chosen is None:chosen=backup
        self.counts[mode]+=1
        return chosen['velocity'],chosen['acceleration'],chosen['jerk'],dict(guard_used=mode=='backup',
            certificate='straight_and_turn' if chosen['straight'] and chosen['turning'] else ('turn' if chosen['turning'] else 'straight'),projection_mode=mode)

def environment_factory(reference):
    import joint_runtime
    from learned_action_pulse_policy import environment_factory as historical_factory
    env=historical_factory(reference);old_step=env.step;bridge=TurnAwareProjector(env)
    def step(latent):
        previous=joint_runtime.projector
        joint_runtime.projector=bridge
        try:return old_step(latent)
        finally:joint_runtime.projector=previous
    env.step=step;env.experimental_projector=bridge
    return env
