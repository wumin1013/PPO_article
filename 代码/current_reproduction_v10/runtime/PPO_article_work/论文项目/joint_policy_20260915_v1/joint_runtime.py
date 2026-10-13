"""Three held actions, continuously updated LOS, and the frozen R2 projector."""
from __future__ import annotations
import copy
import importlib.util
import math
import sys
import numpy as np
from geometry_spec import Corridor, GateTracker
from study_io import BASE, ROOT, sha

I = ROOT/'论文项目/revision_20260913_preview_efficiency_v2'


def load_source(name, path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError('Frozen dependency changed: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


projector = load_source('joint_frozen_r2', I/'recursive_kcm_roundoff_v1/projector.py',
    'aad4650e72de3c4ec47abc7e7c77fe325575297047acb3fa6f6c928768433d82')
tracking = load_source('joint_frozen_demand', I/'no_kcm_coverage_v2/tracking.py',
    'c85e6097311643ec1ba8ef76bd1272dfbe719c847ca57341e531f5d0ea30cabb')
terminal = load_source('joint_frozen_terminal', ROOT/'论文项目/revision_20260913_experiments_v1/planar_metrics.py',
    'e2be506c926013da1b7929b000aad8350e3a6ecf0c74f5d3c95ad2c415696b3f')

DT = .001
MACRO_TICKS = 20
FILTER_TAU = .012
SENSOR_DISTANCES = np.array([0., .8, 1.5, 3., 4.6, 7., 10., 15., 22., 30.])


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def map_action(latent, fixed_d=None):
    bounded = np.tanh(np.asarray(latent, float))
    if bounded.shape != (3,) or not np.isfinite(bounded).all():
        raise ValueError('Three finite latent actions required')
    return np.array([math.pi/2*bounded[0], 50.*(1+bounded[1]),
                     2.7+1.9*bounded[2] if fixed_d is None else float(fixed_d)])


class JointRuntime:
    max_preview = 4.6
    decode_action = staticmethod(map_action)

    def __init__(self, reference, *, fixed_d=None, max_time=8., closed=False):
        self.corridor = Corridor(reference, closed=closed)
        if self.corridor.adjacent_gate_conflicts():
            raise ValueError('Unresolved intersecting ordered gates')
        if fixed_d is not None and not .8 <= fixed_d <= self.max_preview:
            raise ValueError('Fixed D must use the common action range')
        self.fixed_d = fixed_d
        self.max_ticks = int(round(max_time/DT))
        self.position = self.corridor.reference[0].copy()
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.heading = math.atan2(*self.corridor.tangents[0][::-1])
        self.held = np.array([0.,30.,3. if fixed_d is None else fixed_d])
        self.tracker = GateTracker(self.corridor, self.position)
        self.ticks = 0
        self.decisions = 0
        self.done = False
        self.success = False
        self.reason = None
        self.brake = None
        self.brake_index = 0
        self.rest_ticks = 0

    def state(self):
        """Exact dynamic state; shared immutable geometry is not copied per step."""
        keys = ('position','velocity','acceleration','heading','held','ticks','decisions',
                'done','success','reason','brake','brake_index','rest_ticks')
        value = {key:copy.deepcopy(getattr(self,key)) for key in keys}
        value['tracker'] = copy.deepcopy({key:value for key,value in vars(self.tracker).items()
                                         if key != 'corridor'})
        return value

    def restore(self, state):
        for key,value in state.items():
            if key == 'tracker':
                for tkey,tvalue in value.items():
                    setattr(self.tracker,tkey,copy.deepcopy(tvalue))
            else:
                setattr(self,key,copy.deepcopy(value))

    def progress(self):
        # The gate frontier chooses the source segment, including at crossings.
        i = min(self.tracker.frontier, len(self.corridor.lengths)-1)
        distance = float((self.position-self.corridor.starts[i])@self.corridor.tangents[i])
        return float(self.corridor.arcs[i]+np.clip(distance,0.,self.corridor.lengths[i])), i

    def sample_reference(self, arcs):
        arcs = np.clip(np.asarray(arcs,float),0.,self.corridor.arcs[-1])
        index = np.minimum(np.searchsorted(self.corridor.arcs,arcs,side='right')-1,
                           len(self.corridor.lengths)-1)
        positions = self.corridor.starts[index]+(arcs-self.corridor.arcs[index])[...,None]*self.corridor.tangents[index]
        return positions, self.corridor.tangents[index]

    def guidance_target(self):
        """Reference query used by the guide; independent of coverage auditing."""
        s,_ = self.progress()
        return self.sample_reference(s+self.held[2])[0]

    def observation(self):
        s,index = self.progress()
        tangent = self.corridor.tangents[index]
        normal = np.array([-tangent[1],tangent[0]])
        frame = np.stack([tangent,normal],axis=1)
        point = self.corridor.starts[index]+(s-self.corridor.arcs[index])*tangent
        samples,directions = self.sample_reference(s+SENSOR_DISTANCES)
        relative = (samples-self.position)@frame/30.
        local_directions = directions@frame
        valid = ((s+SENSOR_DISTANCES)<=self.corridor.arcs[-1]+1e-9).astype(float)
        angle = wrap(self.heading-math.atan2(tangent[1],tangent[0]))
        gate_distance = ((self.corridor.gates[self.tracker.frontier].point-self.position)@frame/30.
                         if self.tracker.frontier<len(self.corridor.gates) else np.zeros(2))
        scalars = np.r_[(self.position-point)@frame/1.25,self.velocity@frame/100.,
            self.acceleration@frame/2000., math.cos(angle),math.sin(angle),
            self.held/np.array([math.pi/2,100.,4.6]), gate_distance,
            (self.corridor.reference[-1]-self.position)@frame/30.,
            min((self.corridor.arcs[-1]-s)/30.,1.),s/self.corridor.arcs[-1],
            self.tracker.frontier/max(len(self.corridor.gates),1),
            float(self.tracker.pending_negative),self.ticks/self.max_ticks,
            float(self.brake is not None),self.rest_ticks/20.]
        return np.r_[scalars,relative.ravel(),local_directions.ravel(),valid].astype(
            getattr(self,'observation_dtype',np.float32))

    def potential(self):
        return 2.*self.progress()[0]/self.corridor.arcs[-1]

    def _consider_terminal(self):
        s,index = self.progress()
        if self.tracker.frontier < len(self.corridor.gates) or self.corridor.arcs[-1]-s>12.:
            return
        brake = terminal.plan_planar_terminal_brake(self.position,self.velocity,self.acceleration,
            dt=DT,limits=terminal.LIMITS)
        columns = brake['columns']
        endpoint = np.array([columns['x_mm'][-1],columns['y_mm'][-1]])
        if np.linalg.norm(endpoint-self.corridor.reference[-1]) <= 1.0:
            self.brake = columns
            self.brake_index = 0

    def step(self, latent):
        if self.done:
            raise RuntimeError('Cannot advance a terminated environment')
        if self.brake is not None:
            raise RuntimeError('Terminal tail is consumed within its initiating decision')
        phi = self.potential()
        start_tick = self.ticks
        self.held = self.decode_action(latent,self.fixed_d)
        self.decisions += 1
        rows = []
        self._consider_terminal()
        for micro in range(MACRO_TICKS+2100):
            if self.brake is None and micro>=MACRO_TICKS:
                break
            old_v,old_a = self.velocity.copy(),self.acceleration.copy()
            terminal_tick = self.brake is not None
            if terminal_tick:
                self.brake_index += 1
                k = self.brake_index
                self.velocity = np.array([self.brake['velocity_x_mm_s'][k],self.brake['velocity_y_mm_s'][k]])
                self.acceleration = (self.velocity-old_v)/DT
                jerk = (self.acceleration-old_a)/DT
                desired = self.velocity.copy()
                rate,guard = 0.,False
            else:
                target = self.guidance_target()
                difference = target-self.position
                beta = math.atan2(difference[1],difference[0]) if np.linalg.norm(difference)>1e-12 else self.heading
                increment = (1-math.exp(-DT/FILTER_TAU))*wrap(beta+self.held[0]-self.heading)
                self.heading = wrap(self.heading+increment)
                rate = increment/DT
                desired = self.held[1]*np.array([math.cos(self.heading),math.sin(self.heading)])
                _,_,demand,_ = tracking.compute_tracking_demand(old_v,old_a,desired,rate)
                self.velocity,self.acceleration,jerk,diagnostics = projector.enforce_kinematic_constraints(old_v,old_a,demand)
                guard = diagnostics['guard_used']
            self.position = self.position+DT*self.velocity
            self.tracker.update(self.position)
            self.ticks += 1
            rows.append(np.r_[self.position,self.velocity,self.acceleration,jerk,desired,
                               rate,float(guard),float(terminal_tick)])
            rest = np.linalg.norm(self.velocity)<=.001 and np.linalg.norm(self.acceleration)<=.01
            self.rest_ticks = self.rest_ticks+1 if rest else 0
            ratios = [np.linalg.norm(self.velocity)/100.,np.linalg.norm(self.acceleration)/2000.,np.linalg.norm(jerk)/62500.]
            if not self.tracker.inside:
                self.done,self.reason = True,'corridor_violation'
            elif max(ratios)>1.00001:
                self.done,self.reason = True,'kinematic_violation'
            elif terminal_tick and self.brake_index == len(self.brake['time_s'])-1:
                geometry = self.tracker.snapshot(endpoint_tolerance=1.25)
                self.success = bool(geometry['geometry_complete'] and self.rest_ticks>=21)
                self.done,self.reason = True,('success' if self.success else 'terminal_incomplete')
            elif self.ticks>=self.max_ticks:
                self.done,self.reason = True,'task_timeout'
            if self.done:
                break
        elapsed = (self.ticks-start_tick)*DT
        reward = -elapsed+(0. if self.done else self.potential())-phi
        if self.done and not self.success:
            reward -= self.max_ticks*DT+1.
        info = dict(rows=np.asarray(rows),elapsed_s=elapsed,success=self.success,reason=self.reason,
            held_action=self.held.copy(),gates=self.tracker.frontier,terminal_tail=bool(self.brake is not None))
        return self.observation(),float(reward),self.done,info
