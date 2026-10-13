"""State-learned three-action temporal residual; original physics is inherited."""
import numpy as np
import math
from gate_band_state import GateBandRuntime
from deep_joint_credit_policy import load_credit_policy
from decision_innovation_policy import innovation_features
from probe_joint_target_coordinates import guide_angle
from joint_runtime import wrap
from canonical_gate_state import CanonicalGatePolicy
from joint_anticipation_features import anticipation_features


SCALE = np.array([.12, .25, .25])


class LearnedActionPulseRuntime(GateBandRuntime):
    def __init__(self, reference, **kwargs):
        super().__init__(reference, **kwargs)
        self.pending_pulse = np.zeros(3)
        self._executed = None

    def guidance_target(self):
        if self._executed is not None:
            fraction = min((self.ticks-self._start_tick+1)/20., 1.)
            if np.any(self._pulse):
                latent = (self._latent+self._pulse*(1.-fraction)**2).astype(self._latent.dtype)
                self.held = self.decode_action(latent, self.fixed_d)
            self._executed.append(self.held.copy())
        return super().guidance_target()

    def step(self, latent):
        self._latent = np.asarray(latent).copy()
        self._pulse = self.pending_pulse.copy()
        self._start_tick = self.ticks
        self._executed = []
        observation, reward, done, info = super().step(latent)
        executed = np.full((len(info['rows']), 3), np.nan)
        count = len(self._executed)
        assert count == int((info['rows'][:, 12] == 0).sum())
        if count:
            executed[:count] = self._executed
        info['executed_actions_1ms'] = executed
        self._executed = None
        return observation, reward, done, info


def environment_factory(reference):
    length = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    return LearnedActionPulseRuntime(reference, max_time=max(5., float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1]) < 1e-9))


class LearnedActionPulsePolicy:
    def __init__(self, parent, coefficients, mode, expert_state=None,feature_spec=None):
        self.base = load_credit_policy(parent)
        self.coefficients = np.asarray(coefficients).reshape(-1, 3)
        assert self.coefficients.shape[0] in (3, 6, 9, 12)
        assert mode in ('pulse', 'constant', 'target_coordinates','expert_residual','anticipatory_expert','spatial_expert','innovation_coordinates')
        self.mode = mode
        self.env = None
        self.expert=None
        self.feature_spec=feature_spec
        if mode in ('expert_residual','anticipatory_expert','spatial_expert','innovation_coordinates'):
            assert expert_state is not None
            self.expert=CanonicalGatePolicy(256)
            self.expert.load_state_dict(expert_state)

    def bind(self, env):
        self.env = env
        self.base.bind(env)
        return env

    def act(self, observation, deterministic=True):
        original = self.base.act(observation, deterministic)[0]
        _, inputs = innovation_features(self.env, original)
        x = np.tanh(inputs/.5)
        if self.mode=='innovation_coordinates':
            from joint_innovation_coordinates import innovation_coordinate_action,anticipation_command_features
            self.env.pending_pulse=np.zeros(3)
            if not np.any(self.coefficients):return original,0.,0.
            weights=((x*x)[:,None]*self.coefficients[:3]).sum(axis=0)
            if len(self.coefficients)>=6 and np.any(self.coefficients[3:6]):
                velocity,acceleration=self.env.velocity,self.env.acceleration
                rate=(velocity[0]*acceleration[1]-velocity[1]*acceleration[0])/(velocity@velocity+25.)
                dynamic=x*np.tanh(rate/self.feature_spec['turn_rate_scale_rad_s'])
                weights+=(dynamic[:,None]*self.coefficients[3:6]).sum(axis=0)
            if len(self.coefficients)>=9 and np.any(self.coefficients[6:9]):
                speed=float(np.linalg.norm(self.env.velocity))
                phase=np.tanh(30.*observation[11]/((speed+5.)*self.feature_spec['gate_forward_time_scale_s']))
                weights+=((phase*x*x)[:,None]*self.coefficients[6:9]).sum(axis=0)
            extra=None
            if len(self.coefficients)==12 and np.any(self.coefficients[9:]):
                anticipatory=anticipation_command_features(observation)
                features=np.column_stack([anticipatory,anticipatory**2,anticipatory**2])
                extra=(features*self.coefficients[9:]).sum(axis=0)
            return innovation_coordinate_action(self.env,original,self.expert.act(observation,True)[0],weights,extra),0.,0.
        if self.mode=='spatial_expert':
            geometry=anticipation_features(observation,inputs)
            bend=.5*(geometry[0]+geometry[1])
            speed=100.*np.linalg.norm(np.asarray(observation)[2:4])
            phase=30.*np.linalg.norm(np.asarray(observation)[11:13])/(speed+5.)
            radial=np.exp(-.5*((phase-self.feature_spec['centers'])/self.feature_spec['width'])**2)
            f=np.r_[bend,bend*radial,x[1:]**2]
            self.env.pending_pulse=np.zeros(3)
            if not np.any(self.coefficients):return original,0.,0.
            weights=(f[:,None]*self.coefficients).sum(axis=0)
            difference=self.expert.act(observation,True)[0]-original
            return (original+weights*difference).astype(original.dtype),0.,0.
        if self.mode in ('expert_residual','anticipatory_expert'):
            # Even state-conditioned weights preserve the experts' reflection symmetry.
            f=x**2
            weights=(f[:,None]*self.coefficients[:3]).sum(axis=0)
            if len(self.coefficients)==6 and np.any(self.coefficients[3:]):
                if self.mode=='anticipatory_expert':
                    weights+=(anticipation_features(observation,inputs)[:,None]*self.coefficients[3:]).sum(axis=0)
                else:
                    weights+=(f[:,None]*self.coefficients[3:]).sum(axis=0)*(np.linalg.norm(self.env.velocity)-60.)/40.
            self.env.pending_pulse=np.zeros(3)
            if not np.any(self.coefficients):return original,0.,0.
            difference=self.expert.act(observation,True)[0]-original
            return (original+weights*difference).astype(original.dtype),0.,0.
        features = np.stack([x**3, x**2, x**2], axis=1)
        residual = (features*self.coefficients[:3]).sum(axis=0)*SCALE
        if len(self.coefficients)==6 and np.any(self.coefficients[3:]):
            # Continuous observed speed conditions the learned action response;
            # 60/40 only center and scale the input, never prescribe a speed.
            speed_feature=(np.linalg.norm(self.env.velocity)-60.)/40.
            residual += (features*self.coefficients[3:]).sum(axis=0)*SCALE*speed_feature
        self.env.pending_pulse = residual if self.mode == 'pulse' else np.zeros(3)
        if self.mode == 'target_coordinates' and np.any(self.coefficients):
            action=(original+residual).astype(original.dtype)
            if residual[2] != 0.:
                previous=self.env.decode_action(original,self.env.fixed_d)
                requested=self.env.decode_action(action,self.env.fixed_d)
                compensation=wrap(guide_angle(self.env,previous)-guide_angle(self.env,requested))
                unit=np.clip((requested[0]+compensation)/(math.pi/2),-1.+1e-9,1.-1e-9)
                action[0]=np.arctanh(unit)
            return action,0.,0.
        if self.mode == 'constant' and np.any(self.coefficients):
            return (original+residual).astype(original.dtype), 0., 0.
        return original, 0., 0.
