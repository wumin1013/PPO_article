"""v5_2: six-feedback prediction plus a finite stopping extension.

v5_1 remains frozen. This version retains its candidate set, action ranking,
raw-action fallback, true execution and isolation contract. The only new
acceptance predicate is endpoint feasibility for the existing cubic stop.
"""
import json
import pickle
import time
import numpy as np
from common import write_json
from physical_teacher_v5 import BranchArchive
from braking_reachability_v5 import stopping_extension
from predictive_action_candidate_v5 import (PredictiveActionCandidate, snapshot,
    state_bytes, recorder_lists, new_environment, prediction, first_action,
    change_cost, OFFSETS, FACTORS)

VERSION='predictive_stop_candidate_v5_2'


class PredictiveStopCandidate(PredictiveActionCandidate):
    def __init__(self,model,output,*,deadline=None):
        super().__init__(model,output,deadline=deadline)
        self.stop_output=self.output/'stopping_extensions'
        self.stop_output.mkdir()
        self.stop_archive=BranchArchive(self.stop_output)
        self.stop_counts=dict(assessments=0,feasible=0,failed=0,command_intervals=0)
        self.stop_cost=dict(wall_s=0.,cpu_s=0.)

    def action_for(self,env):
        # This method is explicitly versioned instead of refactoring the
        # byte-frozen v5_1 policy that is already bound to retained evidence.
        started,cpu=time.perf_counter(),time.process_time()
        state=snapshot(env);before=pickle.dumps(state,protocol=5)
        recorder=env.tracker.update;list_sizes=recorder_lists(recorder)
        record=dict(decision_id=self.decisions,actual_start_tick=env.ticks,
            actual_start_progress_mm=float(env.progress()[0]),max_ticks=env.max_ticks,
            fixed_d=env.fixed_d,raw_model_action=None,submitted_action=None,
            source='aborted',selected_branch_id=None,branch_ids=[],error=None)
        pickle.dump(state,self.state_log,protocol=5)
        try:
            raw=self.teacher.mean(env.observation()).numpy()
            record['raw_model_action']=raw.tolist()
            if self.clone is None:
                init=time.perf_counter();self.clone=new_environment(env.corridor.reference,state)
                self.cost['clone_initialization_wall_s']+=time.perf_counter()-init
                assert self.clone.env is not env.env and self.clone.tracker is not env.tracker
                assert self.clone.env.step is not env.env.step
            def evaluate(action,kind,order,offset=None,factor=None):
                rows,positions,outcome=prediction(self.clone,state,action,self.teacher,self.deadline)
                outcome.update(branch_id=self.branches,decision_id=self.decisions,kind=kind,
                    candidate_order=order,first_action=action.tolist(),steering_offset_rad=offset,
                    feed_factor=factor,action_change_cost=change_cost(raw,action),
                    six_decision_horizon_feasible=outcome['feasible'],stopping_check='not_attempted',
                    stopping_extension=None)
                if outcome['feasible']:
                    if outcome['terminal_absorbing_rest']:
                        outcome['stopping_check']='complete_terminal'
                    else:
                        sr,sp,stop=stopping_extension(self.clone)
                        stop.update(branch_id=self.branches,decision_id=self.decisions)
                        self.stop_archive.append(stop,sr,sp)
                        outcome.update(stopping_check='assessed',stopping_extension=stop.copy(),
                                       feasible=bool(stop['feasible']))
                        self.stop_counts['assessments']+=1
                        self.stop_counts['feasible']+=int(stop['feasible'])
                        self.stop_counts['failed']+=int(not stop['feasible'])
                        self.stop_counts['command_intervals']+=len(sr)
                        self.stop_cost['wall_s']+=stop['assessment_wall_s']
                        self.stop_cost['cpu_s']+=stop['assessment_cpu_s']
                self.archive.append(outcome,rows,positions)
                record['branch_ids'].append(self.branches);self.branches+=1
                self.counts['branch_model_calls']+=outcome['model_calls']
                self.counts['branch_ticks']+=len(rows)
                self.counts['failed_branches']+=int(not outcome['feasible'])
                self.cost['branch_wall_s']+=outcome['branch_wall_s']
                self.cost['branch_cpu_s']+=outcome['branch_cpu_s']
                if outcome['error'] and outcome['error'].startswith('TimeoutError:'):
                    raise TimeoutError('Registered complete-run wall budget during prediction')
                if self.deadline is not None and time.perf_counter()>=self.deadline:
                    raise TimeoutError('Registered complete-run wall budget after stopping assessment')
                return action,outcome
            _,base=evaluate(raw,'original_model_prediction',-1)
            if base['feasible']:
                submitted,selected,source=raw.copy(),base,'raw_model'
            else:
                valid=[]
                for offset in OFFSETS:
                    for factor in FACTORS:
                        order=OFFSETS.index(offset)*len(FACTORS)+FACTORS.index(factor)
                        chosen,outcome=evaluate(first_action(raw,offset,factor),
                            'candidate_first_action',order,offset,factor)
                        if outcome['feasible']:valid.append((chosen,outcome))
                if valid:
                    submitted,selected=min(valid,key=lambda pair:(-pair[1]['final_ordered_progress_mm'],
                        pair[1]['action_change_cost'],pair[1]['candidate_order']))
                    source='predictive_correction'
                else:
                    submitted,selected,source=raw.copy(),base,'no_feasible_candidate'
            record.update(submitted_action=submitted.tolist(),source=source,
                          selected_branch_id=selected['branch_id'])
            self.counts[source]+=1
            return submitted
        except Exception as exc:
            record['error']=type(exc).__name__+': '+str(exc)
            self.counts['aborted']+=1
            raise
        finally:
            unchanged=(before==state_bytes(env) and recorder is env.tracker.update and
                       list_sizes==recorder_lists(env.tracker.update))
            record.update(real_state_and_recorder_unchanged=unchanged,recorder_list_sizes=list_sizes,
                          wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu)
            self.cost['total_action_wall_s']+=record['wall_s']
            self.cost['total_action_cpu_s']+=record['cpu_s']
            self.counts['isolation_checks']+=int(unchanged)
            self.decision_log.write(json.dumps(record,allow_nan=False)+'\n')
            self.decision_log.flush();self.state_log.flush();self.decisions+=1
            if not unchanged:raise AssertionError('Predictive clone mutated actual state or recorder')

    def close(self):
        self.stop_archive.close()
        result=super().close()
        result.update(version=VERSION,stopping_counts=self.stop_counts,stopping_cost=self.stop_cost,
            interpretation='New development-only stopping-extension predictive candidate; no recursive safety guarantee because raw fallback is retained when the finite candidate set has no solution')
        return result
