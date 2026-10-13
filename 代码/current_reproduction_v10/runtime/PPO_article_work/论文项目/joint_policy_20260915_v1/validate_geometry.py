"""Bounded new synthetic geometry validation, with pre-run source registration."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import traceback
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent.parent
STAGE=BASE/'geometry_validation_v1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name,value):
    (STAGE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    STAGE.mkdir(exist_ok=True)
    if (STAGE/'registration.json').exists():
        raise RuntimeError('This finite stage was already started; inspect its receipt rather than rerun')
    references=sorted((ROOT/'论文项目/revision_20260913_preview_efficiency_v2/geometry_design_v1/references').glob('*.csv'))
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),BASE/'geometry_spec.py',*references]}
    save('registration.json',{'status':'registered_before_checks','source_sha256':sources,
        'created_utc':datetime.now(timezone.utc).isoformat(),'maximum_synthetic_checks':32,
        'maximum_reference_inventories':3,'environment_steps':0,'training_updates':0,
        'scope':'Independent new corridor/gate synthetic checks and read-only reference structure inventory'})
    results=[]
    try:
        from geometry_spec import Corridor,GateTracker,audit_commands,capsule_intervals
        def check(name,condition,detail=None):
            if len(results)>=32:
                raise RuntimeError('Synthetic check budget exhausted')
            result={'name':name,'passed':bool(condition)}
            if detail is not None: result['detail']=detail
            results.append(result)
            if not condition: raise AssertionError(name)

        straight=Corridor([[-5.,0.],[0.,0.],[5.,0.]])
        check('straight_inside',straight.segment_inside(np.array([-5.,1.]),np.array([5.,1.])))
        check('straight_outside',not straight.segment_inside(np.array([-5.,1.251]),np.array([5.,1.251])))
        check('capsule_boundary',straight.segment_inside(np.array([-5.,1.25]),np.array([5.,1.25])))
        check('finite_capsule_end_outside',not straight.segment_inside(np.array([5.,0.]),np.array([6.251,0.])))
        check('stationary_inside',straight.segment_inside(np.array([0.,0.]),np.array([0.,0.])))
        c90=Corridor([[-10.,0.],[0.,0.],[0.,10.]])
        check('endpoints_inside_middle_outside',not c90.segment_inside(np.array([-5.,0.]),np.array([0.,5.])))
        check('corner_diagonal_inside',c90.segment_inside(np.array([-2.,0.]),np.array([0.,2.])))
        result,_=audit_commands(straight.reference,[[-5.,0.],[5.,0.]])
        check('forward_gate',result['geometry_complete'])
        t=GateTracker(straight,[-1.,0.]);t.update([0.,0.]);t.update([0.,.1])
        check('touch_and_wait_not_cross',t.frontier==0)
        t.update([1.,0.]);check('contact_then_forward',t.frontier==1)
        t.update([-1.,0.]);t.update([1.,0.]);check('repeat_gate_no_new_progress',t.frontier==1)
        reverse=GateTracker(straight,[1.,0.]);reverse.update([-1.,0.])
        check('backward_cross_not_counted',reverse.frontier==0)
        t=GateTracker(straight,[-1.,0.]);t.update([0.,0.]);t.update([-1.,0.]);t.update([0.,0.])
        check('touch_reverse_not_counted',t.frontier==0)
        many=Corridor([[-4.,0.],[-2.,0.],[0.,0.],[2.,0.],[4.,0.]])
        t=GateTracker(many,[-4.,0.]);t.update([4.,0.])
        check('one_command_multiple_ordered_gates',t.frontier==3 and [e['fraction'] for e in t.events]==[.25,.5,.75])
        result,_=audit_commands([[-5.,0.],[5.,0.]],[[5.,0.],[5.,0.]])
        check('start_position_required',not result['geometry_complete'])

        phi=np.deg2rad(130.)
        r=2.;distance=r*np.tan(phi/2)
        ref=np.array([[-20.,0.],[0.,0.],[20*np.cos(phi),20*np.sin(phi)]])
        center=np.array([-distance,r])
        angles=np.linspace(-np.pi/2,-np.pi/2+phi,27)
        arc=center+r*np.c_[np.cos(angles),np.sin(angles)]
        commands=np.vstack([ref[0],arc,ref[-1]])
        result,events=audit_commands(ref,commands)
        vertex_distance=float(np.linalg.norm(arc,axis=1).min())
        check('130deg_inner_cut_far_from_vertex',result['geometry_complete'] and vertex_distance>1.245,
              {'vertex_distance_mm':vertex_distance,'gate_crossing':events})
        t=GateTracker(Corridor(ref),ref[0]);t.update(ref[-1])
        check('shortcut_across_whole_v_rejected',not t.inside)

        square=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.],[0.,0.]])
        result,_=audit_commands(square,[[0.,0.],[0.,0.]],closed=True)
        check('closed_stationary_not_complete',not result['geometry_complete'])
        result,_=audit_commands(square,square,closed=True)
        check('closed_one_lap',result['geometry_complete'])
        result,_=audit_commands(square,square[::-1],closed=True)
        check('closed_reverse_lap_not_complete',not result['geometry_complete'])
        result,_=audit_commands(square,[[0.,0.],[10.,0.],[0.,0.]],closed=True)
        check('closed_missing_gates_not_complete',not result['geometry_complete'])
        reversal=Corridor([[0.,0.],[3.,0.],[0.,0.]])
        check('180deg_ambiguity_explicit',reversal.ambiguities[0]['kind']=='reversal_180')
        try:
            GateTracker(reversal,[0.,0.]);rejected=False
        except ValueError: rejected=True
        check('180deg_cannot_silently_pass',rejected)
        tiny=Corridor([[0.,0.],[.5,0.],[.5,.5],[0.,.5],[0.,0.]],closed=True)
        check('tiny_loop_gate_conflict_detected',len(tiny.adjacent_gate_conflicts())>0)
        rotation=np.array([[0.,-1.],[1.,0.]])
        result,_=audit_commands(ref@rotation.T+np.array([359.53,-18.]),commands@rotation.T+np.array([359.53,-18.]))
        check('rotated_translated_cut',result['geometry_complete'])

        inventory=[]
        for p in references:
            ref=np.loadtxt(p,delimiter=',',skiprows=1)
            closed=np.linalg.norm(ref[0]-ref[-1])<1e-8
            corridor=Corridor(ref,closed=closed)
            turns=np.rad2deg(np.arccos(np.clip(np.sum(corridor.tangents[:-1]*corridor.tangents[1:],axis=1),-1.,1.)))
            conflicts=corridor.adjacent_gate_conflicts()
            inventory.append({'path':p.relative_to(ROOT).as_posix(),'vertices':len(ref),'closed':bool(closed),
                'length_mm':float(corridor.arcs[-1]),'min_segment_mm':float(corridor.lengths.min()),
                'max_turn_deg':float(turns.max()),'gates':len(corridor.gates),
                'undefined_turns':corridor.ambiguities,'adjacent_gate_conflicts':conflicts})
        save('results.json',{'checks':results,'reference_inventory':inventory,'new_geometry_definition_only':True})
        save('receipt.json',{'status':'passed_synthetic_checks_reference_semantics_inventory_only',
             'checks':len(results),'passed':sum(x['passed'] for x in results),
             'reference_inventory_count':len(inventory),'results_sha256':sha(STAGE/'results.json'),
             'registration_sha256':sha(STAGE/'registration.json'),'training_updates':0,'environment_steps':0,
             'performance_or_full_trajectory_certificate':False})
        print(json.dumps({'passed':len(results),'reference_inventory':inventory},ensure_ascii=False))
    except Exception as error:
        save('failure.json',{'error':str(error),'traceback':traceback.format_exc(),'completed_checks':results})
        raise
