"""Replace the consumed open-hook family before learned test evaluation.

The old registration, reference and outcomes remain intact. This generator does
not import an executor, model or previous outcome values.
"""
import math
import numpy as np
from common import ROOT,sha,utc,write_json
import json


def main():
    folder=ROOT/'prospective/final_v3';regfile=folder/'registration.json'
    previous=regfile.read_bytes();old=json.loads(previous)
    retired=folder/'registration_v3a_retired.json'
    if retired.exists():raise FileExistsError(retired)
    retired.write_bytes(previous)
    vertices=np.array([[0.,0.],[60.,0.],[24.,47.]],float)
    radii=[6.,5.,8.];pieces=[np.array([[30.,0.]])];primitives=[];bounds=[]
    # Closed triangular contour: unequal corner fillets and three straight links.
    for k in [1,2,0]:
        incoming=vertices[k]-vertices[(k-1)%3];incoming/=np.linalg.norm(incoming)
        outgoing=vertices[(k+1)%3]-vertices[k];outgoing/=np.linalg.norm(outgoing)
        turn=math.atan2(np.cross(incoming,outgoing),incoming@outgoing)
        r=radii[k];offset=r*math.tan(turn/2)
        first=vertices[k]-offset*incoming;last=vertices[k]+offset*outgoing
        center=first+r*np.array([-incoming[1],incoming[0]])
        start=math.atan2(first[1]-center[1],first[0]-center[0]);n=math.ceil(turn*r/.6)
        angles=np.linspace(start,start+turn,n+1)
        samples=center+r*np.c_[np.cos(angles),np.sin(angles)]
        pieces.extend([first[None,:],samples[1:]])
        bounds.append(r*(1-math.cos(turn/n/2)))
        primitives.append(dict(center=center.tolist(),radius_mm=r,start=first.tolist(),end=last.tolist(),sweep_radian=turn,intervals=n))
    pieces.append(np.array([[30.,0.]]));points=np.vstack(pieces)
    file=folder/'final_rounded_triangle.csv'
    np.savetxt(file,points,delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12g')
    bound=max(bounds)+1e-9
    row=dict(id='final_rounded_triangle',file=file.name,sha256=sha(file),group='closed_scalene_triangle_with_unequal_circular_fillets',origin='synthetic',
        motivation='A complete closed line/arc program with unequal radii and turn angles, distinct from the retired open hook topology.',vertices=len(points),
        length_mm=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),original_tolerance_mm=1.25,
        arc_chord_error_bound_mm=bound,evaluation_tolerance_mm=1.25-bound,exact_arc_primitives=primitives,closed=True)
    old['paths']=[row if p['id']=='final_line_arc_chain' else p for p in old['paths']]
    old.update(created_utc=utc(),generator_sha256=sha(__file__),registry_version='v3b',
        representative_figure_paths=['final_dense_fold','final_rounded_triangle'],
        status_at_registration='New rounded-triangle program has never been evaluated. Four unchanged programs retain their original registration; no learned final evaluation has occurred.',
        scope='Polyline command trajectories; rounded_triangle reserves analytic circular chord error from the nominal one-sided tolerance.',
        amendment=dict(prior_registration_file=retired.name,prior_registration_sha256=sha(retired),
            retired_program='final_line_arc_chain',retired_family='three_arc_hook_with_terminal_offset',
            reason='A deterministic-baseline assembly failure on this program was inspected to fix a generic empty-slice bug. The full consumed program family is development-only.',
            old_reference_retained=True,all_observed_runs_retained=True,other_four_reference_bytes_unchanged=True,
            learned_final_models_or_policies_run_before_amendment=0))
    write_json(regfile,old)
    write_json(ROOT/'audit/final_v3_registry_transition.json',dict(created_utc=utc(),new_registry_sha256=sha(regfile),**old['amendment']))
    print('Registered',row['id'],len(points),'bound',bound)


if __name__=='__main__':main()
