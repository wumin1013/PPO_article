"""Count observed coordinated approach excursions in the closed AR training journal."""
import gzip
import math
from pathlib import Path
import numpy as np
from probe_coordinated_suffix import read
from conditional_exploration import RHO
from study_io import BASE,register,receipt,write_json,sha


def main():
    source=BASE/'ar_conditional_joint_20260916';storage=read(source/'storage.json')
    log=Path(storage['decision_journal']);episodes=Path(storage['training_episode_directory'])
    assert read(log.parent/'resume_pointer.json')['training_complete']
    stage=register('coordinated_exploration_coverage_20260916',
        [Path(__file__),source/'receipt.json',source/'registration.json',source/'storage.json',
         BASE/'conditional_exploration.py',BASE/'train_conditional_exploration.py',
         BASE/'coordinated_suffix_probe_20260916/receipt.json',BASE/'inner_approach_amplitude_probe_20260916/receipt.json',
         BASE/'learning_preparation_v4/paths.json'],
        dict(journal_rows=60000,closed_episode_loads_max=740,model_calls=0,physics_ticks=0,training_updates=0),
        'Measure whether the completed AR run naturally visited sustained positive steering/feed/D offsets demonstrated by the finite probes, restricted to saved single-corner training approaches; skip history reconstruction across parameter updates.')
    binding=dict(path=str(log),sha256=sha(log));paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    specs={}
    for i,p in enumerate(paths):
        ref=np.asarray(p,float)
        if len(ref)!=3:continue
        a,b=ref[1]-ref[0],ref[2]-ref[1];turn=np.cross(a,b)
        if abs(turn)<1e-8:continue
        specs[i]=(ref,a/np.linalg.norm(a),float(np.sign(turn)))
    counters={str(deg):dict(rows=0,longest=0,windows_eight=0) for deg in (2,8)}
    current={k:0 for k in counters};all_rows=eligible=valid_mean_rows=loads=0
    prev_episode=None;mean=None;prev_z=None;elapsed=0;trace=None;trace_path=None;bindings=[]
    skipped_parameter_update_rows=0
    with gzip.open(log,'rt',encoding='utf-8') as stream:
        for line in stream:
            import json
            row=json.loads(line);all_rows+=1;assert row['decision']==all_rows
            if row['episode']!=prev_episode:
                mean=np.asarray(row['conditional_mean'],float);elapsed=0;trace=None
                for key in current:current[key]=0
                trace_path=episodes/f"episode_{row['episode']}.npz"
                if row['path'] in specs and trace_path.exists():
                    with np.load(trace_path) as d:trace=d['rows'].copy()
                    loads+=1;bindings.append(dict(path=str(trace_path),sha256=sha(trace_path)))
            elif (row['decision']-1)%2048==0:
                mean=None
            elif mean is not None:
                mean=np.asarray(row['conditional_mean'],float)-RHO*(prev_z-mean)
            if mean is not None:valid_mean_rows+=1
            else:skipped_parameter_update_rows+=1
            inside=False
            if trace is not None and mean is not None:
                ref,tangent,sign=specs[row['path']]
                position=ref[0] if elapsed==0 else trace[elapsed-1,:2]
                distance=float((position-ref[1])@tangent)
                inside=(-12<=distance<=-2)
                if inside:
                    eligible+=1
                    physical_mean=np.array([math.pi/2*np.tanh(mean[0]),50*(1+np.tanh(mean[1])),6.4+5.6*np.tanh(mean[2])])
                    delta=np.asarray(row['held'])-physical_mean
                    for key in counters:
                        qualifies=sign*delta[0]>=math.radians(float(key)) and delta[1]>=5 and delta[2]>=2
                        current[key]=current[key]+1 if qualifies else 0
                        counters[key]['rows']+=int(qualifies)
                        counters[key]['longest']=max(counters[key]['longest'],current[key])
                        counters[key]['windows_eight']+=int(current[key]>=8)
            if not inside:
                for key in current:current[key]=0
            elapsed+=round(row['elapsed_s']/.001)
            prev_z=np.asarray(row['latent'],float);prev_episode=row['episode']
    assert all_rows==60000 and loads<=740 and sha(log)==binding['sha256']
    result=dict(journal=binding,episode_sources=bindings,counts=counters,eligible_approach_rows=eligible,
        mean_reconstructed_rows=valid_mean_rows,skipped_across_parameter_update_rows=skipped_parameter_update_rows,
        mean_reconstruction='Within constant-parameter blocks: mu_t = conditional_mean_t - rho*(z_prev-mu_prev). Restart at each episode; discard a continuing episode after each PPO parameter update. Float32 logged centers imply small rounding error.',
        boundary='Observed offset counts, not event probability, likelihood of an exact continuous path, independent trials or optimal timing. Restricts three-vertex training geometry and -12 to -2 mm approach; does not claim all routes to better behavior are absent.',
        costs=dict(journal_rows=all_rows,closed_episode_loads=loads,model_calls=0,physics_ticks=0,training_updates=0))
    write_json(stage/'results.json',result);receipt(stage,dict(status='completed_observed_joint_excursion_coverage',**result['costs']))
    print({k:v for k,v in result.items() if k not in ('episode_sources','journal')})


if __name__=='__main__':main()
