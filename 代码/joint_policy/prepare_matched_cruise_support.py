"""Match update geometry and feed preservation, varying only cruise loss support."""
import time
import numpy as np
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    raw=BASE/'canonical_gate_band_segment_cruise_geometry_20260918'
    audit=BASE/'canonical_gate_band_segment_cruise_support_20260918'
    metadata=read(raw/'results.json');states=read(audit/'results.json')
    assert metadata['parent_sha256']==states['parent_sha256']
    with np.load(raw/'geometry.npz') as z:arrays={k:z[k].copy() for k in z.files}
    pure_paths={r['path'] for r in states['records'] if r['category']=='straight'}
    indices=arrays['cruise_state_indices'];paths=arrays['state_path'][indices]
    mask=np.array([int(p) in pure_paths for p in paths]);assert mask.sum()==599
    preserve=arrays['cruise_features'][mask].copy()
    products=[]
    for arm in ('pure_straights','segment_interiors'):
        stage=register('canonical_gate_band_matched_'+arm+'_geometry_20260918',
            [__file__,raw/'geometry.npz',raw/'results.json',raw/'receipt.json',audit/'results.json',
             BASE/'cruise_update_geometry.py',BASE/'solve_transition_shape_step.py'],
            dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=45,preservation_states=599,
                 target_support=arm,matched_source_geometry=True),
            'Form matched cruise-objective support arms from one exact current saved-state '
            'collection. Preserve all metric, action-bound and support arrays byte-for-byte '
            'and keep the SAME599 pure-straight feed-preservation states in BOTH arms. '
            'Only the99.5mm/s loss target-state membership changes: pure straight routes '
            'versus their union with576 conservative curved-route segment interiors. '
            'Use equal weight per target-bearing path and equal state weight within each '
            'path. This prevents accidentally changing both cruise objective and feasible '
            'set when comparing training support. Raw state reconstruction is reused; '
            'no second simulation or actor replay, policy update or success claim.')
        start=time.perf_counter();out={k:v.copy() for k,v in arrays.items()}
        chosen=mask if arm=='pure_straights' else np.ones(len(mask),dtype=bool)
        for key in ('cruise_features','cruise_latent','cruise_state_indices'):
            out[key]=out[key][chosen]
        selected_paths=paths[chosen];unique=np.unique(selected_paths)
        out['cruise_weight']=np.array([1./(len(unique)*np.sum(selected_paths==p)) for p in selected_paths])
        out['cruise_preservation_features']=preserve.copy()
        assert abs(out['cruise_weight'].sum()-1.)<1e-12
        np.savez_compressed(stage/'geometry.npz',**out)
        write_json(stage/'results.json',dict(metadata,cruise_support=arm,cruise_states=int(chosen.sum()),
            cruise_target_paths=len(unique),preservation_states=599,source_geometry=raw.name))
        receipt(stage,dict(status='prepared_matched_physical_and_latent_update_geometries',
            states=len(out['state_path']),cruise_states=int(chosen.sum()),preservation_states=599,
            wall_s=time.perf_counter()-start,new_physics_ticks=0,goal_complete=False))
        products.append(out)
        print('MATCHED_CRUISE_SUPPORT',arm,int(chosen.sum()),'paths',len(unique),flush=True)
    varied={'cruise_features','cruise_latent','cruise_state_indices','cruise_weight'}
    assert set(products[0])==set(products[1])
    assert all(np.array_equal(products[0][key],products[1][key]) for key in products[0] if key not in varied)


if __name__=='__main__':main()
