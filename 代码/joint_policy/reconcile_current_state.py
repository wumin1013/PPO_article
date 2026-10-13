"""Verify completion receipts and bind remaining frozen analysis inputs."""
import json
from pathlib import Path
from datetime import datetime, timezone
from study_io import BASE, register, receipt, write_json, sha


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    names = ['joint_independent_refinement_v7', 'joint_gsde_refinement_v7',
             'fixed3_independent_refinement_v7', 'fixed3_gsde_refinement_v7']
    names += [f'minimal_{arm}_{mode}_pilot_v1' for arm in ('base','state','reward')
              for mode in ('joint','fixed')]
    names += ['credit_horizon_joint_pilot_v1','credit_horizon_fixed_pilot_v1',
              'wide_preview_independent_v1']
    sources = [Path(__file__), BASE/'study_io.py']
    sources += [BASE/n/f for n in names for f in ('registration.json','receipt.json','selection.json')]
    stage = register('current_state_reconciliation_20260916', sources,
        dict(completed_stages=13, model_loads=0, model_calls=0, training_decisions=0, physics_ticks=0),
        'Reconcile original completed runs and storage metadata before a new mechanism diagnosis; no experiment reruns.')
    rows = []
    for name in names:
        folder = BASE/name
        r, reg = read(folder/'receipt.json'), read(folder/'registration.json')
        assert r['registration_sha256'] == sha(folder/'registration.json'), name
        assert sha(folder/'final.pt') == r['final_sha256'], name
        storage = read(folder/'storage.json')
        pointer_path = Path(storage['rolling_state_directory'])/'resume_pointer.json'
        pointer = read(pointer_path)
        assert pointer['training_complete'] and pointer['phase']=='complete', name
        selected = read(folder/'selection.json')['selected']
        assert selected == r['selected'], name
        rows.append(dict(stage=name, receipt_sha256=sha(folder/'receipt.json'),
            registration_budget=reg['budget'], completed_status=r['status'],
            selected=selected, pointer=pointer, pointer_sha256=sha(pointer_path),
            final_model_sha256=r['final_sha256'], historical_failure_preserved=(folder/'failure.json').exists()))
    analysis_names = ['joint_independent_exploration_learning_v1','fixed3_independent_exploration_learning_v1',
                      'joint_independent_refinement_v7','fixed3_independent_refinement_v7','wide_preview_independent_v1']
    old = {r['stage']:r for r in read(BASE/'saved_action_credit_v1/external_sources.json')}
    buffers = []
    for name in analysis_names:
        root = Path(read(BASE/name/'storage.json')['rolling_state_directory'])
        # The final committed slot has an empty batch; preceding slot retains the final update input.
        slot = 0 if name.endswith('_refinement_v7') else 1
        path = root/f'resume_{slot}.pt'
        digest = sha(path)
        if name in old:
            assert digest == old[name]['sha256']
        buffers.append(dict(stage=name,path=str(path),sha256=digest,bytes=path.stat().st_size))
    audit = Path('D:/PPO_storage_cleanup_20260915')
    cloud = read(audit/'cloud_inventory_checkpoint.json')
    upload = [json.loads(l) for l in (audit/'cloud_core_upload_receipts.jsonl').read_text(encoding='utf-8-sig').splitlines() if l]
    storage_summary = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        inventory_time=cloud['time'], inventory_complete=cloud['complete'], queued=len(cloud['queued']),
        historical_cumulative_bytes=cloud['bytes'], cumulative_bytes_are_not_current_total=True,
        core_package=read(audit/'cloud_core_package_receipt.json'),
        upload_verified_files=len(upload), upload_verified_bytes=sum(r['bytes'] for r in upload),
        upload_in_progress_observed_pid=17012, full_cloud_acceptance_complete=False,
        action='Preserve ongoing uploader; do not rescan or duplicate its upload. No credentials read.')
    write_json(stage/'results.json',dict(stages=rows,external_buffers=buffers,storage=storage_summary,
        application_goal=None,create_goal_called=False,training_processes_observed=[],
        scope='Receipt/registration/model SHA and pointer metadata checks, without deserializing models or reopening sealed test.'))
    write_json(stage/'external_buffers.json',buffers)
    receipt(stage,dict(status='completed_current_state_reconciliation',verified_training_stages=len(rows),
        model_loads=0,model_calls=0,physics_ticks=0,training_decisions=0,cloud_acceptance_complete=False))
    print(json.dumps(dict(completed=len(rows),selected=[dict(stage=r['stage'],**r['selected']) for r in rows],
        frozen_buffers=buffers,cloud_queued=len(cloud['queued']),verified_uploads=len(upload)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
