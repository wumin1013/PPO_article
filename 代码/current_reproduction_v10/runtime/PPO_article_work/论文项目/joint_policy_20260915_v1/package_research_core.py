"""Create a small additive research core package; leave raw rolling logs on D:."""
import json
import shutil
import zipfile
from pathlib import Path
from study_io import BASE,ROOT,register,receipt,write_json,sha


def main():
    tracker=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/ts_local_revision_v1'
    core=Path('D:/Nutstore/1/基于强化学习的运动规划方法')
    archive_name='PPO_core_research_20260916.zip'
    assert not (core/archive_name).exists()
    files=set(BASE.glob('*.py'))|set(BASE.glob('*.md'))
    names=['current_state_reconciliation_20260916','feed_update_context_20260916','task_length_actor_probe_20260916',
        'storage_lifecycle_binding_20260916','conditional_exploration_validation_20260916',
        'ar_conditional_joint_20260916','ar_conditional_fixed_20260916','conditional_exploration_results_20260916',
        'conditional_exploration_corners_20260916','conditional_credit_geometry_20260916',
        'conditional_results_recorded_20260916','coordinated_suffix_probe_20260916',
        'inner_approach_amplitude_probe_20260916','coordinated_suffix_figures_20260916',
        'coordinated_exploration_coverage_20260916','student_exploration_validation_20260916',
        'student_exploration_physical_probe_20260916','suffix_findings_recorded_20260916']
    for name in names:
        directory=BASE/name
        for p in directory.iterdir():
            if not p.is_file() or p.name.startswith(('partial_','checkpoint_')):continue
            if p.suffix in ('.json','.md','.pdf','.svg','.csv') or p.name in ('final.pt','initial.pt'):files.add(p)
        if name.startswith('ar_conditional_'):files.update((directory/'final_validation').glob('*'))
        if name in ('coordinated_suffix_probe_20260916','inner_approach_amplitude_probe_20260916'):files.update(directory.glob('*.npz'))
    for name in ('progress.py','continuation_observer.py','storage_lifecycle.py','current_progress.json','root_goal_observation.json'):
        files.add(tracker/name)
    for name in [*names,'feed_context_and_conditional_start_20260916']:
        p=tracker/(name+'.json')
        if p.exists():files.add(p)
    current=json.loads((tracker/'current_progress.json').read_text(encoding='utf-8'))
    files.update(ROOT/n for n in current['updated_files'])
    files.update(ROOT/n for n in ('AGENTS.md','PPO_存储与清理规则.md','00_工作目录与存储说明.md',
                                  '论文项目/PPO_article_新对话续接指令_20260916.md'))
    files=sorted(files);assert all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in files)
    stage=register('research_core_sync_20260916',[Path(__file__),BASE/'suffix_findings_recorded_20260916/receipt.json',
        tracker/'current_progress.json'],dict(new_archives=1,new_raw_training_logs=0,model_calls=0,physics_ticks=0),
        'Selective additive core code, reports, final/initial key models, compact results and essential plotted traces; original before-edits backups and bulk rolling journals remain on non-sync D:. No deletion or repository commit.')
    items=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    manifest=dict(files=items,file_count=len(items),raw_bytes=sum(r['bytes'] for r in items),
        base_archives='Existing PPO_core_01.zip through PPO_core_14.zip and original manuscript package are retained.',
        omitted='Rolling journals/resume slots, transient renders, duplicate source snapshots and intermediate checkpoints remain preserved on D:.',
        final_submission=False,goal_complete=False)
    write_json(stage/'manifest.json',manifest)
    archive=stage/archive_name
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for p,item in zip(files,items):
            assert sha(p)==item['sha256'];out.write(p,item['path'])
        out.write(stage/'manifest.json','PPO_core_research_20260916_manifest.json')
    import hashlib
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
        for item in items:assert hashlib.sha256(check.read(item['path'])).hexdigest()==item['sha256']
    shutil.copy2(archive,core/archive_name)
    assert sha(archive)==sha(core/archive_name)
    receipt(stage,dict(status='selective_core_archive_locally_verified',file_count=len(items),
        raw_bytes=manifest['raw_bytes'],archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),
        local_sync_copy=str(core/archive_name),cloud_upload_pending=True,deleted_files=0,goal_complete=False))
    print(dict(archive=str(archive),files=len(items),bytes=archive.stat().st_size,sha256=sha(archive)))


if __name__=='__main__':main()
