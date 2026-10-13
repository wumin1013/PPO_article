"""Check the prepared source and both storage locations before one launch."""
import json
import runpy
import sys
from pathlib import Path
from joint_runtime import load_source
from study_io import BASE, ROOT, sha, write_json


if __name__ == '__main__':
    preparation=BASE/'preview_exploration_preparation_v1'
    binding=json.loads((preparation/'source_binding.json').read_text(encoding='utf-8'))
    target=Path(binding['path'])
    assert sha(target)==binding['sha256']
    assert sha(BASE/'train_optimizer_refinement_v7.py')==binding['original_sha256']
    name='wide_preview_independent_v1'
    if (Path('D:/PPO_article_joint_runs/rolling_states')/name).exists():
        raise FileExistsError('External state already exists; do not relaunch')
    guard=load_source('preview_stage_guard',ROOT/'论文项目/revision_20260910_continued/root_cause_iteration_v1/experiment_preflight_v1.py')
    checked=guard.check_new_stage(BASE/name,
        ['registration.json','receipt.json','failure.json','checkpoint_60000.pt','final.pt','final.tmp'],
        {target.relative_to(ROOT).as_posix():binding['sha256']})
    if (preparation/'launch_preflight.json').exists():
        raise FileExistsError('Launch already checked; inspect its actual status')
    write_json(preparation/'launch_preflight.json',checked)
    sys.argv=[str(target),'--stage',name,'--mode','joint','--sampler','independent',
        '--initial-reuse','joint_independent_refinement_v7']
    runpy.run_path(str(target),run_name='__main__')
