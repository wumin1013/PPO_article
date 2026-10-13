"""Two declared PPO-clipping variants, preserving frozen main sources/models."""
import argparse
import importlib.util
import json
import shutil
from common import ROOT,sha,utc,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--clip',type=float,choices=[.1,.3],required=True);a=p.parse_args()
    frozen=json.loads((ROOT/'audit/final_v3b_freeze.json').read_text('utf-8'))
    source=ROOT/'src/unified_training_v3.py'
    if sha(source)!=frozen['sha256']['src/unified_training_v3.py']:raise ValueError('Main trainer changed')
    variants=ROOT/'results/sensitivity_v3';variants.mkdir(exist_ok=True)
    derived=variants/('registered_trainer_clip_'+str(a.clip)+'.py')
    if derived.exists():raise FileExistsError('Sensitivity variant already registered')
    text=source.read_text('utf-8')
    replacements={
        "OUT=ROOT/'results'/'unified_v3'":f"CLIP={a.clip!r}\nOUT=ROOT/'results'/'sensitivity_v3'/'clip_{a.clip}'",
        'ppo_clip=.2':'ppo_clip=CLIP',
        'ratio.clamp(.8,1.2)':'ratio.clamp(1.-CLIP,1.+CLIP)',
        '((ratio-1).abs()>.2)':'((ratio-1).abs()>CLIP)',
    }
    for before,after in replacements.items():
        if text.count(before)!=1:raise ValueError('Unexpected trainer patch location: '+before)
        text=text.replace(before,after)
    derived.write_text(text,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('sensitivity_registered',derived);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    registration=module.register()
    registration.update(scope='DEVELOPMENT sensitivity only; never evaluates final-test programs or changes primary model selection',
        seeds=[101],arms=['adaptive'],parent_main_freeze_sha256=sha(ROOT/'audit/final_v3b_freeze.json'),
        single_changed_hyperparameter=dict(name='PPO clip epsilon',main=.2,variant=a.clip),
        generated_source_sha256=sha(derived),generation_replacements=replacements,
        teacher_data='Byte-identical main shared records; no additional teacher selection or normalization data',
        final_test_executions_permitted=0)
    registration['paired_training']['selection']='Final 8192 checkpoint; initial/intermediate/final validation retained. No selection among variants.'
    write_json(module.OUT/'registration.json',registration)
    for name in ['shared_teacher_records.npz','teacher_selection.json']:
        shutil.copy2(ROOT/'results/unified_v3'/name,module.OUT/name)
    write_json(module.OUT/'sensitivity_registration.json',dict(created_utc=utc(),seed=101,arm='adaptive',clip=a.clip,
        training_decisions=8192,supervised_steps=2000,teacher_sha256=sha(module.OUT/'shared_teacher_records.npz'),
        source_sha256=sha(derived),parent_main_model='results/unified_v3/paired/seed_101/adaptive/final.pt',
        no_final_test_evaluation=True,no_model_selection_change=True))
    module.train('adaptive',101)


if __name__=='__main__':main()
