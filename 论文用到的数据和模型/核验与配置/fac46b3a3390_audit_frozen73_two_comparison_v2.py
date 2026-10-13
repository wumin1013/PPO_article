"""Select frozen candidate0 explicitly from its historical multi-candidate folder."""
from pathlib import Path

adapter_source=Path(__file__).with_name('audit_current_two_comparison_v2.py')
code=adapter_source.read_text(encoding='utf-8')
old="code = source.read_text(encoding='utf-8')"
addition=old+'''
needle="found=list(directory.glob(f'eval_*_{case}_{key}.json'))"
replacement="found=list(directory.glob(f'eval_0000_{case}_{key}.json')) if directory.resolve()==ROOT/'direct14_dictionary02_resume_round73' else list(directory.glob(f'eval_*_{case}_{key}.json'))"
assert code.count(needle)==1
code=code.replace(needle,replacement)
'''
assert code.count(old)==1
code=code.replace(old,addition)
old='[Path(__file__),source]'
assert code.count(old)==1
code=code.replace(old,'[Path(__file__),source,adapter_source]')
exec(compile(code,str(adapter_source),'exec'))
