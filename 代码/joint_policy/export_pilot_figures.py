"""Actual SVG-to-PDF conversion of the two saved pilot figures."""
import importlib.util
from pathlib import Path
from study_io import BASE,ROOT,register,write_json,receipt,sha

if __name__=='__main__':
    converter_path = ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources = [__file__,converter_path,BASE/'pilot_figure_contract.md']+[BASE/'pilot_analysis_v1'/f'corner_{angle}.svg' for angle in [130,155]]
    stage = register('pilot_pdf_v1',sources,dict(figures=2,conversions=2,models=0,physics_ticks=0),
                     'Convert actual saved SVGs; full-page visual QA remains separate.')
    spec = importlib.util.spec_from_file_location('pilot_svg_converter',converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    output_root = Path('D:/PPO_article_joint_runs/20260915_pilot_pdf_v1')
    output_root.mkdir(parents=True,exist_ok=False)
    outputs = []
    for angle in [130,155]:
        source = BASE/'pilot_analysis_v1'/f'corner_{angle}.svg'
        output = output_root/f'corner_{angle}'
        converter.convert(source,output)
        pdf = output/f'corner_{angle}.pdf'
        outputs.append(dict(pdf=str(pdf),sha256=sha(pdf),preview=str(output/'preview.png')))
    receipt(stage,dict(status='two_saved_svg_pdfs_converted_pending_visual_QA',outputs=outputs,
                        original_svg_preserved=True,model_calls=0,physics_ticks=0))
