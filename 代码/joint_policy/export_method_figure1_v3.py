"""Retain the original interpolation-point motif in the first method figure."""
import importlib.util
import shutil
from pathlib import Path
from study_io import BASE,ROOT,register,receipt,sha
from draw_method_figures import figure_one


if __name__=='__main__':
    converter_path=ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    stage=register('method_figure1_v4',[__file__,BASE/'draw_method_figures.py',
        BASE/'method_figure_contract_v1.md',BASE/'method_figures_v2/receipt.json',converter_path],
        dict(figures=1,conversions=1,models=0,physics_ticks=0),
        'Show discrete interpolation points in the retained serial-planning diagram; other two checked method figures are reused.')
    figure_one(stage)
    spec=importlib.util.spec_from_file_location('single_method_svg_converter',converter_path)
    converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    output=Path('D:/PPO_article_joint_runs/method_figure1_v4')
    converter.convert(stage/'Figure_1.svg',output)
    shutil.copy2(output/'Figure_1.pdf',stage/'Figure_1.pdf')
    receipt(stage,dict(status='interpolation_points_figure_converted_pending_visual_QA',
        svg_sha256=sha(stage/'Figure_1.svg'),pdf_sha256=sha(stage/'Figure_1.pdf'),
        preview=str(output/'preview.png'),models=0,physics_ticks=0))
