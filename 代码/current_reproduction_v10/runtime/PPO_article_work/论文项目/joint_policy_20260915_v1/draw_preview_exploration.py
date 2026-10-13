"""Render the common 60k corner comparisons with the accepted geometry layout."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from study_io import BASE, ROOT, register, receipt, write_json, sha
from summarize_conditioned_training import distance_to_commands


def main():
    analysis = BASE/'preview_exploration_60000_analysis_v1'
    specifications = [
        ('joint_independent_refinement_v7', 'validation_60000', 'Narrow preview', '#444444', '-'),
        ('wide_preview_independent_v1', 'final_validation', 'Wide preview', '#245784', '--'),
        ('fixed3_independent_refinement_v7', 'validation_60000', 'Fixed 3 mm', '#cd7825', '-.')]
    converter_path = ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources = [Path(__file__), BASE/'v7_checkpoint_figure_contract.md',
        BASE/'preview_exploration_design_v1.md', BASE/'minimal_factor_corner_figures_v2/visual_qa.json',
        analysis/'receipt.json', analysis/'results.json', converter_path,
        BASE/'summarize_conditioned_training.py']
    sources += [BASE/name/tag/item for name, tag, *_ in specifications
        for item in ['summary.json', *[f'{i}.npz' for i in range(4)]]]
    if not all(p.exists() for p in sources):
        raise RuntimeError('Wait for the registered endpoint analysis')
    stage = register('preview_exploration_corners_v1', sources,
        dict(saved_trace_files=12, svg_conversions=1, model_calls=0, physics_ticks=0),
        'Compare true corner shapes, local speed and complete time for the single-factor exploration endpoint; no final superiority claim.')
    (stage/'figure_contract.md').write_text(
        '# 前瞻探索四角图\n\n科学问题：扩大D探索是否同时改善形状与效率，以实际数据判断。'
        '四个面板覆盖全部孤立开发角50/95/125/150度，三方法为同60k。'
        '沿用用户已确认的等比例配对轨迹图型及已验收简化四角版式，优先于额外外部图型搜索；未新检索期刊范例。'
        'Python定量网格，7.1×7.5英寸，颜色与线型双编码；1.25mm欧氏允差域、共同局部视窗。'
        '输出可编辑SVG、实际SVG转PDF、PNG和CSV。每种设计仅一条开发训练谱系，无独立误差带。\n', encoding='utf-8')
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8,
        'svg.fonttype': 'none', 'axes.linewidth': .65})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 7.5))
    traces, summaries = {}, {}
    for name, tag, *_ in specifications:
        summaries[name] = json.loads((BASE/name/tag/'summary.json').read_text(encoding='utf-8'))['results']
        for index in range(4):
            with np.load(BASE/name/tag/f'{index}.npz') as data:
                traces[name, index] = {k: data[k].copy() for k in ('reference', 'rows')}
    records = []
    for index, (ax, angle) in enumerate(zip(axes.ravel(), (50, 95, 125, 150))):
        reference = traces[specifications[0][0], index]['reference']
        localref = reference-reference[1]
        region = LineString(localref).buffer(1.25, resolution=192)
        x, y = region.exterior.xy
        ax.fill(x, y, color='#edf4fa', zorder=0)
        ax.plot(x, y, color='#8db2d2', lw=.65, ls='--')
        ax.plot(localref[:, 0], localref[:, 1], color='#888888', ls=':', lw=.9, label='Reference')
        ax.plot(0, 0, 'o', ms=3, color='#333333')
        for j, (name, tag, label, color, style) in enumerate(specifications):
            data = traces[name, index]
            assert np.array_equal(data['reference'], reference)
            xy = data['rows'][:, :2]-reference[1]
            speed = np.linalg.norm(data['rows'][:, 2:4], axis=1)
            local = np.linalg.norm(xy, axis=1) < 10.
            result = next(r for r in summaries[name] if r['path'] == index)
            row = dict(stage=name, label=label, turn_deg=angle, success=result['success'],
                full_time_s=result['completion_time_s'],
                vertex_distance_mm=distance_to_commands(np.zeros(2), np.vstack([localref[0], xy])),
                local_median_speed_mm_s=float(np.median(speed[local])) if local.any() else None)
            records.append(row)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.15, ls=style, label=label)
            speedlabel = 'n/a' if row['local_median_speed_mm_s'] is None else f"{row['local_median_speed_mm_s']:.1f}"
            timelabel = 'prefix' if row['full_time_s'] is None else f"{row['full_time_s']:.3f} s"
            ax.text(.02, -.36-.10*j,
                f"{label}: {row['vertex_distance_mm']:.2f} / {speedlabel} / {timelabel}",
                transform=ax.transAxes, color=color, fontsize=7.2)
        ax.set(xlim=(-10, 10), ylim=(-3, 12), xlabel='x from vertex (mm)', ylabel='y from vertex (mm)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'({chr(97+index)}) {angle}° turn', loc='left', fontsize=9)
        ax.grid(color='#e7eaed', lw=.35)
        ax.text(.02, -.26, 'Distance (mm) / local speed (mm/s) / full time', transform=ax.transAxes, fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .97), ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Preview exploration at 60,000 decisions', fontsize=10, y=.997)
    fig.subplots_adjust(left=.09, right=.985, top=.89, bottom=.17, hspace=.90, wspace=.28)
    svg = stage/'preview_corners.svg'
    fig.savefig(svg)
    fig.savefig(stage/'preview_corners.png', dpi=180)
    plt.close(fig)
    spec = importlib.util.spec_from_file_location('preview_corner_converter', converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    external = Path('D:/PPO_article_joint_runs')/stage.name
    converter.convert(svg, external)
    shutil.copy2(external/'preview_corners.pdf', stage/'preview_corners.pdf')
    with (stage/'source_data.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(stage/'figure_manifest.json', dict(pdf_sha256=sha(stage/'preview_corners.pdf'),
        svg_sha256=sha(svg), preview=str(external/'preview.png'), visual_qa_pending=True,
        model_calls=0, physics_ticks=0, goal_complete=False))
    receipt(stage, dict(status='preview_corner_figure_pending_visual_review', saved_trace_files=12,
        model_calls=0, physics_ticks=0, training_decisions=0))
    print(str(external/'preview.png'))


if __name__ == '__main__':
    main()
