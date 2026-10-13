"""Assemble comparator descriptions without altering their source evidence."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'shared_manuscript/frozen_main_v6'
text=(ROOT/'shared_manuscript/baseline_methods_v3.tex').read_text('utf-8')
text=text.replace('\\subsection{Deterministic comparison methods}', '\\subsection{Comparison methods}')
text=text.replace('the final programs use their declared polygonal representation',
                  'the dense program uses its declared polygonal representation')
text += r'''
COPP supplies a second feed-scheduling formulation on the same smoothed geometry. The official version 0.2.3 implementation of TOPP3-SOCP is used with planar vector velocity, acceleration and jerk constraints\citep{Wang2026COPPSoftware}. The path grid uses 0.5 mm spacing on curves and near straight-section endpoints, with 5 mm spacing in the interior of long straight sections. Three sequential convex iterations are executed, and the third supplies the scheduled profile. A common time dilation accounts for discretized command kinematics. Its additional 0.1\% reserve is selected on the two validation programs. If command sampling still exceeds a vector limit, the fixed projection rule increases the dilation using the largest speed, square-root acceleration or cube-root jerk ratio across both resolutions, with at most two further projections. Every candidate remains subject to the complete geometric and terminal-state checks. This comparator combines the published open-source scheduler with the common geometry and command evaluator.

The fixed-preview neural comparator uses a separately trained $14$--$64$--$64$--$2$ policy with 192 local features. Its two action outputs request steering and feedrate, with guide distance held at 3 mm. Both neural controllers use the same prescribed vector limits, reference geometry and complete-task criteria. The fixed-preview controller is evaluated with its trained parameters and associated execution configuration.

\input{nnc_method.tex}
'''
(OUT/'baseline_methods.tex').write_text(text, encoding='utf-8')
bib=OUT/'references.bib'
contents=bib.read_text('utf-8')
if '@misc{Wang2026COPPSoftware,' not in contents:
    contents+=r'''
@misc{Wang2026COPPSoftware,
  author = {Wang, Yunan and He, Suqin and Lin, Shize and Hu, Chuxiong},
  title = {{COPP}: An Open-Source Ultra-Fast Library for Convex-Objective Path Parameterization},
  year = {2026},
  howpublished = {Software, version 0.2.3},
  url = {https://github.com/TOPP-THU/copp},
  note = {Commit a5fd651b9a1e9d91a0f03ff785057cd74712d9e2}
}
'''
    bib.write_text(contents, encoding='utf-8')
