"""Create the shared frozen-controller paper from the verified formulation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'shared_manuscript/frozen_main_v6'


def main():
    source = (ROOT/'shared_manuscript/manuscript.tex').read_text('utf-8')
    start = source.index('\\begin{abstract}')
    end = source.index('\\end{abstract}', start)
    source = source[:start] + '\\begin{abstract}\n\\input{abstract.tex}\n' + source[end:]
    source = source.replace('\\begin{document}', '\\input{numbers.tex}\n\\input{generated_frozen_macros.tex}\n\\begin{document}', 1)
    source = source.replace('The experiments compare fixed and adaptive preview with the same observations, constraint module and learning budget.',
        'The experiments include a separately trained fixed-preview neural controller and deterministic geometric and temporal planners.')
    source = source.replace('Third, independently trained controllers are evaluated on complete, reserved program groups under common geometric and kinematic rules, with local motion explaining their cycle times.',
        'Third, complete command trajectories relate cycle time to corner motion and sustained feedrate under common geometric and kinematic checks. Two development programs provide the main comparisons, and five additional synthetic programs evaluate transfer of the frozen controller.')
    source = source.replace('subject to the ordered-gate safeguard below', 'at the current reference progress')
    guard_start = source.index('The guide-target rule checks the next gate channel')
    guard_end = source.index('The simulator evaluates geometry', guard_start)
    source = source[:guard_start] + source[guard_end:]
    source = source.replace('After all gates are passed and remaining reference distance is at most 12 mm, a bounded search tests nonreversing cubic braking trajectories against the vector limits and a 1 mm endpoint-distance condition.',
        'When the remaining reference distance is at most 12 mm, a bounded search tests nonreversing cubic braking trajectories against the vector limits and a 1 mm endpoint-distance condition. Before execution, the complete candidate brake is checked for traversal of every remaining gate and feasibility of every command interval.')
    start = source.index('\\subsection{Reward and policy learning}')
    end = source.index('\\section{Experiments and Results}', start)
    source = source[:start] + '\\input{method_learning.tex}\n\n' + source[end:]
    source = source.replace('\\input{experiments_v3.tex}', '\\input{experiments.tex}')
    source = source.replace('\\input{conclusion_v3.tex}', '\\input{conclusion.tex}')
    (OUT/'manuscript.tex').write_text(source, encoding='utf-8')


if __name__ == '__main__':
    main()
