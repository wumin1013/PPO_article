"""Apply the author's table placement request, preserving source history."""
import shutil
from datetime import datetime, timezone
from build_lowdim_original_manuscript import ROOT, WORK


def main():
    appendix_path, experiments_path = WORK/'appendix.tex', WORK/'experiments.tex'
    appendix = appendix_path.read_text(encoding='utf-8')
    experiments = experiments_path.read_text(encoding='utf-8')
    marker = r'\section{Additional Experimental Checks}'
    assert marker in appendix and r'\label{tab:window_sensitivity}' not in experiments
    history = ROOT/'lowdim_manuscript_history'/('tables_to_body_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    history.mkdir(parents=True)
    for path in (appendix_path, experiments_path):
        shutil.copy2(path, history/path.name)
    retained, extra = appendix.split(marker, 1)
    content = extra.split(r'\setcounter{table}{0}', 1)[1].strip()
    first, remainder = content.split(r'\begin{table}[H]', 1)
    provenance, snapshots = (r'\begin{table}[H]'+remainder).split(r'\tabref{tab:refinement_snapshots}', 1)
    snapshots = r'\tabref{tab:refinement_snapshots}'+snapshots
    first = first.replace(r'\begin{table}[htbp]', r'\begin{table}[H]')
    # This is a parameter-selection step, distinct from PPO and offline fitting.
    retained = retained.replace('Offline refinement fits feasible action targets with anchors from complete trajectories.',
        'Offline parameter selection first adds 0.04 to the feed-output bias, corresponding to a 2 mm/s request increment before clipping. This is a fixed actor parameter selected on the development records, not an additional PPO update. Subsequent offline refinement fits feasible action targets with anchors from complete trajectories.')
    provenance = provenance.replace('Preview features, joint correction and dictionary refinement',
        'Output-bias selection, preview features, joint correction and dictionary refinement')
    experiments = experiments.replace(r'The window totals and the distinct full-route partition are reported in \ref{app:experimental_checks}; they retain all corners and all recorded motion.',
        r'\tabref{tab:window_sensitivity} reports the window totals and the distinct full-route partition, retaining all corners and all recorded motion.')
    insert = r'\subsection{Ablation Results}'
    assert experiments.count(insert) == 1
    experiments = experiments.replace(insert, first+'\n\n'+insert)
    experiments = experiments.replace(r'\ref{app:experimental_checks} distinguishes this selected ancestry from the wider development budget.',
        r'\tabref{tab:training_provenance} distinguishes this selected ancestry from the wider development budget.')
    target = r'\begin{figure}[!htbp]'+'\n'+r'\centering'+'\n'+r'\includegraphics[width=0.98\linewidth]{Figure_9.pdf}'
    assert experiments.count(target) == 1
    replacement = provenance.strip()+'\n\n'+snapshots.strip()+'\n\n'+target.replace('[!htbp]','[H]')
    experiments = experiments.replace(target, replacement)
    experiments = experiments.replace('Taken together, the comparisons retain', 'The comparisons retain')
    assert r'app:experimental_checks' not in retained+experiments
    for label in ('window_sensitivity','training_provenance','refinement_snapshots'):
        assert (retained+experiments).count(r'\label{tab:'+label+'}') == 1
        assert r'\label{tab:'+label+'}' in experiments
    appendix_path.write_text(retained.rstrip()+'\n', encoding='utf-8', newline='\n')
    experiments_path.write_text(experiments, encoding='utf-8', newline='\n')
    print('THREE_EXPERIMENT_TABLES_MOVED_TO_BODY', history)


if __name__ == '__main__':
    main()
