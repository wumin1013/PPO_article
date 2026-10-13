"""Keep measured results in experiments and training configuration in appendix."""
import shutil
from datetime import datetime, timezone
from build_lowdim_original_manuscript import ROOT, WORK


def main():
    ep, ap = WORK/'experiments.tex', WORK/'appendix.tex'
    experiments, appendix = ep.read_text(encoding='utf-8'), ap.read_text(encoding='utf-8')
    label = r'\label{tab:training_provenance}'
    assert label in experiments and label not in appendix
    label_at = experiments.index(label)
    start = experiments.rfind(r'\begin{table}[H]',0,label_at)
    end = experiments.index(r'\tabref{tab:refinement_snapshots}',label_at)
    provenance = experiments[start:end].strip()
    history = ROOT/'lowdim_manuscript_history'/('only_results_in_body_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    history.mkdir(parents=True)
    for path in (ep,ap):shutil.copy2(path,history/path.name)
    experiments = experiments[:start]+experiments[end:]
    experiments = experiments.replace(r'\tabref{tab:training_provenance} distinguishes this selected ancestry from the wider development budget.',
        r'\tabref{tab:training_provenance} in Appendix~B distinguishes this selected ancestry from the wider development budget.')
    appendix = appendix.rstrip()+'\n\n'+provenance+'\n'
    for label_name in ['main_results','window_sensitivity','refinement_snapshots']:
        assert (r'\label{tab:'+label_name+'}') not in appendix
    ep.write_text(experiments,encoding='utf-8',newline='\n')
    ap.write_text(appendix,encoding='utf-8',newline='\n')
    print('ONLY_MEASURED_RESULTS_IN_BODY',history)


if __name__=='__main__':main()
