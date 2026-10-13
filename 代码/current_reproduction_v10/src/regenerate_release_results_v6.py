"""Rebuild the manuscript numbers and figures, checking the preserved Fig. 8 data."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from common import ROOT,write_json


def main():
    expected={n:(ROOT/'shared_manuscript/frozen_main_v6'/n).read_bytes() for n in
        ('numbers.tex','complete_table.tex','resolution_table.tex','generalization_table.tex')}
    with np.load(ROOT/'shared_manuscript/main_model_v5/figure8_source.npz') as z:
        figure8={k:z[k].copy() for k in z.files}
    commands=[['src/main_evidence_v5.py','verify'],
        ['src/audit_frozen_release_v6.py','--output','audit/regenerated_scientific_check_v6.json'],
        ['manuscript/generate_main_bound_v5.py'],
        ['manuscript/generate_frozen_main_v6.py'],
        ['src/report_frozen_main_generalization_v6.py'],
        ['src/summarize_frozen_main_cost_v6.py'],
        ['manuscript/generate_frozen_generalization_v6.py']]
    for command in commands:
        subprocess.run([sys.executable,'-B',*command],cwd=ROOT,check=True)
    for name,content in expected.items():
        assert (ROOT/'shared_manuscript/frozen_main_v6'/name).read_bytes()==content,name
    with np.load(ROOT/'shared_manuscript/main_model_v5/figure8_source.npz') as z:
        assert set(z.files)==set(figure8)
        assert all(np.array_equal(z[k],v) for k,v in figure8.items())
    result=dict(all_four_generated_tables_identical=True,figure8_all_plot_arrays_identical=True,
        all_registered_outcomes_included=True,commands=commands,
        preserved_manuscript_figure8='shared_manuscript/frozen_main_v6/figure8.pdf')
    write_json(ROOT/'audit/portable_regeneration_receipt_v6.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
