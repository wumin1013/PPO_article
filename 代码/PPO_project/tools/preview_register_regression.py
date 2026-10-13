"""Register exact historical regression references before final evaluation."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['direct-fine','direct-explore'], default='direct-explore')
    args = parser.parse_args()
    study = importlib.import_module('preview_' + args.study.replace('-', '_')).install()
    study.check_protocol()
    source = study.ROOT / '论文项目/generated/configs/full_method_snapshot.yaml'
    record_file = study.OUT / 'regression_reference_provenance.json'
    if record_file.exists():
        old = json.loads(record_file.read_text(encoding='utf-8'))
        if (old['historical_config_sha256'] != study.backend.sha(source)
                or old['protocol_sha256'] != study.backend.sha(study.OUT / 'protocol.json')):
            raise ValueError('Historical regression reference or registered protocol changed')
        for item in old['references']:
            if item['sha256'] != study.backend.sha(study.OUT / item['file']):
                raise ValueError('Registered reference points changed')
        print('Verified registered historical references', flush=True)
        return
    directory = study.OUT / 'regression_references'
    directory.mkdir(exist_ok=True)
    (directory / source.name).write_bytes(source.read_bytes())
    cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
    references = []
    for spec in cfg['training']['path_curriculum']['paths']:
        points = np.asarray(study.backend._build_path(spec), dtype=float)
        target = directory / (spec['name'] + '.csv')
        np.savetxt(target, points, delimiter=',', header='x_mm,y_mm', comments='', fmt='%.17g')
        references.append(dict(path_id=spec['name'], file=target.relative_to(study.OUT).as_posix(),
            sha256=study.backend.sha(target), samples=len(points),
            length_mm=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())))
    study.backend.write_json(record_file, dict(command='python ' + ' '.join(sys.argv),
        registered_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        historical_config=str(source), historical_config_sha256=study.backend.sha(source),
        protocol_sha256=study.backend.sha(study.OUT / 'protocol.json'),
        source_sha256=study.source_state()['source_sha256'], script_sha256=study.backend.sha(__file__),
        references=references, scope='Known historical regression paths, not unseen test data',
        verification='Run this command before and after the regression matrix; it rejects source or point changes'))
    (directory / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(record_file, flush=True)


if __name__ == '__main__':
    main()
