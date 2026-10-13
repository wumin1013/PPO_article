"""Direct PPO with shared, finer ordered-coverage witnesses at unchanged tolerance.

The only controller/training change from preview_direct_gain is the common
0.01 mm witness spacing. Online progress/terminal rewards and offline coverage
use the same certificate. Imports do not activate adapters or evaluate paths.
"""
from pathlib import Path
import hashlib
import json

import preview_direct_gain as gain
from src.utils import revision_metrics
from src.utils.ordered_coverage import OrderedCoverageTracker

study=gain.study
OUT=study.ROOT/'论文项目/revision_20260908/preview_direct_fine'
WITNESS_SPACING_MM=.01
BASE_CONFIG=gain.config_for
BASE_SOURCE=gain.source_state
BASE_PROTOCOL=gain.protocol_definition
BASE_V2_MAKE_ENV=study.v2.make_env
BASE_ORDERED_PROJECTION=revision_metrics.ordered_projection


def config_for(variant,seed):
    cfg=BASE_CONFIG(variant,seed)
    cfg['revision'].update(study='direct-distance-shared-gain15-fine-witness-v1',
        ordered_witness_spacing_mm=WITNESS_SPACING_MM,
        ordered_witness_change_role='Common online reward/termination and offline certificate refinement; not preview-only benefit')
    return cfg


class FineTimeObjective(study.v2.TimeIntegratedObjective):
    def _reset_coverage(self):
        spacing=self.cfg['revision'].get('ordered_witness_spacing_mm',WITNESS_SPACING_MM)
        if spacing!=WITNESS_SPACING_MM:
            raise ValueError('The fine study requires the registered 0.01 mm witness spacing')
        self.coverage=OrderedCoverageTracker(self.reference,tolerance_mm=self.env.half_epsilon,
            initial_position=self.env.current_position,witness_spacing_mm=spacing)
        self._objective_done=False


def fine_make_env(cfg,path_id,episode=0,reference=None):
    wrapped,observation=BASE_V2_MAKE_ENV(cfg,path_id,episode,reference=reference)
    return FineTimeObjective(wrapped.env,wrapped.cfg,wrapped.reference),observation


def fine_ordered_projection(reference,positions,*,tolerance_mm,witness_spacing_mm=None):
    """Explicit CLI-scoped default; original projection and all failure semantics remain."""
    return BASE_ORDERED_PROJECTION(reference,positions,tolerance_mm=tolerance_mm,
        witness_spacing_mm=WITNESS_SPACING_MM if witness_spacing_mm is None else witness_spacing_mm)


def source_state():
    state=BASE_SOURCE()
    for path in [Path(__file__),Path(__file__).with_name('preview_direct_fine_batch.py')]:
        state['source_files'][path.relative_to(study.ROOT).as_posix()]=study.backend.sha(path)
    state['source_sha256']=hashlib.sha256(json.dumps(state['source_files'],sort_keys=True).encode()).hexdigest()
    return state


def protocol_definition():
    value=BASE_PROTOCOL()
    value.update(version='preview-direct-shared-gain15-fine-witness-v1',
        predecessor='preview_direct_gain: six completed pilots and immutable coarse-certificate validation records; no holdout use',
        candidate_change='Only common ordered witness spacing: 0.25 mm to 0.01 mm, in online rewards/termination and offline evaluation; physical tolerance unchanged',
        diagnostic_basis=['preview_direct_gain/ordered_witness_diagnostic/summary.json',
                          'preview_direct_gain/refinement_diagnostic/summary.json'],
        shared_ordered_coverage=dict(witness_spacing_mm=WITNESS_SPACING_MM,
            radius_definition='one-sided physical tolerance minus witness_spacing_mm/2',
            nominal_one_sided_tolerance_mm=1.25,nominal_matching_radius_mm=1.245,
            tight_one_sided_tolerance_mm=.625,tight_matching_radius_mm=.620,
            scope='Same reference-witness and monotone command-segment interval rule; finer sufficient coverage certificate, not an exact continuous Frechet solver',
            online='FineTimeObjective: ordered progress and completion/failure reward',
            offline='Original evaluate_trajectory with CLI-scoped ordered_projection default spacing=0.01 mm',
            attribution='Shared fixed/direct correction; all six pilots retrained, then uniform fixed selection and all fifteen formal models retrained',
            original_results='Coarse labels and models preserved; diagnostic recertification is not used as final training results'),
        unchanged='V2 physical reward rates and coefficients, scalar constraints, geometric tolerance, KCM, endpoint handling; gain15 motion anchor and PPO parameters; same train/validation/test split',
        test_policy='No holdout use until all fifteen newly retrained formal models/configurations freeze; never automatic evaluation')
    return value


def install():
    """Install only in this explicitly selected CLI process; safe to call twice."""
    gain.install()
    study.OUT=OUT
    study.config_for=config_for
    study.source_state=source_state
    study.protocol_definition=protocol_definition
    study.v2.make_env=fine_make_env
    revision_metrics.ordered_projection=fine_ordered_projection
    return study


if __name__=='__main__':install().main()
