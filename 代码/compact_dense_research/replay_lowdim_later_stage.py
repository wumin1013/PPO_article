"""Bounded later stages with own-seed contexts and explicit shared-data scope."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import ast
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from evaluate_historical_dense import ROOT, BASE, write, sha, verify_bound_inputs
from replay_lowdim_training_stage import SeedConstants, SEEDS, SUPPLEMENT

STAGES = {
    'ppo_preview': ('train_direct_previewcue_ppo.py', 'preview_fit/fit_12000.pt', 'checkpoint_0024.pt'),
    'contexts': ('train_previewcue_guided_time_ppo.py', 'ppo_preview/checkpoint_0024.pt', 'all34_contexts.pt'),
    'ppo_guided': ('train_previewcue_batchreset_ppo.py', 'ppo_preview/checkpoint_0024.pt', 'checkpoint_0024.pt'),
    'farvector_fit': ('fit_farvector_feedback_round19.py', 'ppo_guided/checkpoint_0024.pt', 'fit_16000.pt'),
    'local_feedback_fit': ('fit_farvector_local_feedback_round21.py', 'farvector_fit/fit_16000.pt', 'fit_08000.pt'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class OwnPreviewTrace(ast.NodeTransformer):
    def __init__(self, path):
        self.path, self.replacements = path, 0

    def visit_BinOp(self, node):
        if (isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name)
                and node.left.id == 'ROOT' and isinstance(node.right, ast.Constant)
                and node.right.value == 'direct14_previewcue_fit_seed922302_round11/eval_12000_original.npz'):
            self.replacements += 1
            return ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                args=[ast.Constant(str(self.path))], keywords=[])
        return self.generic_visit(node)


class GuidedReplay(ast.NodeTransformer):
    """Replace old-seed context source and its inapplicable same-seed diagnosis."""
    def __init__(self, contexts):
        self.contexts = contexts
        self.context_replacements = self.batch_replacements = 0

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == 'reused' for t in node.targets):
            self.context_replacements += 1
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(self.contexts))], keywords=[])
        return self.generic_visit(node)

    def visit_If(self, node):
        # The original assertion compared round16/17 at the SAME historical seed.
        # It is not applicable to an independent initialization. Keep an audited
        # own-seed batch check in its place; do not weaken context/physics checks.
        if isinstance(node.test, ast.Compare) and ast.unparse(node.test) == 'update_count == 0':
            if 'paired_first_batch_audit.json' in ast.unparse(node):
                self.batch_replacements += 1
                node.body = ast.parse("""
assert len(batch['reward']) == 2048
assert all(np.isfinite(np.asarray(batch[k])).all() for k in keys)
write(STAGE/'independent_first_batch_audit.json', dict(seed=SEED,
    source_checkpoint=str(INITIAL), source_sha256=sha(INITIAL),
    context_sha256=sha(reused/'all34_contexts.pt'),
    samples=2048, finite_fields=True,
    scope='Fresh own-seed trajectories; no equality claim to a different historical seed'))
""").body
        return self.generic_visit(node)


def collect_contexts(stage, initial):
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from previewcue_time_runtime import PreviewCueTimeRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    started = time.perf_counter()
    nominal = initial.parent/'eval_0024_original.json'
    expected_trace = nominal.with_suffix('.npz')
    expected_result = read(nominal)
    if not expected_result['success']:
        write(stage/'result.json', dict(status='prerequisite_failed',
            reason='Fixed own-seed PPO output has no successful complete original trace',
            source_result=str(nominal), source_result_sha256=sha(nominal),
            source_failure=expected_result['reason'], full_pipeline_complete=False))
        return
    torch.set_num_threads(1)
    reference = np.loadtxt(BASE/'paper_evidence_review/natural_closed_case/reference.csv', delimiter=',', skiprows=1)
    geometry = read(BASE/'paper_evidence_review/natural_closed_case/geometry.json')
    policy = FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(torch.load(initial, map_location='cpu', weights_only=False)['policy'])
    policy.eval()

    class Collector(PreviewCueTimeRuntime):
        def __init__(self, reference):
            super().__init__(reference, potential_speed=100.)
            self.contexts = []

        def observation(self):
            value = super().observation()
            corners = geometry['sharp_corners']
            while len(self.contexts) < 34 and self.env.progress()[0] >= corners[len(self.contexts)]['reference_s_mm']-35.:
                corner = corners[len(self.contexts)]
                self.contexts.append(dict(id=corner['id'], progress_mm=self.env.progress()[0],
                    state=self.state(), observation=value.copy()))
            return value

    collector = Collector(reference)
    result = evaluate(policy, collector, stage/'context_collection.npz', started+1640.)
    with np.load(stage/'context_collection.npz') as data:
        rows = data['rows']
    with np.load(expected_trace) as data:
        expected = data['rows']
    exact = bool(np.array_equal(rows, expected))
    assert result['success'] and exact, 'Own-seed context replay differs from its complete evaluation'
    result['analytic'] = analytic_command_audit(reference[0], rows, geometry)
    assert result['analytic']['analytic_corridor_pass'] and len(collector.contexts) == 34
    restore = PreviewCueTimeRuntime(reference, potential_speed=100.)
    for context in collector.contexts:
        restore.restore(context['state'])
        assert np.array_equal(restore.observation(), context['observation'])
    torch.save(collector.contexts, stage/'all34_contexts.pt')
    result.update(context_count=34, bitwise_full_trace_reproduction=exact,
        exact_observation_restore=True, contexts_sha256=sha(stage/'all34_contexts.pt'),
        expected_trace=str(expected_trace), expected_trace_sha256=sha(expected_trace),
        checkpoint=str(initial), checkpoint_sha256=sha(initial), training_only=True,
        new_ppo_updates=0, wall_s=time.perf_counter()-started)
    write(stage/'context_collection_audit.json', result)
    write(stage/'result.json', dict(status='contexts_complete', **result))
    print('OWN_CONTEXTS_COMPLETE', len(collector.contexts), flush=True)


def worker(args, stage, initial, source):
    if args.stage == 'contexts':
        collect_contexts(stage, initial)
        return
    tree = SeedConstants(args.seed).visit(ast.parse(source.read_text(encoding='utf-8')))
    if args.stage == 'ppo_preview':
        nominal = initial.parent/'eval_12000_original.json'
        prior = read(nominal)
        if not prior['success']:
            write(stage/'result.json', dict(status='prerequisite_failed',
                reason='Fixed own-seed preview fit lacks the required complete original trajectory',
                source_result=str(nominal), source_result_sha256=sha(nominal),
                source_failure=prior['reason'], full_pipeline_complete=False))
            return
        transform = OwnPreviewTrace(nominal.with_suffix('.npz'))
        tree = transform.visit(tree)
        assert transform.replacements == 2  # Worker check and original main's provenance check.
    if args.stage == 'ppo_guided':
        transform = GuidedReplay(stage.parent/'contexts')
        tree = transform.visit(tree)
        assert (transform.context_replacements, transform.batch_replacements) == (1, 1)
        # Allocate the unused optional-refinement reserve to the same fixed
        # 24 updates and scheduled checks, within the author's 30-minute cap.
        deadlines = [node for node in ast.walk(tree) if isinstance(node,ast.Constant)
                     and isinstance(node.value,float) and node.value==1380.]
        assert len(deadlines)==1
        deadlines[0].value=1640.
    ast.fix_missing_locations(tree)
    namespace = {'__name__': 'independent_later_stage', '__file__': str(source)}
    exec(compile(tree, str(source), 'exec'), namespace)
    namespace.update(STAGE=stage, INITIAL=initial, SEED=args.seed)
    if args.stage == 'local_feedback_fit':
        # Preserve recorded physical labels AND their historical teacher so the
        # original reconstruction assertion remains valid. Student is own-seed.
        namespace['initial_path'] = lambda: initial
    (stage/'transformed_worker.py').write_text(ast.unparse(tree)+'\n', encoding='utf-8')
    namespace['worker']()


def main(args):
    source_name, previous, selected = STAGES[args.stage]
    seedroot = SUPPLEMENT/'independent_training'/f'seed_{args.seed}'
    stage = seedroot/args.stage
    initial = seedroot/previous
    source = ROOT/source_name
    if args.worker:
        worker(args, stage, initial, stage/'source'/source_name)
        return
    assert initial.exists(), initial
    if args.stage == 'ppo_guided':
        context = read(seedroot/'contexts/context_collection_audit.json')
        assert context['success'] and context['checkpoint_sha256'] == sha(initial)
    stage.mkdir(parents=True, exist_ok=False)
    (stage/'source').mkdir()
    sources = [source, Path(__file__), ROOT/'replay_lowdim_training_stage.py']
    for path in sources:
        shutil.copy2(path, stage/'source'/path.name)
    verify_bound_inputs()
    hashes = {str(path): sha(path) for path in sources}
    write(stage/'registration.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        stage=args.stage, seed=args.seed, all_preregistered_seeds=list(SEEDS),
        initial_checkpoint=str(initial), initial_checkpoint_sha256=sha(initial),
        fixed_continuation_checkpoint=selected, source_sha256=hashes,
        wall_budget_s=1800, child_timeout_s=1740,
        worker_deadline_s=1640 if args.stage!='ppo_preview' else 1380,
        evidence_gap='Three independently initialized pipelines have not reached the final actor and paired physical evaluation.',
        procedure='Use own-seed parent and random seed; reconstruct corner contexts from that exact complete trace. Keep fixed corpus and historical physical/optimizer parameters. Local feedback labels retain the recorded historical teachers and their consistency assertion.',
        adaptation='Preview PPO validates context collection against its own-seed fit trace. Guided PPO uses own-seed contexts and replaces the historical same-seed round16/17 batch-equality diagnostic with an own-seed finite-batch audit. Physics and optimizer are unchanged.',
        adoption='Fixed checkpoints, no success-conditioned seed replacement, final two-case two-resolution acceptance required; frozen paper model stays unchanged.',
        stop='Iteration limit, prerequisite failure, exception, or 1740s worker timeout.',
        independence_scope='Independent initialization/optimization conditional on shared historical training corpus and teacher functions; not an independent reconstruction of corpus discovery.',
        full_pipeline_complete=False))
    started = time.perf_counter()
    code = None
    with (stage/'stdout.txt').open('w', encoding='utf-8') as out, (stage/'stderr.txt').open('w', encoding='utf-8') as err:
        try:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--stage', args.stage,
                '--seed', str(args.seed), '--worker'], stdout=out, stderr=err, timeout=1740)
            code = result.returncode
            status = 'finished' if code == 0 else 'error'
        except subprocess.TimeoutExpired:
            status = 'timeout'
    if (stage/'result.json').exists() and read(stage/'result.json').get('status') == 'prerequisite_failed':
        status = 'prerequisite_failed'
    write(stage/'replay_receipt.json', dict(status=status, exit_code=code, wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p) == h for p, h in hashes.items()),
        selected_checkpoint_exists=(stage/selected).exists(), full_pipeline_complete=False))
    print('LATER_STAGE', args.seed, args.stage, status, round(time.perf_counter()-started, 2), flush=True)
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-1200:], flush=True)
    if code != 0:
        print((stage/'stderr.txt').read_text(encoding='utf-8')[-1600:], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=list(STAGES), required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--worker', action='store_true')
    main(parser.parse_args())
