"""Bind the unchanged local-frame helper to the existing main-model manifest."""
from common import ROOT, sha, utc, write_json
from main_evidence_v5 import MANIFEST, FIGURE_DEPENDENCIES, read, check


def main():
    archived = ROOT / 'archive/main_model_preview_before_style_and_dependency/main_model_v5.json'
    check(archived.exists(), 'Preserve the prior manifest before changing its dependency binding')
    manifest = read(MANIFEST)
    relative = FIGURE_DEPENDENCIES['local_frame']
    expected = '8962ce650a2cf44a720d1af47f1d601a31b7965bc0d6ea5c64365e222802d08d'
    check(sha(ROOT / relative) == expected, 'Local-frame helper differs from the reviewed implementation')
    if manifest.get('figure_dependencies') == FIGURE_DEPENDENCIES:
        check(manifest['sources'][relative]['sha256'] == expected, 'Existing helper binding differs')
        return
    check(sha(MANIFEST) == sha(archived), 'Manifest changed since its preserved snapshot')
    manifest['figure_dependencies'] = FIGURE_DEPENDENCIES
    manifest['sources'][relative] = dict(sha256=expected, role='Reference-based corner and plotting-frame selection')
    manifest['dependency_binding_revision'] = dict(created_utc=utc(), prior_manifest_sha256=sha(archived),
        unchanged_run_and_model_bindings=True, unchanged_local_frame_implementation=True)
    write_json(MANIFEST, manifest)
    print('Bound the unchanged local-frame helper; original manifest and generated preview retained.')


if __name__ == '__main__':
    main()
