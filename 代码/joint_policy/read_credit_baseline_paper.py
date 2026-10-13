"""Archive an openly published paper for a focused credit-assignment review."""
import subprocess
import urllib.request
from pathlib import Path
from study_io import BASE, register, receipt, write_json, sha


def main():
    stage = register('credit_baseline_fulltext_v1', [Path(__file__),
        BASE/'credit_outcome_decomposition_v1/receipt.json', BASE/'学习信号与后续选择_20260915.md'],
        dict(http_requests=2, pdf_text_extractions=1, models=0, physics_ticks=0),
        'Read the official ICML/PMLR paper on action-dependent baselines before choosing any new credit-assignment mechanism.')
    sources = [('publisher.html', 'https://proceedings.mlr.press/v80/tucker18a.html'),
               ('tucker18a.pdf', 'https://proceedings.mlr.press/v80/tucker18a/tucker18a.pdf')]
    entries = []
    for filename, url in sources:
        request = urllib.request.Request(url, headers={'User-Agent': 'PPO-article-literature-review/1.0'})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read()
            final_url, content_type = response.url, response.headers.get('Content-Type')
        (stage/filename).write_bytes(data)
        entries.append(dict(filename=filename, url=url, resolved_url=final_url,
            content_type=content_type, sha256=sha(stage/filename), size=len(data)))
    assert (stage/'tucker18a.pdf').read_bytes().startswith(b'%PDF')
    process = subprocess.run(['D:/texlive/2025/bin/windows/pdftotext.exe', '-layout',
        str(stage/'tucker18a.pdf'), str(stage/'tucker18a.txt')], capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.decode('utf-8', errors='replace'))
    write_json(stage/'sources.json', dict(sources=entries,
        connector_attempt='scite connector USER_NOT_LOGGED_IN; Chrome unavailable to current browser tool; publisher original used',
        title='The Mirage of Action-Dependent Baselines in Reinforcement Learning',
        venue='Proceedings of the 35th International Conference on Machine Learning',
        year=2018, volume=80, pages='5015-5024', reviewed_fulltext=False))
    receipt(stage, dict(status='official_fulltext_archived_pending_reading', http_requests=2,
        pdf_text_extractions=1, models=0, physics_ticks=0))
    print(stage/'tucker18a.txt')


if __name__ == '__main__':
    main()
