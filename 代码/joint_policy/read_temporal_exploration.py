"""Read the official gSDE paper and discover the colored-noise paper."""
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
from study_io import BASE,register,receipt,write_json


def main():
    stage=register('temporal_exploration_fulltext_v1',[__file__,BASE/'temporal_exploration_sources_v1/smooth_pmlr.html'],
        dict(http_requests_max=2,pdf_text_extractions=1,models=0,physics_ticks=0),
        'Read verified official gSDE full text and discover a second exploration source after Crossref title misses.')
    urls=[('smooth_exploration.pdf','https://proceedings.mlr.press/v164/raffin22a/raffin22a.pdf'),
        ('colored_openalex.json','https://api.openalex.org/works?'+urllib.parse.urlencode(dict(
            search='Colored Noise for Action Space Exploration in Deep Reinforcement Learning',**{'per-page':3})))]
    outcomes=[]
    for name,url in urls:
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PPO-article-literature-review/1.0'}),timeout=30) as response:
                data=response.read()
            (stage/name).write_bytes(data)
            outcomes.append(dict(url=url,path=name,bytes=len(data)))
        except Exception as error:
            outcomes.append(dict(url=url,error=repr(error)))
    if (stage/'smooth_exploration.pdf').exists():
        subprocess.run(['D:/texlive/2025/bin/windows/pdftotext.exe','-layout',str(stage/'smooth_exploration.pdf'),
                        str(stage/'smooth_exploration.txt')],check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    write_json(stage/'fetches.json',outcomes)
    receipt(stage,dict(status='public_source_read_completed',outcomes=outcomes,models=0,physics_ticks=0))
    if (stage/'colored_openalex.json').exists():
        data=json.loads((stage/'colored_openalex.json').read_text(encoding='utf-8'))
        for r in data['results']:
            print(json.dumps(dict(title=r['title'],year=r['publication_year'],doi=r['doi'],
                authors=[a['author']['display_name'] for a in r['authorships']],
                primary_location=r['primary_location'],locations=r['locations'],abstract=r.get('abstract_inverted_index')),ensure_ascii=False))


if __name__=='__main__':main()
