"""Fetch bounded public primary sources for the exploration diagnosis."""
import json
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from study_io import BASE,register,receipt,write_json


class Text(HTMLParser):
    def __init__(self):
        super().__init__();self.parts=[]
    def handle_data(self,data):
        if data.strip(): self.parts.append(data.strip())


def main():
    stage=register('temporal_exploration_sources_v1',[__file__],
        dict(http_requests_max=4,models=0,physics_ticks=0),
        'Read primary-source metadata and an official paper page concerning smooth exploration; no algorithm change.')
    queries=[('smooth_crossref','https://api.crossref.org/works?'+urllib.parse.urlencode(dict(
        **{'query.title':'Smooth Exploration for Robotic Reinforcement Learning','rows':3}))),
        ('colored_crossref','https://api.crossref.org/works?'+urllib.parse.urlencode(dict(
        **{'query.title':'Colored Noise for Action Space Exploration in Reinforcement Learning','rows':3}))),
        ('smooth_pmlr','https://proceedings.mlr.press/v164/raffin22a.html')]
    results=[]
    for name,url in queries:
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'PPO-article-literature-review/1.0'})
            with urllib.request.urlopen(request,timeout=30) as response:
                data=response.read();status=response.status
            extension='.json' if 'crossref' in name else '.html'
            (stage/(name+extension)).write_bytes(data)
            if extension=='.html':
                parsed=Text();parsed.feed(data.decode('utf-8'))
                (stage/(name+'.txt')).write_text('\n'.join(parsed.parts),encoding='utf-8')
                print('\n'.join(parsed.parts)[-11000:])
            else:
                values=json.loads(data)['message']['items']
                print(json.dumps([dict(title=v.get('title'),DOI=v.get('DOI'),URL=v.get('URL'),
                    year=v.get('published'),authors=v.get('author'),abstract=v.get('abstract'),links=v.get('link')) for v in values],ensure_ascii=False))
            results.append(dict(name=name,url=url,status=status,saved_bytes=len(data)))
        except Exception as error:
            results.append(dict(name=name,url=url,error=repr(error)))
    write_json(stage/'fetches.json',results)
    receipt(stage,dict(status='bounded_source_retrieval_completed',requests=3,results=results,models=0,physics_ticks=0))


if __name__=='__main__':main()
