"""Verify inherited bibliography metadata; do not infer full-text support."""
import difflib
import html
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from study_io import BASE,ROOT,register,write_json,receipt


def normalized(text):
    text=unicodedata.normalize('NFKD',html.unescape(str(text))).lower()
    return re.sub('[^a-z0-9]','',text)


if __name__=='__main__':
    bib=ROOT/'论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous/references.bib'
    entries=[]
    for kind,key,body in re.findall(r'@(\w+)\{([^,]+),\s*(.*?)\n\}',bib.read_text(encoding='utf-8'),re.S):
        fields={k.lower():v for k,v in re.findall(r'(\w+)\s*=\s*\{(.*?)\}\s*(?:,|$)',body,re.S)}
        entries.append(dict(key=key,kind=kind,fields=fields))
    if not 1<=len(entries)<=50:
        raise ValueError('Unexpected bounded bibliography size')
    stage=register('manuscript_reference_metadata_v1',[__file__,bib],
        dict(entries=len(entries),http_requests_max=2*sum('doi' in r['fields'] for r in entries),
             timeout_per_request_s=15,fulltext_claims=0,models=0,physics_ticks=0),
        'Check original citation identity against Crossref; preserve original bibliography and distinguish metadata verification from full-text reading.')
    research=ROOT/'论文项目/research_20260915_joint_policy_v1'
    cached={}
    for name in ['sencer','toppra','ruckig']:
        path=research/'motion_sources'/(name+'_crossref.json')
        if path.exists():
            raw=json.loads(path.read_text(encoding='utf-8'))
            message=raw.get('message',raw)
            if 'DOI' in message:
                cached[message['DOI'].lower()]=(message,str(path.relative_to(ROOT)))
    results=[]
    requests_made=0
    for entry in entries:
        fields=entry['fields']; doi=fields.get('doi')
        result=dict(key=entry['key'],doi=doi,original_title=fields.get('title'),original_year=fields.get('year'),
                    fulltext_read_in_this_stage=False)
        metadata=None
        if not doi:
            result.update(status='manual_needed',note='No DOI in original entry; retain original identifier for separate verification.')
        elif doi.lower() in cached:
            metadata,result['source']=cached[doi.lower()]
        else:
            url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
            for attempt in range(2):
                try:
                    request=urllib.request.Request(url,headers={'User-Agent':'PPO-article-reference-audit/1.0','Accept':'application/json'})
                    requests_made+=1
                    with urllib.request.urlopen(request,timeout=15) as response:
                        raw=json.loads(response.read().decode('utf-8'))
                    metadata=raw['message']
                    write_json(stage/(entry['key']+'_crossref.json'),raw)
                    result['source']=url
                    break
                except urllib.error.HTTPError as error:
                    result.update(status='not_found' if error.code==404 else 'manual_needed',error=str(error))
                    if attempt==0 and (error.code==429 or error.code>=500):
                        time.sleep(20 if error.code==429 else 2)
                        continue
                    break
                except (OSError,ValueError) as error:
                    result.update(status='manual_needed',error=str(error))
                    if attempt==0:
                        time.sleep(2)
                        continue
            time.sleep(.2)
        if metadata is not None:
            actual_title=metadata.get('title',[''])[0]
            years=[]
            for name in ['published','published-print','published-online','issued']:
                dates=metadata.get(name,{}).get('date-parts',[])
                if dates and dates[0]:years.append(str(dates[0][0]))
            similarity=difflib.SequenceMatcher(None,normalized(fields.get('title','')),normalized(actual_title)).ratio()
            doi_matches=metadata.get('DOI','').lower()==doi.lower()
            year_matches=fields.get('year') in years
            result.update(retrieved_title=actual_title,retrieved_journal=metadata.get('container-title',[]),
                retrieved_years=sorted(set(years)),retrieved_authors=metadata.get('author',[]),
                title_similarity=similarity,doi_matches=doi_matches,year_matches=year_matches,
                status='verified_identity' if doi_matches and year_matches and similarity>=.94 else 'metadata_mismatch_review_needed')
        results.append(result)
        write_json(stage/'results.json',dict(results=results,http_requests=requests_made,complete=len(results)==len(entries)))
        print(entry['key'],result['status'],flush=True)
    counts={status:sum(r['status']==status for r in results) for status in sorted({r['status'] for r in results})}
    receipt(stage,dict(status='inherited_bibliography_identity_check_completed',entries=len(entries),
        counts=counts,http_requests=requests_made,fulltext_read=0,bibliography_modified=False,goal_complete=False))
