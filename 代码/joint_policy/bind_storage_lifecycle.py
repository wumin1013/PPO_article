"""Bind existing cleanup proofs to the exact missing historical writer inputs."""
import gzip
import hashlib
import json
from pathlib import Path
from study_io import BASE, ROOT, register, receipt, write_json, sha


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    i=ROOT/'论文项目/revision_20260913_preview_efficiency_v2'
    regs=[i/'adaptive_feed_integration_v1/failure_frame_diagnostic_v1/registration.json',
          i/'recursive_kcm_roundoff_continuation_v1/initial_extraction_v1/registration.json',
          i/'recursive_kcm_roundoff_v1/vector_validation_v1/receipt.json']
    stage=register('storage_lifecycle_binding_20260916',[Path(__file__),ROOT/'PPO_存储与清理规则.md',*regs],
        dict(registered_files_upper=100,archive_decompressions_upper=1,model_loads=0,physics_ticks=0),
        'Reconcile authorized retired checkpoints/chunks and compressed logs using original registration hashes, retention records and deletion receipts; never synthesize missing bytes or hashes.')
    missing={}
    for p in regs:
        r=read(p);base=ROOT if 'inputs' in r else p.parent
        for name,digest in r.get('inputs',r.get('files',{})).items():
            path=base/name
            if not path.is_file():
                rel=path.relative_to(ROOT).as_posix()
                missing[rel]=dict(original_sha256=digest,registration=p.relative_to(ROOT).as_posix(),registration_sha256=sha(p))
    audit=Path('D:/PPO_storage_cleanup_20260915')
    proofs={};sources={}
    for filename in ('core_retention.jsonl','d_prune_units.jsonl','d_prune_log.jsonl','compressed_logs.jsonl'):
        source=audit/filename
        sources[filename]=dict(path=str(source),sha256=sha(source))
        selected={}
        with source.open(encoding='utf-8-sig') as stream:
            for lineno,line in enumerate(stream,1):
                row=json.loads(line);rel=row.get('rel','')
                if rel in missing or (row.get('kind')=='directory' and any(k.startswith(rel+'/') for k in missing)):
                    selected[rel]=dict(line_number=lineno,record=row,line_sha256=hashlib.sha256(line.encode('utf-8')).hexdigest())
        proofs[filename]=selected
    entries={}
    for rel,binding in missing.items():
        retained=proofs['core_retention.jsonl'][rel]['record']
        assert retained['verified'] and retained['sha256']==binding['original_sha256']
        compressed=proofs['compressed_logs.jsonl'].get(rel)
        if compressed:
            record=compressed['record'];archive=ROOT/record['archive_rel']
            assert record['original_sha256']==binding['original_sha256'] and sha(archive)==record['archive_sha256']
            digest=hashlib.sha256()
            with gzip.open(archive,'rb') as f:
                for block in iter(lambda:f.read(1048576),b''):digest.update(block)
            assert digest.hexdigest()==binding['original_sha256']
            entries[rel]=dict(**binding,status='archived_losslessly',archive=record['archive_rel'],
                archive_sha256=record['archive_sha256'],restored_stream_sha256=digest.hexdigest(),
                retention=proofs['core_retention.jsonl'][rel],compression=compressed)
        else:
            assert retained['d_action'] in ('delete_old_evaluation_restart','delete_noncore_process'),retained
            candidates=[k for k in proofs['d_prune_log.jsonl'] if rel==k or rel.startswith(k+'/')]
            target=max(candidates,key=len)
            deletion=proofs['d_prune_log.jsonl'][target];plan=proofs['d_prune_units.jsonl'][target]
            assert deletion['record']['status']=='removed_noncore_or_archived_original'
            assert deletion['record']['bytes']==plan['record']['bytes']
            if plan['record']['kind']=='file':assert plan['record']['sha256']==binding['original_sha256']
            entries[rel]=dict(**binding,status='authorized_cleanup_retired',
                retention=proofs['core_retention.jsonl'][rel],plan=plan,deletion=deletion)
    write_json(stage/'registry.json',dict(status='verified_historical_storage_lifecycle',entries=entries,
        audit_sources=sources,scope='Registered historical inputs only. Retired bytes cannot be rehashed or restored; original SHA provenance and actual deletion are recorded separately. Core models remain mandatory.'))
    receipt(stage,dict(status='completed_storage_lifecycle_binding',files=len(entries),
        retired=sum(v['status']=='authorized_cleanup_retired' for v in entries.values()),
        archived=sum(v['status']=='archived_losslessly' for v in entries.values()),registry_sha256=sha(stage/'registry.json'),
        model_loads=0,physics_ticks=0,training_decisions=0))
    print(json.dumps(read(stage/'receipt.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
