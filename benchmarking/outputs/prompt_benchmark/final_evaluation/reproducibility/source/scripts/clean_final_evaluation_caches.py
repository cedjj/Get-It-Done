"""Remove only owned, regenerable execution caches after protecting final artifacts."""
import json
from pathlib import Path
import shutil
from consolidate_final_benchmark import FINAL, file_hash
from prompt_benchmark.core import write_json, now

def clean():
    root=FINAL.resolve()
    assert root.name=='final_evaluation' and root.parent.name=='prompt_benchmark'
    targets=[root/'runtime/notebook_ipython',root/'runtime/notebook_jupyter',root/'runtime/matplotlib',
             root/'runtime/browser_profile',root/'reproducibility/__pycache__']
    present=[p for p in targets if p.exists()]
    for p in present:
        resolved=p.resolve()
        assert resolved==p and root in resolved.parents,'Cache path escaped final package'
        assert not p.is_symlink() and not p.is_junction(),'Refuse linked cache directory'
        assert not any(x.is_symlink() or x.is_junction() for x in p.rglob('*')),'Refuse cache containing directory links'
    # Every deliverable stays byte-identical. Only the listed runtime caches are excluded.
    protected={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()
               and not any(t==p or t in p.parents for t in targets)
               and p.name not in {'cleanup.json','checkpoint.sqlite-wal','checkpoint.sqlite-shm'}}
    removed=[]
    for p in present:
        size=sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
        shutil.rmtree(p)
        removed.append({'path':str(p.relative_to(root)),'bytes':size,'reason':'Regenerable notebook/browser/Python cache; authoritative source and outputs retained'})
    assert all(file_hash(root/k)==h for k,h in protected.items()),'Protected artifact changed during cache cleanup'
    write_json(root/'validation/cleanup.json',{'at':now(),'removed':removed,'protected_files_unchanged':len(protected),
        'old_experiments_deleted':False,'raw_outputs_retained_in_sqlite':True,'duplicate_raw_jsonl_not_created':True})
    print(json.dumps({'cache_directories_removed':len(removed),'protected_files_unchanged':len(protected)},indent=2))

if __name__=='__main__': clean()
