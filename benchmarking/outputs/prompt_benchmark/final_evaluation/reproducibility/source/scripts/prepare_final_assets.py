"""Package the exact scored cohort and case-owned input images; never invoke a model."""
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from consolidate_final_benchmark import FINAL, file_hash
from project_paths import ROOT
from prompt_benchmark.core import load_plan, write_json

def prepare():
    plan=load_plan(FINAL)
    shutil.copy2(ROOT/'outputs/semantic_review/service_definitions.json',FINAL/'config/service_definitions.json')
    write_json(FINAL/'config/output_schema.json',plan['schema'])
    (FINAL/'data').mkdir(exist_ok=True)
    source=ROOT/'results/category_matched_common_support_with_photos.csv'
    with source.open(encoding='utf-8-sig',newline='') as f: lookup={r['ID']:r for r in csv.DictReader(f)}
    fields=['ID','ORIGINAL_TYPE','CURRENT_TYPE','label','LABEL_PROVENANCE','input_sha256']
    with (FINAL/'data/scoring_labels.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for c in plan['cases']:
            w.writerow({**{k:lookup[c['id']].get(k,'') for k in fields[:-1]},'input_sha256':c['evaluation_join_hash']})
    with (FINAL/'data/blinded_cases.jsonl').open('w',encoding='utf-8') as f:
        for c in plan['cases']: f.write(json.dumps(c,ensure_ascii=False)+'\n')
    cache=ROOT/'outputs/prompt_benchmark/photo_cache_100'
    ownership=json.loads((cache/'download_audit.json').read_text(encoding='utf-8'))
    records=[]
    for r in ownership:
        target=FINAL/'assets/photos'/f"{r['sha256']}.jpg"; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            try: os.link(r['path'],target)
            except OSError: shutil.copy2(r['path'],target)
        assert file_hash(target)==r['sha256']
        records.append({**r,'bundle_path':str(target.relative_to(FINAL)).replace('\\','/')})
    write_json(FINAL/'assets/photo_ownership.json',records)
    packages=['requests','pandas','numpy','matplotlib','seaborn','Pillow','nbformat','nbclient','nbconvert','ipykernel','jsonschema']
    environment={'python':platform.python_version(),'platform':platform.platform(),
                 'packages':{p:importlib.metadata.version(p) for p in packages},
                 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip()}
    write_json(FINAL/'reproducibility/environment.json',environment)
    (FINAL/'reproducibility/requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in environment['packages'].items())+'\n',encoding='utf-8')
    write_json(FINAL/'data/source_manifest.json',{'original_evaluation_source':str(source),'original_evaluation_sha256':file_hash(source),
        'scoring_labels_sha256':file_hash(FINAL/'data/scoring_labels.csv'),'blinded_cases_sha256':file_hash(FINAL/'data/blinded_cases.jsonl'),
        'case_order':[c['id'] for c in plan['cases']],'case_input_hashes':{c['id']:c['evaluation_join_hash'] for c in plan['cases']},
        'photo_attachments':len(records),'cases_with_photos':sum(bool(c['photos']) for c in plan['cases']),
        'photo_storage':'Normalized images are hard linked when supported, avoiding duplicate image storage. Original downloaded bytes remain in the preserved source cache.',
        'evaluation_fields_never_sent_to_models':True})
    print('Prepared 100-case scoring-only labels, blinded cases, verified photo assets, and environment snapshot.')

if __name__=='__main__': prepare()
