"""Create the six-model frozen checkpoint, importing prior rows without inference."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import shutil
import sqlite3
from project_paths import ROOT
from prompt_benchmark.core import bind_plan, connect, experiments, load_plan, sha, write_json

FINAL=ROOT/'outputs/prompt_benchmark/final_evaluation'
SOURCE=ROOT/'outputs/prompt_benchmark/multimodal_v2'
NEW_MODELS=[
    {'id':'llama3.1-8b','adapter':'ollama','model':'llama3.1:8b','supports_photos':False,'supports_thinking':False,
     'options':{'temperature':0,'seed':42,'num_ctx':16384,'num_predict':384},'pricing':{'input_per_million':0,'output_per_million':0}},
    {'id':'ministral3-8b','adapter':'ollama','model':'ministral-3:8b','supports_photos':True,'supports_thinking':False,
     'options':{'temperature':0,'seed':42,'num_ctx':16384,'num_predict':384},'pricing':{'input_per_million':0,'output_per_million':0}},
]

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()

def consolidate(output=FINAL,source=SOURCE):
    output,source=Path(output),Path(source)
    old=load_plan(source)
    plan=copy.deepcopy(old)
    plan.pop('id')
    plan['config']['models']+=copy.deepcopy(NEW_MODELS)
    plan['parent_plan_id']=old['id']
    plan['id']=sha(plan)
    if (output/'plan.json').exists():
        assert load_plan(output)==plan,'Existing final plan differs'
        if (output/'provenance/import.json').exists():
            return plan
        # Recover an interrupted import transaction without discarding any completed rows.
    output.mkdir(parents=True,exist_ok=True)
    manifest={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))}
    write_json(output/'provenance/source_files.json',manifest)
    write_json(output/'plan.json',plan)
    write_json(output/'config/models.json',plan['config'])
    for variant,prompt in plan['prompts'].items():
        (output/'config/templates').mkdir(parents=True,exist_ok=True)
        setting=plan['config']['prompts'][variant]
        (output/'config'/setting['template']).write_text(prompt['template'],encoding='utf-8')
        (output/f'{variant}_prompt.txt').write_text(prompt['system'],encoding='utf-8')
    assert plan['cases']==old['cases'] and plan['prompts']==old['prompts'] and plan['schema']==old['schema']
    mapping={(e['model']['id'],e['modality'],e['variant']):e['id'] for e in experiments(plan)}
    old_experiments={e['id']:mapping[e['model']['id'],e['modality'],e['variant']] for e in experiments(old)}
    src=sqlite3.connect(f'file:{(source/"checkpoint.sqlite").as_posix()}?mode=ro',uri=True)
    src.row_factory=sqlite3.Row
    dst=connect(output)
    bind_plan(dst,plan)
    dst.executescript('''CREATE TABLE IF NOT EXISTS prediction_origins(experiment_id TEXT,case_id TEXT,source_experiment_id TEXT,source_plan_id TEXT,source_row_hash TEXT,PRIMARY KEY(experiment_id,case_id));
      CREATE TABLE IF NOT EXISTS attempt_origins(id INTEGER PRIMARY KEY,source_id INTEGER,source_experiment_id TEXT,source_row_hash TEXT);''')
    imported=0
    imported_attempts=0
    with dst:
        for r in src.execute('SELECT * FROM predictions'):
            values=dict(r); original=sha(values); values['experiment_id']=old_experiments[r['experiment_id']]
            dst.execute('INSERT OR IGNORE INTO predictions VALUES ('+','.join('?' for _ in values)+')',list(values.values()))
            actual=dict(dst.execute('SELECT * FROM predictions WHERE experiment_id=? AND case_id=?',(values['experiment_id'],r['case_id'])).fetchone())
            assert actual==values,'Existing imported prediction differs'
            dst.execute('INSERT OR IGNORE INTO prediction_origins VALUES (?,?,?,?,?)',(values['experiment_id'],r['case_id'],r['experiment_id'],old['id'],original))
            imported+=1
        for r in src.execute('SELECT * FROM attempts ORDER BY id'):
            values=dict(r); original=sha(values); values['experiment_id']=old_experiments[r['experiment_id']]
            dst.execute('INSERT OR IGNORE INTO attempts VALUES ('+','.join('?' for _ in values)+')',list(values.values()))
            actual=dict(dst.execute('SELECT * FROM attempts WHERE id=?',(r['id'],)).fetchone())
            assert actual==values,'Existing imported attempt differs'
            dst.execute('INSERT OR IGNORE INTO attempt_origins VALUES (?,?,?,?)',(r['id'],r['id'],r['experiment_id'],original))
            imported_attempts+=1
        for r in src.execute("SELECT * FROM meta WHERE key LIKE 'model_fingerprint:%'"):
            dst.execute('INSERT OR IGNORE INTO meta VALUES (?,?)',tuple(r))
    assert imported==len(old['cases'])*len(old['config']['models'])*len(old['modalities'])*2
    assert dst.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    src.close(); dst.close()
    for name in ('execution_summary.json','model_execution.json','independent_audit.json','abstention_investigation.json','postrun_validation.json','model_downloads.json','server_configuration.json'):
        if (source/name).exists():
            target=output/'provenance/existing_four'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source/name,target)
    write_json(output/'provenance/import.json',{'source':str(source),'source_plan_id':old['id'],'final_plan_id':plan['id'],
        'prediction_rows_reused':imported,'attempt_rows_reused':imported_attempts,
        'mapping':'Only experiment IDs change to the new plan namespace; origin tables preserve source IDs and exact source-row hashes.',
        'shared_photo_cache':str(ROOT/'outputs/prompt_benchmark/photo_cache_100'),'prior_files_unchanged':all(file_hash(source/k)==v for k,v in manifest.items())})
    print(f'Imported {imported} prior prediction/status rows; no model inference. Final directory: {output}',flush=True)
    return plan

if __name__=='__main__': consolidate()
