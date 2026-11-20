"""Execute only Llama 3.1 / Ministral 3 additions; never rerun imported experiments."""
import asyncio
from collections import Counter
import json
import sqlite3
import time
import requests
from consolidate_final_benchmark import FINAL, NEW_MODELS, consolidate
from execute_multimodal_benchmark import measured, verify_resume
from prompt_benchmark.core import now, sha, write_json

def imported_snapshot(output):
    c=sqlite3.connect(output/'checkpoint.sqlite'); c.row_factory=sqlite3.Row
    rows=[dict(r) for r in c.execute('SELECT p.* FROM predictions p JOIN prediction_origins o USING(experiment_id,case_id) ORDER BY p.experiment_id,p.case_id')]
    attempts=[tuple(r) for r in c.execute('SELECT a.id,a.experiment_id,a.case_id,a.request_hash,a.raw_response,a.error FROM attempts a JOIN attempt_origins o ON a.id=o.id ORDER BY a.id')]
    c.close()
    return sha(rows),sha(attempts)

async def execute():
    plan=consolidate(); before=imported_snapshot(FINAL)
    start=time.perf_counter(); started=now()
    pilot={c['id'] for c in plan['cases'][:10]}; all_ids={c['id'] for c in plan['cases']}
    models=[m for m in plan['config']['models'] if m['id'] in {x['id'] for x in NEW_MODELS}]
    records=[]
    for phase,ids in [('smoke_10',pilot),('full_100',all_ids)]:
        for model in models:
            record=next((r for r in records if r['model']==model['id']),None)
            if record is None:
                record={'model':model['id']}; records.append(record)
            if phase=='full_100' and not record.get('smoke_10_verification',{}).get('passed'):
                continue
            try:
                record[phase]=await measured(plan,FINAL,model,ids,phase)
                record[phase+'_verification']=await verify_resume(plan,FINAL,model,ids)
                record[phase+'_runtime']=requests.get('http://127.0.0.1:11434/api/ps',timeout=15).json()
            except Exception as exc:
                record[phase+'_error']=f'{type(exc).__name__}: {exc}'
            finally:
                requests.post('http://127.0.0.1:11434/api/generate',json={'model':model['model'],'keep_alive':0},timeout=60).raise_for_status()
            assert before==imported_snapshot(FINAL),'Imported experiment changed'
            write_json(FINAL/'runtime/new_models_execution.json',records)
    conn=sqlite3.connect(FINAL/'checkpoint.sqlite')
    statuses=dict(conn.execute('SELECT status,count(*) FROM predictions GROUP BY status')); conn.close()
    summary={'started_at':started,'finished_at':now(),'wall_seconds':time.perf_counter()-start,'models':records,
             'final_statuses':statuses,'imported_predictions_and_attempts_unchanged':True,'imported_hashes':before,'paid_api_calls':0}
    write_json(FINAL/'runtime/new_models_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': asyncio.run(execute())
