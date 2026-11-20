"""Run the anchored 10-case gate and full 100-case matrix, with one local model at a time."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

import requests
from project_paths import ROOT
from prompt_benchmark.core import (connect, dump, load_plan, make_plan, now, sha,
                                   validate_prediction, write_json)
from prompt_benchmark.runner import run
from prompt_benchmark.evaluate import evaluate


def snapshot(output, model_id, ids):
    conn = connect(output)
    try:
        rows = [dict(r) for r in conn.execute('SELECT * FROM predictions WHERE model_id=? ORDER BY experiment_id,case_id', (model_id,)) if r['case_id'] in ids]
        # Hash immutable attempt metadata without reading multi-megabyte image payloads.
        attempts = [tuple(r) for r in conn.execute('''SELECT a.id,a.experiment_id,a.case_id,a.request_hash,a.error
            FROM attempts a JOIN predictions p ON a.experiment_id=p.experiment_id AND a.case_id=p.case_id
            WHERE p.model_id=? ORDER BY a.id''', (model_id,)) if r['case_id'] in ids]
        return rows, sha(rows), sha(attempts)
    finally:
        conn.close()


async def measured(plan, output, model, ids, phase):
    start = time.perf_counter()
    print(dump({'event':'start','phase':phase,'model':model['id'],'cases':len(ids),'at':now()}), flush=True)
    task = asyncio.create_task(run(plan, output, model_ids=[model['id']], case_ids=ids, workers=1, retries=1))
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=15)
        if not done:
            conn = sqlite3.connect(output / 'checkpoint.sqlite')
            counts = conn.execute('SELECT prompt_variant,modality,status,count(*) FROM predictions WHERE model_id=? GROUP BY 1,2,3', (model['id'],)).fetchall()
            conn.close()
            update = {'event':'progress','phase':phase,'model':model['id'],'seconds':time.perf_counter()-start,'counts':counts}
            write_json(output / 'progress.json', update)
            print(dump(update), flush=True)
    counts = await task
    result = {'counts':counts,'wall_seconds':time.perf_counter()-start}
    print(dump({'event':'finish','phase':phase,'model':model['id'],**result}), flush=True)
    return result


async def verify_resume(plan, output, model, ids):
    before = snapshot(output, model['id'], ids)
    assert len(before[0]) == len(ids)*8
    statuses = Counter(r['status'] for r in before[0])
    for row in before[0]:
        if row['status'] == 'completed':
            validate_prediction(row['prediction_json'], plan['categories'])
        elif row['status'] == 'skipped':
            assert not model['supports_photos'] and 'photos' in row['modality']
    if statuses.get('failed'):
        return {'passed':False,'statuses':dict(statuses),'errors':[r['error'] for r in before[0] if r['status']=='failed']}
    resumed = await run(plan, output, model_ids=[model['id']], case_ids=ids, workers=1, retries=0)
    after = snapshot(output, model['id'], ids)
    assert before == after, 'Resume changed completed predictions or attempt history'
    assert resumed == {'unchanged':len(ids)*8}
    return {'passed':True,'statuses':dict(statuses),'resume':resumed}


async def execute(output):
    start, started = time.perf_counter(), now()
    if (output / 'plan.json').exists():
        plan = load_plan(output)
    else:
        plan = make_plan(ROOT / 'results/blinded_benchmark/model_inputs.csv',
            ROOT / 'scripts/prompt_benchmark/models.multimodal.json',output,blinded=True,
            photo_manifest=ROOT / 'outputs/prompt_benchmark/photo_cache_100/manifest.json',
            anchor_path=ROOT / 'outputs/prompt_benchmark/qwen_blinded_100')
    assert all(m['adapter']=='ollama' for m in plan['config']['models'])
    assert len(plan['cases'])==100 and all(c['weak_misroute'] is None and c['verified']=={} for c in plan['cases'])
    pilot_ids={c['id'] for c in plan['cases'][:10]}
    all_ids={c['id'] for c in plan['cases']}
    results=[]
    for model in plan['config']['models']:
        try:
            execution=await measured(plan,output,model,pilot_ids,'smoke_10')
            verification=await verify_resume(plan,output,model,pilot_ids)
            results.append({'model':model['id'],'smoke':execution,'verification':verification})
        except Exception as exc:
            results.append({'model':model['id'],'verification':{'passed':False,'error':f'{type(exc).__name__}: {exc}'}})
        write_json(output/'model_execution.json',results)
        requests.post('http://127.0.0.1:11434/api/generate',json={'model':model['model'],'keep_alive':0},timeout=60).raise_for_status()
    evaluate(plan,output,evaluation_path=ROOT/'results/category_matched_common_support_with_photos.csv',
             case_ids=pilot_ids,report_name='smoke_report')
    for model, record in zip(plan['config']['models'],results):
        if not record['verification']['passed']:
            continue
        try:
            record['full']=await measured(plan,output,model,all_ids,'full_100')
            record['full_verification']=await verify_resume(plan,output,model,all_ids)
            record['runtime']=requests.get('http://127.0.0.1:11434/api/ps',timeout=15).json()
        except Exception as exc:
            record['full_error']=f'{type(exc).__name__}: {exc}'
        write_json(output/'model_execution.json',results)
        requests.post('http://127.0.0.1:11434/api/generate',json={'model':model['model'],'keep_alive':0},timeout=60).raise_for_status()
    evaluate(plan,output,evaluation_path=ROOT/'results/category_matched_common_support_with_photos.csv')
    summary={'started_at':started,'finished_at':now(),'wall_seconds':time.perf_counter()-start,
             'models':results,'paid_api_calls':0,'anchor_plan_id':plan['anchor_plan_id']}
    write_json(output/'execution_summary.json',summary)
    print(dump({'event':'finished','wall_seconds':summary['wall_seconds'],'models':results}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/prompt_benchmark/multimodal_v2')
    args=parser.parse_args()
    asyncio.run(execute(args.output))
