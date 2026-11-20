"""Bounded asynchronous execution with raw attempt audits and resumable cells."""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import time

from .core import (bind_plan, connect, dump, experiments, now, request_for, sha,
                   validate_prediction)
from .cloud import RateLimitedHTTP, assert_blinded_request
from . import local_transport


class AdapterError(RuntimeError):
    def __init__(self, message, raw=None):
        super().__init__(message)
        self.raw = raw


class MockAdapter:
    """Synthetic plumbing checks only; never a source of model-quality evidence."""

    async def generate(self, model, request):
        user = json.loads(request['user'])
        text = user['DESCRIPTION'].lower()
        category = next((category for word, category in (
            ('graffiti', 'Graffiti'), ('mattress', 'Illegal Dumping'),
            ('pothole', 'Pothole'), ('lamp', 'Street Light')) if word in text), None)
        if category not in request['schema']['properties']['recommended_category']['enum']:
            category = None
        prediction = {'misroute': category != user['ORIGINAL_TYPE'] if category else None,
                      'recommended_category': category, 'reason': 'Synthetic offline fixture.'}
        raw = dump({'synthetic': True, 'message': {'content': dump(prediction)}})
        return {'content': dump(prediction), 'raw': raw,
                'input_tokens': max(1, len(request['system'] + request['user']) // 4),
                'output_tokens': max(1, len(dump(prediction)) // 4)}


class OllamaAdapter:
    """Use the fixed-loopback benchmark transport; never initialize or pull models."""

    async def generate(self, model, request):
        assert_blinded_request(request)
        options = {'temperature': 0, 'seed': 42, 'num_ctx': 16384, 'num_predict': 512}
        options.update(model.get('options', {}))
        user = {'role': 'user', 'content': request['user']}
        if request['images']:
            user['images'] = request['images']
        payload = {'model': model['model'], 'messages': [
            {'role': 'system', 'content': request['system']}, user],
            'format': request['schema'], 'stream': False, 'options': options,
            'keep_alive': model.get('keep_alive', '30m')}
        if model.get('supports_thinking', model['model'].startswith('qwen3:')):
            payload['think'] = False
        response = await asyncio.to_thread(local_transport.api, '/api/chat', payload, timeout=600)
        raw = dump(response)
        # Return raw before validation so malformed/truncated outputs and their usage
        # are retained and charged in the attempt audit too.
        return {'content': response.get('message', {}).get('content', ''), 'raw': raw,
                'input_tokens': response.get('prompt_eval_count'),
                'output_tokens': response.get('eval_count'),
                'transport_error': ('Truncated or incomplete response' if not response.get('done')
                                    or response.get('done_reason') == 'length' else
                                    'Context capacity exceeded' if
                                    response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
                                    >= options['num_ctx'] - 8 else None)}


ADAPTERS = {'mock': MockAdapter, 'ollama': OllamaAdapter}


async def preflight(conn, model):
    if model['adapter'] != 'ollama':
        return
    tags = await asyncio.to_thread(local_transport.api, '/api/tags')
    installed = next((m for m in tags['models'] if m['name'] == model['model']), None)
    if not installed:
        raise ValueError(f'{model["model"]} is not installed; this runner never downloads models')
    show = await asyncio.to_thread(local_transport.api, '/api/show', {'model': model['model']})
    if model['model'].endswith((':cloud','-cloud')) or any(
            installed.get(k) or show.get(k) for k in ('remote_model','remote_host')):
        raise ValueError('Ollama cloud-backed models are forbidden in the local adapter')
    if model['supports_photos'] and 'vision' not in show.get('capabilities', []):
        raise ValueError(f'{model["id"]} declares photos but installed model has no vision capability')
    fingerprint = dump({'digest': installed['digest'], 'capabilities': show.get('capabilities'),
                        'ollama_version': await asyncio.to_thread(local_transport.api, '/api/version')})
    key = 'model_fingerprint:' + model['id']
    old = conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    if old and old[0] != fingerprint:
        raise ValueError('Installed model/runtime changed; create a new plan')
    with conn:
        conn.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', (key, fingerprint))


def token_count(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError('Adapter returned an invalid token count')
    return value


def cost(model, usage):
    pricing = model.get('pricing', {})
    rates = [pricing.get('input_per_million'), pricing.get('output_per_million')]
    if None in rates:
        return None
    if rates == [0, 0]:
        return 0.0
    counts = [usage.get('input_tokens'), usage.get('output_tokens')]
    if None in counts:
        return None
    return sum(a * b for a, b in zip(rates, counts)) / 1_000_000


async def run(plan, output, *, workers=4, retries=1, model_ids=None, variants=None,
              modalities=None, batches=None, experiment_ids=None, failed_only=False,
              adapters=None, case_ids=None, allow_paid=False):
    if workers < 1 or retries < 0:
        raise ValueError('workers must be positive and retries nonnegative')
    if adapters is None:
        cloud = RateLimitedHTTP(authorized=allow_paid)
        adapters = {**{name: cls() for name, cls in ADAPTERS.items()},
                    **{name: cloud for name in ('openai', 'gemini', 'anthropic')}}
    selected = [e for e in experiments(plan)
                if (not model_ids or e['model']['id'] in model_ids)
                and (not variants or e['variant'] in variants)
                and (not modalities or e['modality'] in modalities)
                and (not experiment_ids or e['id'] in experiment_ids)]
    if not selected:
        raise ValueError('No matching experiments')
    if any(e['model']['adapter'] != 'mock' for e in selected) and plan.get('input_policy') != 'blinded':
        raise ValueError('Real inference now requires a blinded plan; current-visible case inputs are forbidden')
    if not allow_paid and any(e['model']['adapter'] in ('openai', 'gemini', 'anthropic') for e in selected):
        raise PermissionError('Paid API execution is disabled; select local models or obtain explicit authorization')
    conn = connect(output)
    counts = Counter()
    try:
        bind_plan(conn, plan)
        existing = {(r['experiment_id'], r['case_id']): r for r in conn.execute('SELECT * FROM predictions')}
        jobs = []
        for experiment in selected:
            for case in plan['cases']:
                if case_ids is not None and case['id'] not in case_ids:
                    continue
                if batches is not None and case['batch'] not in batches:
                    continue
                old = existing.get((experiment['id'], case['id']))
                if old and old['input_hash'] != sha(case):
                    raise ValueError('Checkpoint input association mismatch')
                if old and old['status'] in ('completed', 'skipped'):
                    counts['unchanged'] += 1
                    continue
                if failed_only and (not old or old['status'] != 'failed'):
                    continue
                jobs.append((experiment, case))
        if not jobs and not counts:
            raise ValueError('No matching cases or failed predictions')
        for model in {e['model']['id']: e['model'] for e, _ in jobs
                      if 'photos' not in e['modality'] or e['model']['supports_photos']}.values():
            await preflight(conn, model)

        def save(experiment, case, status, prediction=None, error=None):
            prompt = plan['prompts'][experiment['variant']]
            with conn:
                conn.execute('''INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(experiment_id,case_id) DO UPDATE SET
                    status=excluded.status,prediction_json=excluded.prediction_json,
                    error=excluded.error,completed_at=excluded.completed_at''',
                    (experiment['id'], case['id'], case['batch'], experiment['model']['id'],
                     experiment['modality'], experiment['variant'], prompt['version'], prompt['hash'],
                     sha(case), status, dump(prediction) if prediction is not None else None, error, now()))
            counts[status] += 1

        async def execute(experiment, case):
            model = experiment['model']
            if 'photos' in experiment['modality'] and not model['supports_photos']:
                save(experiment, case, 'skipped', error='Model does not support photographs')
                return
            for _ in range(retries + 1):
                started = now()
                tick, elapsed = None, 0.0
                request, result, error, prediction = None, {}, None, None
                try:
                    request = request_for(plan, experiment, case)
                    tick = time.perf_counter()
                    result = await adapters[model['adapter']].generate(model, request)
                    elapsed = result.get('latency_seconds', time.perf_counter() - tick)
                    result['input_tokens'] = token_count(result.get('input_tokens'))
                    result['output_tokens'] = token_count(result.get('output_tokens'))
                    if result.get('transport_error'):
                        raise AdapterError(result['transport_error'])
                    prediction = validate_prediction(result['content'], plan['categories'])
                except Exception as exc:
                    if tick is not None and elapsed == 0:
                        elapsed = time.perf_counter() - tick
                    error = f'{type(exc).__name__}: {exc}'
                    if getattr(exc, 'raw', None) is not None:
                        result['raw'] = exc.raw
                    elif getattr(exc, 'response', None) is not None:
                        result['raw'] = exc.response.text
                    for field in ('input_tokens', 'output_tokens'):
                        value = result.get(field)
                        if type(value) is not int or value < 0:
                            result[field] = None
                with conn:
                    conn.execute('''INSERT INTO attempts(experiment_id,case_id,started_at,
                        latency_seconds,request_hash,request_json,raw_response,input_tokens,
                        output_tokens,cost_usd,error) VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                        (experiment['id'], case['id'], started, elapsed,
                         sha(request) if request else None, dump(request) if request else None,
                         result.get('raw'), result.get('input_tokens'), result.get('output_tokens'),
                         cost(model, result), error))
                if not error:
                    save(experiment, case, 'completed', prediction)
                    return
            save(experiment, case, 'failed', error=error)

        # Model groups run sequentially. Local requests share one GPU slot even if
        # multiple async workers are requested, avoiding VRAM conflicts/model thrash.
        for model in plan['config']['models']:
            model_jobs = [(e, c) for e, c in jobs if e['model']['id'] == model['id']]
            iterator = iter(model_jobs)
            async def worker():
                for experiment, case in iterator:
                    await execute(experiment, case)
            concurrency = min(workers, len(model_jobs), 1 if model['adapter'] == 'ollama' else
                              model.get('max_concurrency', workers))
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        return dict(counts)
    finally:
        conn.close()
