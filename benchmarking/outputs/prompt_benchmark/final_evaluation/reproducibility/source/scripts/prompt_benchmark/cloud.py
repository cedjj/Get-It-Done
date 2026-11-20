"""Optional paid adapters: disabled until explicit authorization is passed by caller."""
from __future__ import annotations
import asyncio
import base64
import json
import os
import hashlib
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import requests

from .core import dump


def assert_blinded_request(request):
    user = json.loads(request['user'])
    if (not {'DESCRIPTION', 'ORIGINAL_TYPE'} <= set(user) or
            not set(user) <= {'DESCRIPTION', 'ORIGINAL_TYPE', 'LOCATION', 'PHOTOS'}):
        raise ValueError('Real adapters accept only blinded, allowlisted case fields')
    if not all(isinstance(user[k],str) for k in ('DESCRIPTION','ORIGINAL_TYPE')):
        raise ValueError('Complaint and original type must be strings')
    if user.get('LOCATION') is not None:
        if not isinstance(user['LOCATION'],dict) or set(user['LOCATION'])!={'latitude','longitude'}:
            raise ValueError('Location may contain only latitude/longitude')
    if 'PHOTOS' in user:
        meta=user['PHOTOS']
        if (not isinstance(meta,dict) or set(meta)!={'available','status'} or type(meta['available']) is not int
                or meta['available']<0 or meta['status'] not in ('provided','not_available')):
            raise ValueError('Photo metadata may contain only availability/count')


def media_type(encoded):
    data = base64.b64decode(encoded[:64])
    if data.startswith(b'\xff\xd8'):
        return 'image/jpeg'
    if data.startswith(b'\x89PNG'):
        return 'image/png'
    if data.startswith(b'GIF8'):
        return 'image/gif'
    if data.startswith(b'RIFF'):
        return 'image/webp'
    raise ValueError('Unsupported cached image format')


def build_request(provider, model, request):
    assert_blinded_request(request)
    limit = model.get('max_output_tokens', 512)
    if provider == 'openai':
        content = [{'type': 'text', 'text': request['user']}]
        content += [{'type': 'image_url', 'image_url': {'url': f'data:{media_type(b)};base64,{b}'}}
                    for b in request['images']]
        body = {'model': model['model'], 'messages': [{'role': 'system', 'content': request['system']},
                {'role': 'user', 'content': content}], 'max_completion_tokens': limit,
                'response_format': {'type': 'json_schema', 'json_schema': {
                    'name': 'routing', 'strict': True, 'schema': request['schema']}}, 'store': False}
        return 'https://api.openai.com/v1/chat/completions', body
    if provider == 'gemini':
        parts = [{'text': request['user']}] + [{'inlineData': {'mimeType': media_type(b), 'data': b}}
                                               for b in request['images']]
        body = {'systemInstruction': {'parts': [{'text': request['system']}]},
                'contents': [{'role': 'user', 'parts': parts}],
                'generationConfig': {'temperature': 0, 'maxOutputTokens': limit,
                                     'responseMimeType': 'application/json', 'responseJsonSchema': request['schema']}}
        return f'https://generativelanguage.googleapis.com/v1beta/models/{model["model"]}:generateContent', body
    if provider == 'anthropic':
        content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': media_type(b), 'data': b}}
                   for b in request['images']] + [{'type': 'text', 'text': request['user']}]
        body = {'model': model['model'], 'max_tokens': limit, 'system': request['system'],
                'messages': [{'role': 'user', 'content': content}],
                'output_config': {'format': {'type': 'json_schema', 'schema': request['schema']}}}
        return 'https://api.anthropic.com/v1/messages', body
    raise ValueError('Unknown cloud provider')


def parse_response(provider, response):
    if provider == 'openai':
        choice = response.get('choices', [{}])[0]
        usage = response.get('usage', {})
        return {'content': choice.get('message', {}).get('content') or '',
                'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'),
                'transport_error': None if choice.get('finish_reason') == 'stop' else 'Refused or incomplete OpenAI response'}
    if provider == 'gemini':
        candidate = response.get('candidates', [{}])[0]
        usage = response.get('usageMetadata', {})
        completion = usage.get('candidatesTokenCount')
        thoughts = usage.get('thoughtsTokenCount', 0)
        return {'content': ''.join(p.get('text', '') for p in candidate.get('content', {}).get('parts', [])
                                   if not p.get('thought')),
                'input_tokens': usage.get('promptTokenCount'),
                'output_tokens': completion + thoughts if completion is not None else None,
                'transport_error': None if candidate.get('finishReason') == 'STOP' else 'Blocked or incomplete Gemini response'}
    usage = response.get('usage', {})
    # Cached input tokens are still input consumption; flat-rate estimated costs are
    # explicitly conservative when cache-specific pricing is not configured.
    count = usage.get('input_tokens')
    if count is not None:
        count += usage.get('cache_read_input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
    return {'content': ''.join(p.get('text', '') for p in response.get('content', []) if p.get('type') == 'text'),
            'input_tokens': count, 'output_tokens': usage.get('output_tokens'),
            'transport_error': None if response.get('stop_reason') == 'end_turn' else 'Refused or incomplete Anthropic response'}


class RateLimitedHTTP:
    def __init__(self, authorized=False):
        self.authorized = authorized
        self.locks, self.last = {}, {}
        self.semaphores = {}

    async def generate(self, model, request):
        if not self.authorized:
            raise PermissionError('Paid API execution is disabled; explicit user authorization is required')
        provider = model['adapter']
        env = model.get('api_key_env', {'openai': 'OPENAI_API_KEY', 'gemini': 'GEMINI_API_KEY',
                                        'anthropic': 'ANTHROPIC_API_KEY'}[provider])
        key = os.environ.get(env)
        if not key:
            raise ValueError(f'Missing API key environment variable: {env}')
        url, body = build_request(provider, model, request)
        headers = {'Content-Type': 'application/json'}
        if provider == 'openai':
            headers['Authorization'] = 'Bearer ' + key
        elif provider == 'gemini':
            headers['x-goog-api-key'] = key
        else:
            headers.update({'x-api-key': key, 'anthropic-version': '2023-06-01'})
        bucket = (provider, hashlib.sha256(key.encode()).hexdigest())
        lock = self.locks.setdefault(bucket, asyncio.Lock())
        semaphore = self.semaphores.setdefault(bucket, asyncio.Semaphore(model.get('max_concurrency', 2)))
        async with semaphore:
            async with lock:
                interval = 60 / max(1, model.get('requests_per_minute', 30))
                delay = self.last.get(bucket, 0) + interval - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                self.last[bucket] = time.monotonic()
            tick=time.perf_counter()
            response = await asyncio.to_thread(requests.post, url, headers=headers, json=body, timeout=180)
            inference_seconds=time.perf_counter()-tick
            if response.status_code == 429:
                value = response.headers.get('Retry-After', '5')
                try:
                    delay = float(value)
                except ValueError:
                    try:
                        delay = max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError,TypeError):
                        delay=5
                # Future attempts share this backoff, rather than immediately retrying.
                self.last[bucket] = max(self.last[bucket], time.monotonic() + delay)
            if not response.ok:
                # Return the provider body so runner audit preserves non-JSON failures.
                return {'content': '', 'raw': response.text.replace(key,'[REDACTED_API_KEY]'), 'latency_seconds':inference_seconds,
                        'input_tokens': None, 'output_tokens': None,
                        'transport_error': f'{provider} HTTP {response.status_code}'}
            try:
                raw = response.json()
            except ValueError:
                return {'content':'','raw':response.text.replace(key,'[REDACTED_API_KEY]'),'input_tokens':None,'output_tokens':None,'latency_seconds':inference_seconds,
                        'transport_error':f'{provider} returned invalid JSON'}
            return {**parse_response(provider, raw), 'raw': dump(raw), 'latency_seconds':inference_seconds}
