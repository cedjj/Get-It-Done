"""Verify attachment ownership, cache original bytes, and normalize model image inputs."""
from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
import time
from urllib.parse import parse_qs, urljoin, urlparse

from PIL import Image, ImageOps
import requests

from .core import ROOT, load_plan, now, read_json, sha, write_json


def attachment_id(url):
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.hostname != 'getitdone.sandiego.gov'
            or parsed.path != '/servlet/servlet.FileDownload'):
        raise ValueError('Unexpected public attachment URL')
    values = parse_qs(parsed.query).get('file', [])
    if len(values) != 1 or not values[0].isalnum() or len(values[0]) not in (15, 18):
        raise ValueError('Invalid attachment ID')
    return values[0]


def verified_links(cases_path, attachment_export, selected_ids):
    with Path(cases_path).open(encoding='utf-8-sig', newline='') as handle:
        cases = {r['ID']: r for r in csv.DictReader(handle) if r['ID'] in selected_ids}
    if set(cases) != set(selected_ids):
        raise ValueError('Selected case IDs are missing from photo source')
    expected = {}
    for case in cases.values():
        urls = json.loads(case['PUBLIC_ATTACHMENT_URLS_JSON'])
        ids = json.loads(case['PUBLIC_ATTACHMENT_IDS_JSON'])
        names = json.loads(case['PUBLIC_ATTACHMENT_FILENAMES_JSON'])
        if not len(urls) == len(ids) == len(names):
            raise ValueError('Attachment array lengths disagree')
        for url, file_id, name in zip(urls, ids, names):
            if attachment_id(url) != file_id or file_id in expected:
                raise ValueError('Duplicate attachment ownership or URL/ID mismatch')
            expected[file_id] = {'case_id': case['ID'], 'case_number': str(int(case['CASENUMBER'])),
                                 'filename': name, 'source': url}
    seen = {}
    with Path(attachment_export).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['Id'] not in expected:
                continue
            item = expected[row['Id']]
            if (row['ParentId'] != item['case_id'] or str(int(row['Parent.Name'])) != item['case_number']
                    or row['Name'] != item['filename']):
                raise ValueError('Attachment parent ID, case number or filename mismatch')
            if row['Id'] in seen and seen[row['Id']] != row:
                raise ValueError('Conflicting duplicate attachment metadata')
            seen[row['Id']] = row
    if set(expected) != set(seen):
        raise ValueError('Attachment IDs absent from original export')
    return expected


def trusted_redirect(url):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    return parsed.scheme == 'https' and (host == 'getitdone.sandiego.gov' or
        host.endswith(('.force.com', '.salesforce.com', '.salesforce-sites.com', '.sfdcstatic.com')))


def download_image(item, directory):
    file_id = attachment_id(item['source'])
    original = directory / 'originals' / f'{file_id}.bin'
    normalized = directory / 'images' / f'{file_id}.jpg'
    meta_path = directory / 'metadata' / f'{file_id}.json'
    if meta_path.exists() and original.exists() and normalized.exists():
        saved = read_json(meta_path)
        if sha(original.read_bytes()) == saved['original_sha256'] and sha(normalized.read_bytes()) == saved['sha256']:
            if saved['case_id'] != item['case_id']:
                raise ValueError('Cached photo belongs to a different case')
            return saved
    current = item['source']
    headers = {'User-Agent': 'GetItDoneResearchBenchmark/2.0'}
    for _ in range(6):
        if not trusted_redirect(current):
            raise ValueError('Untrusted attachment redirect')
        with requests.get(current, headers=headers, stream=True, allow_redirects=False,
                          timeout=(15, 60)) as response:
            if response.is_redirect:
                current = urljoin(current, response.headers['Location'])
                continue
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 20 * 1024**2:
                    raise ValueError('Photo exceeds 20 MiB download limit')
                chunks.append(chunk)
            blob = b''.join(chunks)
        break
    else:
        raise ValueError('Too many attachment redirects')
    with Image.open(io.BytesIO(blob)) as image:
        image.verify()
    with Image.open(io.BytesIO(blob)) as image:
        dimensions, original_format = image.size, image.format
        if image.width * image.height > 50_000_000:
            raise ValueError('Photo dimensions exceed the normalization limit')
        image = ImageOps.exif_transpose(image)
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image = image.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        model_blob = buffer.getvalue()
        normalized_dimensions = image.size
    original.write_bytes(blob)
    normalized.write_bytes(model_blob)
    result = {**item, 'attachment_id': file_id, 'retrieved_at': now(), 'status': 'cached',
              'final_url': current, 'content_type': content_type, 'original_format': original_format,
              'original_dimensions': dimensions, 'normalized_dimensions': normalized_dimensions,
              'original_sha256': sha(blob), 'sha256': sha(model_blob),
              'path': str(normalized.resolve()), 'case_ownership_verified': True,
              'submission_time_verified': False}
    write_json(meta_path, result)
    return result


async def cache_photos(anchor, output, *, cases_path=None, attachment_export=None, workers=4):
    anchor, output = Path(anchor), Path(output)
    plan = load_plan(anchor)
    if plan.get('input_policy') != 'blinded':
        raise ValueError('Photo benchmark must anchor to a blinded plan')
    selected = {c['id'] for c in plan['cases']}
    links = verified_links(cases_path or ROOT / 'results/category_matched_common_support_with_photos.csv',
                           attachment_export or ROOT / 'Get_it_done_data/public-attachments-cases.csv', selected)
    for name in ('originals', 'images', 'metadata'):
        (output / name).mkdir(parents=True, exist_ok=True)
    records, semaphore = [], asyncio.Semaphore(workers)

    async def fetch(item):
        async with semaphore:
            for attempt in range(3):
                try:
                    record = await asyncio.to_thread(download_image, item, output)
                    records.append(record)
                    break
                except Exception as exc:
                    if attempt == 2:
                        records.append({**item, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
                    else:
                        await asyncio.sleep(2 ** attempt)
            if len(records) % 10 == 0 or len(records) == len(links):
                print(json.dumps({'event': 'photo_progress', 'finished': len(records), 'total': len(links),
                                  'failed': sum(r['status'] == 'failed' for r in records)}), flush=True)
            write_json(output / 'download_audit.json', records)
            write_json(output / 'manifest.json', {r['source']: r['path'] for r in records if r['status'] == 'cached'})

    await asyncio.gather(*(fetch(item) for item in links.values()))
    summary = {'anchor_plan_id': plan['id'], 'cases': len(selected), 'attachments': len(links),
               'cached': sum(r['status'] == 'cached' for r in records),
               'failed': sum(r['status'] == 'failed' for r in records),
               'ownership_check': 'Original attachment ParentId + Parent.Name + attachment ID + filename',
               'normalization': 'EXIF orientation; longest side <=1024; RGB JPEG quality 90; metadata stripped',
               'limitations': 'Case ownership is verified. The export has no upload timestamps or before/after role; submission-time availability is unverified.'}
    write_json(output / 'summary.json', summary)
    return summary
