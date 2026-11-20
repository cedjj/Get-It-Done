"""Immutable plans, common prompt/schema contract, and isolated SQLite checkpoints."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).parent / 'templates'
MODALITIES = ('text', 'text_location', 'text_photos', 'text_location_photos')
VARIANTS = ('simple', 'detailed')
PLAN_VERSION = 'paired-routing-v1'


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else dump(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def schema(categories):
    return {
        'type': 'object',
        'properties': {
            'misroute': {'type': ['boolean', 'null']},
            'recommended_category': {'type': ['string', 'null'], 'enum': [None] + list(categories)},
            'reason': {'type': 'string'},
        },
        'required': ['misroute', 'recommended_category', 'reason'],
        'additionalProperties': False,
    }


def validate_prediction(raw, categories):
    prediction = json.loads(raw)
    if not isinstance(prediction, dict) or set(prediction) != {'misroute', 'recommended_category', 'reason'}:
        raise ValueError('Response must contain exactly the shared schema fields')
    if prediction['misroute'] is not None and type(prediction['misroute']) is not bool:
        raise ValueError('misroute must be a boolean or null')
    if prediction['recommended_category'] is not None and prediction['recommended_category'] not in categories:
        raise ValueError('recommended_category is not a valid category')
    if not isinstance(prediction['reason'], str) or not prediction['reason'].strip():
        raise ValueError('reason must be a nonempty string')
    return prediction


def parse_bool(value, field):
    if value in ('', None):
        return None
    if str(value).strip().lower() in ('true', '1'):
        return True
    if str(value).strip().lower() in ('false', '0'):
        return False
    raise ValueError(f'{field} must be true/false/1/0 or blank')


def allowed_blinded_column(name):
    return name in {'ID', 'DESCRIPTION', 'ORIGINAL_TYPE', 'GEOLOCATION__LATITUDE__S',
                    'GEOLOCATION__LONGITUDE__S'} or bool(re.fullmatch(r'PUBLIC_PHOTO_URL_\d+', name))


def read_cases(path, *, blinded=False):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'ID', 'DESCRIPTION', 'ORIGINAL_TYPE'}
        if not blinded:
            required.add('CURRENT_TYPE')
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Cases require columns {sorted(required)}')
        if blinded and any(not allowed_blinded_column(k) for k in reader.fieldnames):
            raise ValueError('Blinded CSV contains forbidden fields; remove CURRENT_TYPE, labels and downstream metadata')
        rows = list(reader)
    ids = [r['ID'].strip() for r in rows]
    if not ids or not all(ids) or len(set(ids)) != len(ids):
        raise ValueError('Case IDs must be nonempty and unique')
    return rows


def read_verified(path, categories):
    if not path:
        return {}
    verified = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            case_id = row['ID'].strip()
            if not case_id or case_id in verified:
                raise ValueError('Verified IDs must be nonempty and unique')
            if parse_bool(row.get('independently_verified'), 'independently_verified') is not True:
                raise ValueError('Verified file requires independently_verified=true on every row')
            if not row.get('verification_provenance', '').strip():
                raise ValueError('Verified labels require verification_provenance')
            category = row.get('verified_category', '').strip() or None
            if category is not None and category not in categories:
                raise ValueError(f'Unknown verified category: {category}')
            flag = parse_bool(row.get('verified_misroute'), 'verified_misroute')
            if flag is None and category is None:
                raise ValueError('Verified row has no label')
            verified[case_id] = {'misroute': flag, 'category': category,
                                 'provenance': row['verification_provenance']}
    return verified


def location(row):
    lat = row.get('GEOLOCATION__LATITUDE__S', '').strip()
    lon = row.get('GEOLOCATION__LONGITUDE__S', '').strip()
    if not lat and not lon:
        return None
    try:
        lat, lon = float(lat), float(lon)
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError()
        return {'latitude': lat, 'longitude': lon}
    except ValueError:
        raise ValueError(f'Invalid coordinates for case {row["ID"]}') from None


def photo_sources(row):
    # Use only original/public photo columns; never closure/after photos.
    keys = sorted((k for k in row if re.fullmatch(r'PUBLIC_PHOTO_URL_\d+', k)),
                  key=lambda k: int(k.rsplit('_', 1)[1]))
    return list(dict.fromkeys(row[k].strip() for k in keys if row[k].strip()))


def load_models(path):
    config = read_json(path)
    models = config['models']
    if not models or len({m['id'] for m in models}) != len(models):
        raise ValueError('Model IDs must be nonempty and unique')
    for model in models:
        if not model['id'] or model['adapter'] not in ('mock', 'ollama', 'openai', 'gemini', 'anthropic'):
            raise ValueError('Supported adapters: mock, ollama, openai, gemini, anthropic')
        if type(model.get('supports_photos')) is not bool:
            raise ValueError('Each model must declare supports_photos')
        if model['adapter'] != 'mock' and not model.get('model'):
            raise ValueError('Model name required')
        for key in ('requests_per_minute','max_concurrency','max_output_tokens'):
            value=model.get(key)
            if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<=0):
                raise ValueError(f'{key} must be a positive finite number')
            if value is not None and key!='requests_per_minute' and type(value) is not int:
                raise ValueError(f'{key} must be an integer')
        for key in ('input_per_million', 'output_per_million'):
            rate = model.get('pricing', {}).get(key)
            if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0):
                raise ValueError('Token prices must be nonnegative finite numbers')
    return config


def make_plan(cases_path, config_path, output, *, reference_path=None, definitions_path=None,
              verified_path=None, photo_manifest=None, limit=None, seed=42, batch_size=100,
              blinded=False, anchor_path=None):
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError('batch_size and limit must be positive')
    output = Path(output)
    if (output / 'plan.json').exists():
        raise ValueError('Plan already exists; reuse it to run/report, or choose a new directory')
    config = load_models(config_path)
    ref = read_json(reference_path or ROOT / 'outputs/semantic_review/service_definitions.json')
    categories = [r['category'] for r in ref['categories']]
    if not categories or len(set(categories)) != len(categories):
        raise ValueError('Reference must have unique categories')
    aliases = {a: r['category'] for r in ref['categories'] for a in [r['category']] + r.get('aliases', [])}
    # No Markdown definitions are present in the legacy workspace. Render its existing
    # reference losslessly; a supplied Markdown document is included verbatim instead.
    definitions = (Path(definitions_path).read_text(encoding='utf-8-sig') if definitions_path else
                   '\n\n'.join(f'## {r["category"]}\nStatus: {r["definition_status"]}\n{r["definition"]}'
                               for r in ref['categories']))
    prompts = {}
    for variant in VARIANTS:
        setting = config.get('prompts', {}).get(variant, {})
        template_path = (Path(config_path).resolve().parent / setting['template']
                         if setting.get('template') else TEMPLATES / f'{variant}.txt')
        template = template_path.read_text(encoding='utf-8-sig')
        if blinded and 'CURRENT_TYPE' in template:
            raise ValueError('Blinded prompt must not reference CURRENT_TYPE; use models.blinded.json')
        if '{categories}' not in template or (variant == 'detailed' and '{definitions}' not in template):
            raise ValueError(f'{variant} template is missing required placeholders')
        rendered = template.replace('{categories}', dump(categories)).replace('{definitions}', definitions)
        prompts[variant] = {'version': setting.get('version', f'{variant}-v1'), 'template': template,
                            'system': rendered, 'hash': sha(rendered.encode())}
    if blinded and verified_path:
        raise ValueError('Blinded inference plans cannot load evaluation labels; attach them only when reporting')
    verified = read_verified(verified_path, categories)
    all_rows = read_cases(cases_path, blinded=blinded)
    if set(verified) - {r['ID'].strip() for r in all_rows}:
        raise ValueError('Verified file includes IDs absent from the case source')
    rows = sorted(all_rows, key=lambda r: sha([seed, r['ID'].strip()]))
    anchor = load_plan(anchor_path) if anchor_path else None
    if anchor:
        if not blinded or anchor.get('input_policy') != 'blinded':
            raise ValueError('Anchored multimodal experiments require blinded plans')
        lookup = {r['ID'].strip(): r for r in all_rows}
        if any(c['id'] not in lookup for c in anchor['cases']):
            raise ValueError('Anchor case is missing from input CSV')
        rows = [lookup[c['id']] for c in anchor['cases']]
        for case, row in zip(anchor['cases'], rows):
            if {k: row[k] for k in case['inputs']} != case['inputs'] or location(row) != case['location']:
                raise ValueError('Anchor case inputs changed')
        if limit is not None and limit != len(rows):
            raise ValueError('Do not truncate or resample an anchored batch; use run --case for smoke tests')
    if limit is not None:
        rows = rows[:limit]
    cache = read_json(photo_manifest) if photo_manifest else {}
    cases = []
    for i, row in enumerate(rows):
        photos = []
        for url in photo_sources(row):
            item = {'source': url, 'path': None, 'sha256': None}
            if url in cache:
                path = (Path(photo_manifest).resolve().parent / cache[url]).resolve()
                blob = path.read_bytes()
                if not blob or len(blob) > 20 * 1024 * 1024:
                    raise ValueError('Cached photos must be nonempty and at most 20 MiB')
                item.update(path=str(path), sha256=sha(blob))
            photos.append(item)
        cases.append({'id': row['ID'].strip(), 'batch': i // batch_size,
                      'inputs': {k: row[k] for k in (('DESCRIPTION', 'ORIGINAL_TYPE') if blinded else
                                                   ('DESCRIPTION', 'ORIGINAL_TYPE', 'CURRENT_TYPE'))},
                      'location': location(row), 'photos': photos,
                      'weak_misroute': parse_bool(row.get('label'), 'label'),
                      'weak_provenance': row.get('LABEL_PROVENANCE', ''),
                      'verified': verified.get(row['ID'].strip(), {})})
        if blinded:
            cases[-1]['evaluation_join_hash'] = sha(row)
    plan = {'version': PLAN_VERSION, 'seed': seed, 'batch_size': batch_size,
            'source_sha256': sha(Path(cases_path).read_bytes()), 'config': config,
            'reference_sha256': sha(ref), 'definitions_sha256': sha(definitions.encode()),
            'aliases': aliases, 'categories': categories, 'schema': schema(categories),
            'prompts': prompts, 'modalities': list(MODALITIES), 'cases': cases,
            'evaluation_note': 'CURRENT_TYPE is an experimental input, never an independently verified target. '
                               'Proxy labels and exported geographic/photo provenance remain unverified.'}
    if blinded:
        plan['input_policy'] = 'blinded'
        plan['input_columns'] = list(all_rows[0])
        plan['evaluation_note'] = ('CURRENT_TYPE and misroute labels are hidden from inference and joined only '
                                   'for scoring by case ID plus input hash. Proxy labels and exported '
                                   'description/location/photo timing remain unverified.')
    if anchor:
        plan['anchor_plan_id'] = anchor['id']
        if [c['id'] for c in cases] != [c['id'] for c in anchor['cases']]:
            raise ValueError('Anchored batch ordering changed')
    plan['id'] = sha(plan)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    for variant in VARIANTS:
        (output / f'{variant}_prompt.txt').write_text(prompts[variant]['system'], encoding='utf-8')
    return plan


def load_plan(output):
    plan = read_json(Path(output) / 'plan.json')
    body = {k: v for k, v in plan.items() if k != 'id'}
    if plan.get('id') != sha(body):
        raise ValueError('Plan content changed; create a new plan rather than mixing checkpoints')
    return plan


def experiments(plan):
    for model in plan['config']['models']:
        for modality in plan['modalities']:
            for variant in VARIANTS:
                experiment_id = sha([plan['id'], model, modality, variant, plan['prompts'][variant]['hash']])
                yield {'id': experiment_id, 'model': model, 'modality': modality, 'variant': variant}


def request_for(plan, experiment, case):
    allowed = {'DESCRIPTION', 'ORIGINAL_TYPE'}
    if plan.get('input_policy') != 'blinded':
        allowed.add('CURRENT_TYPE')
    if set(case['inputs']) != allowed:
        raise ValueError('Unexpected model-input fields; labels and downstream fields are forbidden')
    user = {k: case['inputs'][k] for k in allowed}
    photos = []
    if 'location' in experiment['modality']:
        user['LOCATION'] = case['location']
    if 'photos' in experiment['modality']:
        user['PHOTOS'] = {'available': len(case['photos']), 'status': 'provided' if case['photos'] else 'not_available'}
        for photo in case['photos']:
            if not photo['path']:
                raise ValueError('Photo input is not cached; supply a photo manifest and create a new plan')
            blob = Path(photo['path']).read_bytes()
            if sha(blob) != photo['sha256']:
                raise ValueError('Cached photo hash changed')
            photos.append(base64.b64encode(blob).decode('ascii'))
    prompt = plan['prompts'][experiment['variant']]
    # Both prompts share this exact schema and allowlisted user data. IDs/labels/URLs
    # and downstream closure fields never enter the model request.
    return {'system': prompt['system'], 'user': dump(user), 'images': photos, 'schema': plan['schema']}


def connect(output):
    conn = sqlite3.connect(Path(output) / 'checkpoint.sqlite', timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=FULL')
    conn.executescript('''
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS predictions(
        experiment_id TEXT NOT NULL, case_id TEXT NOT NULL, batch INTEGER NOT NULL,
        model_id TEXT NOT NULL, modality TEXT NOT NULL, prompt_variant TEXT NOT NULL,
        prompt_version TEXT NOT NULL, prompt_hash TEXT NOT NULL, input_hash TEXT NOT NULL,
        status TEXT NOT NULL, prediction_json TEXT, error TEXT, completed_at TEXT NOT NULL,
        PRIMARY KEY(experiment_id, case_id));
      CREATE TABLE IF NOT EXISTS attempts(
        id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, case_id TEXT NOT NULL,
        started_at TEXT NOT NULL, latency_seconds REAL NOT NULL, request_hash TEXT,
        request_json TEXT, raw_response TEXT, input_tokens INTEGER, output_tokens INTEGER,
        cost_usd REAL, error TEXT);
    ''')
    return conn


def bind_plan(conn, plan):
    old = conn.execute("SELECT value FROM meta WHERE key='plan_id'").fetchone()
    if old and old[0] != plan['id']:
        raise ValueError('Checkpoint belongs to a different plan')
    with conn:
        conn.execute('INSERT OR IGNORE INTO meta VALUES (?,?)', ('plan_id', plan['id']))
