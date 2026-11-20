"""Plan/run/report paired prompt experiments; default commands never start inference."""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
from pathlib import Path

from project_paths import ROOT
from prompt_benchmark.core import (experiments, load_plan, make_plan, write_json)
from prompt_benchmark.evaluate import evaluate
from prompt_benchmark.runner import run


def smoke(output):
    """Exercise the full two-model/eight-cell matrix without network or real models."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists():
        plan = load_plan(output)
        if any(m['adapter'] != 'mock' for m in plan['config']['models']):
            raise ValueError('Smoke output contains a non-mock plan; choose a different output')
    else:
        image = output / 'fixture.png'
        image.write_bytes(base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='))
        cases = [
            {'ID': 'fixture-1', 'DESCRIPTION': 'Blue graffiti painted on wall.', 'ORIGINAL_TYPE': 'Graffiti',
             'CURRENT_TYPE': 'Graffiti', 'label': '0', 'GEOLOCATION__LATITUDE__S': '32.7',
             'GEOLOCATION__LONGITUDE__S': '-117.1', 'PUBLIC_PHOTO_URL_1': 'fixture://photo'},
            {'ID': 'fixture-2', 'DESCRIPTION': 'Mattress dumped in alley.', 'ORIGINAL_TYPE': 'Traffic Signal',
             'CURRENT_TYPE': 'Illegal Dumping', 'label': '1'},
            {'ID': 'fixture-3', 'DESCRIPTION': 'Please help.', 'ORIGINAL_TYPE': 'Curb',
             'CURRENT_TYPE': 'Curb', 'label': '1'},
        ]
        with (output / 'cases.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
        write_json(output / 'photos.json', {'fixture://photo': 'fixture.png'})
        write_json(output / 'models.json', {'models': [
            {'id': name, 'adapter': 'mock', 'supports_photos': True,
             'pricing': {'input_per_million': 0, 'output_per_million': 0}}
            for name in ('fixture-a', 'fixture-b')]})
        with (output / 'verified.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['ID', 'independently_verified', 'verification_provenance', 'verified_misroute', 'verified_category'])
            writer.writerow(['fixture-1', 'true', 'SYNTHETIC TEST FIXTURE ONLY', 'false', 'Graffiti'])
            writer.writerow(['fixture-2', 'true', 'SYNTHETIC TEST FIXTURE ONLY', 'true', 'Illegal Dumping'])
        plan = make_plan(output / 'cases.csv', output / 'models.json', output,
                         verified_path=output / 'verified.csv', photo_manifest=output / 'photos.json')
    first = asyncio.run(run(plan, output, workers=3, retries=0))
    resumed = asyncio.run(run(plan, output, workers=3, retries=0))
    expected = len(plan['cases']) * len(plan['config']['models']) * 8
    if resumed != {'unchanged': expected}:
        raise RuntimeError(f'Resume smoke failed: {resumed}')
    report = evaluate(plan, output, bootstrap=50)
    if any(row['completed'] != len(plan['cases']) for row in report['summary']):
        raise RuntimeError('Incomplete smoke matrix')
    return {'synthetic_only': True, 'expected_predictions': expected, 'first_run': first,
            'resumed': resumed, 'report': str((output / 'report/index.html').resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan', help='Freeze cases/config/prompts without inference or downloads')
    plan.add_argument('--cases', type=Path, default=ROOT / 'results/blinded_benchmark/model_inputs.csv')
    plan.add_argument('--config', type=Path, default=ROOT / 'scripts/prompt_benchmark/models.blinded.json')
    plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--reference', type=Path)
    plan.add_argument('--definitions', type=Path, help='Existing Markdown routing definitions, included verbatim')
    plan.add_argument('--verified', type=Path)
    plan.add_argument('--photo-manifest', type=Path, help='JSON mapping public photo URLs to cached file paths')
    plan.add_argument('--limit', type=int, help='Choose cases once, before crossing the experiment matrix')
    plan.add_argument('--seed', type=int, default=42)
    plan.add_argument('--batch-size', type=int, default=100)
    plan.add_argument('--blinded', action='store_true', default=True, help='Blinded inputs are required for real inference')
    plan.add_argument('--anchor', type=Path, help='Reuse exactly the cases/order of an existing blinded plan')
    for command in ('run', 'report', 'list'):
        command_parser = sub.add_parser(command)
        command_parser.add_argument('--output', type=Path, required=True)
        if command == 'run':
            command_parser.add_argument('--workers', type=int, default=4)
            command_parser.add_argument('--retries', type=int, default=1, help='Additional attempts per invocation')
            command_parser.add_argument('--model', action='append', dest='model_ids')
            command_parser.add_argument('--variant', action='append', choices=['simple', 'detailed'], dest='variants')
            command_parser.add_argument('--modality', action='append', dest='modalities', choices=[
                'text', 'text_location', 'text_photos', 'text_location_photos'])
            command_parser.add_argument('--batch', type=int, action='append', dest='batches')
            command_parser.add_argument('--experiment', action='append', dest='experiment_ids', help='Exact ID from list')
            command_parser.add_argument('--failed-only', action='store_true')
            command_parser.add_argument('--case', action='append', dest='case_ids',
                                        help='Restrict execution to selected IDs in the frozen plan')
            command_parser.add_argument('--allow-paid', action='store_true',
                                        help='Use only after explicit user authorization for paid API calls')
        elif command == 'report':
            command_parser.add_argument('--verified', type=Path,
                                        help='Independent adjudications; attached only after inference for scoring')
            command_parser.add_argument('--evaluation', type=Path,
                                        help='Original CSV or held-out labels; read only during scoring')
            command_parser.add_argument('--bootstrap', type=int, default=1000)
            command_parser.add_argument('--min-verified', type=int, default=30)
            command_parser.add_argument('--min-f1-gain', type=float, default=.01)
            command_parser.add_argument('--max-extra-tokens', type=float, default=10000)
            command_parser.add_argument('--max-extra-latency', type=float, default=10.0)
            command_parser.add_argument('--max-extra-cost', type=float, default=.01)
            command_parser.add_argument('--no-charts', action='store_true')
    smoke_parser = sub.add_parser('smoke', help='Synthetic offline matrix + checkpoint/report smoke; no API calls')
    smoke_parser.add_argument('--output', type=Path, default=ROOT / 'outputs/prompt_benchmark/smoke')
    args = parser.parse_args()
    if args.command == 'plan':
        result = make_plan(args.cases, args.config, args.output, reference_path=args.reference,
                           definitions_path=args.definitions, verified_path=args.verified,
                           photo_manifest=args.photo_manifest, limit=args.limit, seed=args.seed,
                           batch_size=args.batch_size, blinded=args.blinded, anchor_path=args.anchor)
        result = {'plan_id': result['id'], 'cases': len(result['cases']),
                  'experiments': len(list(experiments(result))), 'output': str(args.output)}
    elif args.command == 'smoke':
        result = smoke(args.output)
    else:
        plan = load_plan(args.output)
        if args.command == 'run':
            result = asyncio.run(run(plan, args.output, workers=args.workers, retries=args.retries,
                model_ids=args.model_ids, variants=args.variants, modalities=args.modalities,
                batches=args.batches, experiment_ids=args.experiment_ids, failed_only=args.failed_only,
                case_ids=args.case_ids, allow_paid=args.allow_paid))
        elif args.command == 'list':
            result = [{'experiment_id': e['id'], 'model': e['model']['id'], 'variant': e['variant'],
                       'modality': e['modality'], 'supports_photos': e['model']['supports_photos']}
                      for e in experiments(plan)]
        else:
            result = evaluate(plan, args.output, bootstrap=args.bootstrap, min_verified=args.min_verified,
                min_f1_gain=args.min_f1_gain, max_extra_tokens=args.max_extra_tokens,
                max_extra_latency=args.max_extra_latency, max_extra_cost=args.max_extra_cost,
                charts=not args.no_charts, evaluation_path=args.evaluation, verified_path=args.verified)
            result = {'report': str((args.output / 'report/index.html').resolve()),
                      'assessments': [{'model': r['model'], 'modality': r['modality'],
                                       'assessment': r['tradeoff_assessment']} for r in result['paired']]}
    import json
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
