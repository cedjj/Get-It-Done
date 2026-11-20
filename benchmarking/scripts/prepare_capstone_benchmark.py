"""Freeze a portable six-model rerun of the exact saved 100-case cohort; no inference."""
import argparse
import csv
import os
from pathlib import Path

from project_paths import ROOT
from prompt_benchmark.core import load_plan, make_plan, read_cases, read_json, sha, write_json

BUNDLE = ROOT / 'outputs/prompt_benchmark/final_evaluation'


def prepare(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; frozen evidence is never overwritten.')
    frozen = load_plan(BUNDLE)
    inputs = BUNDLE / 'data/model_inputs.csv'
    rows = {r['ID']: r for r in read_cases(inputs, blinded=True)}
    assert len(rows) == 100 and set(rows) == {c['id'] for c in frozen['cases']}
    assert all(sha(rows[c['id']]) == c['evaluation_join_hash'] for c in frozen['cases'])
    with (BUNDLE / 'data/scoring_labels.csv').open(encoding='utf-8', newline='') as handle:
        labels = {r['ID']: r for r in csv.DictReader(handle)}
    assert all(labels[c['id']]['input_sha256'] == c['evaluation_join_hash'] for c in frozen['cases'])
    ownership = {p['source']: p for p in read_json(BUNDLE / 'assets/photo_ownership.json')}
    manifest = {}
    for case in frozen['cases']:
        for photo in case['photos']:
            record = ownership[photo['source']]
            assert record['case_id'] == case['id'] and record['case_ownership_verified']
            path = BUNDLE / record['bundle_path']
            assert sha(path.read_bytes()) == photo['sha256'] == record['sha256']
            manifest[photo['source']] = os.path.relpath(path, output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'photos.json', manifest)
    plan = make_plan(inputs, BUNDLE / 'config/models.json', output,
                     reference_path=BUNDLE / 'config/service_definitions.json',
                     photo_manifest=output / 'photos.json', blinded=True,
                     anchor_path=BUNDLE, limit=100, seed=frozen['seed'],
                     batch_size=frozen['batch_size'])
    assert plan['prompts'] == frozen['prompts'] and plan['schema'] == frozen['schema']
    assert plan['config'] == frozen['config']
    assert [c['id'] for c in plan['cases']] == [c['id'] for c in frozen['cases']]
    print(f'Prepared 100 cases, 6 models, 48 configurations: {output}')
    print('No model calls or downloads. Photo paths are local; the new plan has its own identity.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/prompt_benchmark/reproduced')
    prepare(parser.parse_args().output)
