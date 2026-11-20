"""Offline tests: never contact a running model, fetch photos or invoke cloud APIs."""
import asyncio
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import project_paths  # noqa: E402; makes the legacy transport importable
from prompt_benchmark.core import (connect, dump, experiments, load_plan, make_plan,
    read_verified, request_for, schema, sha, validate_prediction, write_json)
from prompt_benchmark.evaluate import evaluate, metrics, paired_ci
from prompt_benchmark.runner import MockAdapter, OllamaAdapter, preflight, run
from prepare_blinded_benchmark import prepare


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'run'
        self.rows = [
            {'ID': 'one', 'DESCRIPTION': 'Blue graffiti painted on wall.', 'ORIGINAL_TYPE': 'Graffiti',
             'CURRENT_TYPE': 'Illegal Dumping', 'label': '1', 'STATUS': 'SECRET CLOSURE',
             'GEOLOCATION__LATITUDE__S': '32.7', 'GEOLOCATION__LONGITUDE__S': '-117.1',
             'PUBLIC_PHOTO_URL_1': 'fixture://one'},
            {'ID': 'two', 'DESCRIPTION': 'Mattress dumped in alley.', 'ORIGINAL_TYPE': 'Traffic Signal',
             'CURRENT_TYPE': 'Illegal Dumping', 'label': '1'},
            {'ID': 'three', 'DESCRIPTION': 'Please help.', 'ORIGINAL_TYPE': 'Curb', 'CURRENT_TYPE': 'Curb', 'label': '0'},
        ]
        self.models = [{'id': 'a', 'adapter': 'mock', 'supports_photos': True,
                        'pricing': {'input_per_million': 2, 'output_per_million': 6}},
                       {'id': 'b', 'adapter': 'mock', 'supports_photos': True}]
        (self.root / 'photo.png').write_bytes(b'fixed image bytes for adapter audit tests')
        write_json(self.root / 'photos.json', {'fixture://one': 'photo.png'})

    def make(self, **kwargs):
        with (self.root / 'cases.csv').open('w', newline='', encoding='utf-8') as handle:
            fields = list(dict.fromkeys(k for r in self.rows for k in r))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        write_json(self.root / 'models.json', {'models': self.models})
        return make_plan(self.root / 'cases.csv', self.root / 'models.json', self.out,
                         photo_manifest=self.root / 'photos.json', **kwargs)

    def query(self, sql):
        conn = connect(self.out)
        try:
            return [dict(row) for row in conn.execute(sql)]
        finally:
            conn.close()

    def execute(self, plan, **kwargs):
        return asyncio.run(run(plan, self.out, retries=0, **kwargs))

    def blinded(self):
        self.make()
        data = self.root / 'blinded_data'
        prepare(self.root / 'cases.csv', data)
        config = self.root / 'blinded_models.json'
        write_json(config, {'models': self.models, 'prompts': {'detailed': {
            'version': 'blinded-test-v1',
            'template': str(ROOT / 'scripts/prompt_benchmark/templates/detailed_blinded.txt')}}})
        output = self.root / 'blinded_run'
        plan = make_plan(data / 'model_inputs.csv', config, output, blinded=True,
                         photo_manifest=self.root / 'photos.json')
        return plan, data, output, config

    def test_blinded_csv_plan_and_requests_exclude_evaluation_fields(self):
        plan, data, output, _ = self.blinded()
        self.assertNotIn('CURRENT_TYPE', (data / 'model_inputs.csv').read_text())
        self.assertNotIn('label', (data / 'model_inputs.csv').read_text())
        self.assertNotIn('"CURRENT_TYPE":', dump(plan))
        for experiment in experiments(plan):
            for case in plan['cases']:
                request = request_for(plan, experiment, case)
                user = json.loads(request['user'])
                self.assertNotIn('CURRENT_TYPE', user)
                self.assertNotIn('label', user)
                self.assertEqual(set(case['inputs']), {'DESCRIPTION', 'ORIGINAL_TYPE'})
                self.assertIsNone(case['weak_misroute'])
                self.assertEqual(case['verified'], {})
        injected = copy.deepcopy(plan['cases'][0])
        injected['inputs']['CURRENT_TYPE'] = 'Secret target'
        with self.assertRaisesRegex(ValueError, 'Unexpected model-input'):
            request_for(plan, next(experiments(plan)), injected)

    def test_blinded_scores_join_original_only_after_inference(self):
        plan, data, output, _ = self.blinded()
        asyncio.run(run(plan, output, modalities=['text', 'text_location'], retries=0))
        no_targets = evaluate(plan, output, bootstrap=0, charts=False)
        self.assertEqual(no_targets['summary'][0]['weak_flag_n'], 0)
        original = evaluate(plan, output, bootstrap=0, charts=False, evaluation_path=self.root / 'cases.csv')
        sidecar = evaluate(plan, output, bootstrap=0, charts=False, evaluation_path=data / 'evaluation_labels.csv')
        self.assertEqual(original['summary'], sidecar['summary'])
        self.assertEqual(original['summary'][0]['weak_flag_n'], 3)
        self.assertEqual(original['summary'][0]['current_agreement_n'], 3)
        self.assertEqual(load_plan(output), plan)
        with patch('requests.post', side_effect=AssertionError('Unexpected inference')):
            self.assertEqual(asyncio.run(run(plan, output, modalities=['text', 'text_location'], retries=0)),
                             {'unchanged': 24})

    def test_blinded_mode_rejects_csv_with_labels_or_current_type(self):
        _, data, output, config = self.blinded()
        with self.assertRaisesRegex(ValueError, 'forbidden fields'):
            make_plan(self.root / 'cases.csv', config, self.root / 'rejected', blinded=True)

    def test_blinded_scoring_rejects_wrong_input_association(self):
        plan, data, output, _ = self.blinded()
        text = (data / 'evaluation_labels.csv').read_text()
        old = plan['cases'][0]['evaluation_join_hash']
        (data / 'evaluation_labels.csv').write_text(text.replace(old, 'bad-hash'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'association hash mismatch'):
            evaluate(plan, output, charts=False, evaluation_path=data / 'evaluation_labels.csv')

    def test_full_matrix_and_resume_no_duplicates(self):
        plan = self.make()
        with patch('requests.post', side_effect=AssertionError('Unexpected network')):
            self.assertEqual(self.execute(plan), {'completed': 48})
            self.assertEqual(self.execute(plan), {'unchanged': 48})
        self.assertEqual(len(self.query('SELECT * FROM predictions')), 48)
        self.assertEqual(len(self.query('SELECT * FROM attempts')), 48)
        for experiment in experiments(plan):
            ids = {r['case_id'] for r in self.query('SELECT * FROM predictions') if r['experiment_id'] == experiment['id']}
            self.assertEqual(ids, {'one', 'two', 'three'})

    def test_prompt_schema_and_allowlist_are_identical(self):
        plan = self.make()
        requests = []
        for experiment in experiments(plan):
            request = request_for(plan, experiment, plan['cases'][0])
            self.assertEqual(request['schema'], schema(plan['categories']))
            user = json.loads(request['user'])
            self.assertEqual(user['ORIGINAL_TYPE'], plan['cases'][0]['inputs']['ORIGINAL_TYPE'])
            self.assertEqual(user['CURRENT_TYPE'], plan['cases'][0]['inputs']['CURRENT_TYPE'])
            self.assertNotIn('ID', user)
            self.assertNotIn('label', user)
            self.assertNotIn('SECRET CLOSURE', dump(request))
            self.assertEqual('LOCATION' in user, 'location' in experiment['modality'])
            self.assertEqual('PHOTOS' in user, 'photos' in experiment['modality'])
            requests.append(request)
        self.assertEqual(requests[0]['user'], requests[1]['user'])
        self.assertNotIn('MARKDOWN ROUTING DEFINITIONS', plan['prompts']['simple']['system'])
        self.assertIn('Independently assess ORIGINAL_TYPE and CURRENT_TYPE', plan['prompts']['detailed']['system'])
        self.assertNotEqual(plan['prompts']['simple']['hash'], plan['prompts']['detailed']['hash'])

    def test_schema_validation(self):
        valid = {'misroute': None, 'recommended_category': 'Graffiti', 'reason': 'Visible tags.'}
        self.assertEqual(validate_prediction(dump(valid), ['Graffiti']), valid)
        for invalid in [dict(valid, misroute=1), dict(valid, recommended_category='Nope'),
                        dict(valid, extra='field'), dict(valid, reason=''), ['wrong type']]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_prediction(dump(invalid), ['Graffiti'])

    def test_plan_tampering_and_existing_plan_rejected(self):
        plan = self.make()
        self.assertEqual(load_plan(self.out), plan)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.make()
        plan['cases'][0]['inputs']['CURRENT_TYPE'] = 'Changed'
        write_json(self.out / 'plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_plan(self.out)

    def test_failed_cell_can_rerun_without_duplicating_completed(self):
        plan = self.make()
        class Broken(MockAdapter):
            async def generate(self, model, request):
                if model['id'] == 'a' and 'Mattress' in request['user']:
                    return {'raw': 'RAW INVALID JSON', 'content': '{', 'input_tokens': 7, 'output_tokens': 3}
                return await super().generate(model, request)
        counts = self.execute(plan, adapters={'mock': Broken()})
        self.assertEqual(counts, {'completed': 40, 'failed': 8})
        failed = self.query("SELECT * FROM predictions WHERE status='failed'")
        target = failed[0]['experiment_id']
        self.assertEqual(self.execute(plan, failed_only=True, experiment_ids=[target]), {'unchanged': 2, 'completed': 1})
        self.assertEqual(len(self.query('SELECT * FROM predictions')), 48)
        self.assertEqual(len(self.query('SELECT * FROM attempts')), 49)
        self.assertEqual(len(self.query("SELECT * FROM attempts WHERE raw_response='RAW INVALID JSON'")), 8)
        attempt = self.query("SELECT * FROM attempts WHERE raw_response='RAW INVALID JSON'")[0]
        self.assertAlmostEqual(attempt['cost_usd'], 32 / 1_000_000)
        self.assertEqual(len(self.query("SELECT * FROM predictions WHERE status='failed'")), 7)

    def test_retry_usage_includes_bad_response(self):
        plan = self.make()
        class Once(MockAdapter):
            def __init__(self):
                self.calls = 0
            async def generate(self, model, request):
                self.calls += 1
                if self.calls == 1:
                    return {'raw': 'invalid', 'content': '{}', 'input_tokens': 9, 'output_tokens': 2}
                return await super().generate(model, request)
        asyncio.run(run(plan, self.out, retries=1, workers=1, model_ids=['a'],
                        variants=['simple'], modalities=['text'], adapters={'mock': Once()}))
        self.assertEqual(len(self.query('SELECT * FROM predictions')), 3)
        attempts = self.query('SELECT * FROM attempts')
        self.assertEqual(len(attempts), 4)
        report = evaluate(plan, self.out, bootstrap=0, charts=False)
        row = next(r for r in report['summary'] if r['model'] == 'a' and r['variant'] == 'simple' and r['modality'] == 'text')
        self.assertEqual(row['all_attempts_input_tokens'], sum(a['input_tokens'] for a in attempts))
        self.assertAlmostEqual(row['mean_input_tokens'] * 3, sum(a['input_tokens'] for a in attempts))

    def test_missing_cache_does_not_silently_drop_images(self):
        (self.root / 'photos.json').write_text('{}', encoding='utf-8')
        plan = self.make()
        result = self.execute(plan)
        self.assertEqual(result, {'completed': 40, 'failed': 8})
        self.assertTrue(all('not cached' in r['error'] for r in self.query("SELECT * FROM predictions WHERE status='failed'")))

    def test_changed_image_hash_fails_photo_cells(self):
        plan = self.make()
        (self.root / 'photo.png').write_bytes(b'changed')
        self.assertEqual(self.execute(plan), {'completed': 40, 'failed': 8})

    def test_unsupported_photos_are_visible_skips(self):
        self.models[0]['supports_photos'] = False
        plan = self.make()
        self.assertEqual(self.execute(plan), {'completed': 36, 'skipped': 12})
        self.assertEqual(len(self.query('SELECT * FROM attempts')), 36)
        report = evaluate(plan, self.out, bootstrap=0, charts=False)
        row = next(r for r in report['summary'] if r['model'] == 'a' and r['modality'] == 'text_photos')
        self.assertEqual(row['skipped'], 3)
        self.assertIsNone(row['weak_f1'])

    def test_weak_labels_never_become_verified_accuracy(self):
        plan = self.make()
        self.execute(plan)
        report = evaluate(plan, self.out, bootstrap=20, charts=False)
        for row in report['summary']:
            self.assertEqual(row['weak_flag_n'], 3)
            self.assertEqual(row['verified_flag_n'], 0)
            self.assertEqual(row['verified_category_n'], 0)
            self.assertIsNone(row['verified_f1'])
            self.assertIsNone(row['verified_category_accuracy'])
        self.assertIsNone(next(r for r in report['summary'] if r['model'] == 'b')['mean_cost_usd'])

    def test_verified_provenance_required_and_not_in_requests(self):
        path = self.root / 'verified.csv'
        path.write_text('ID,independently_verified,verification_provenance,verified_misroute,verified_category\n'
                        'one,false,Human,0,Graffiti\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'independently_verified'):
            read_verified(path, ['Graffiti'])
        path.write_text(path.read_text().replace('false,Human', 'true,BLINDED REVIEW'), encoding='utf-8')
        plan = self.make(verified_path=path)
        self.execute(plan)
        report = evaluate(plan, self.out, bootstrap=20, charts=False)
        self.assertEqual(report['summary'][0]['verified_category_accuracy'], 1)
        self.assertEqual(report['summary'][0]['verified_category_n'], 1)
        self.assertEqual(report['summary'][0]['weak_fn'], 1)
        for attempt in self.query('SELECT * FROM attempts'):
            self.assertNotIn('BLINDED REVIEW', attempt['request_json'])

    def test_paired_comparison_uses_only_shared_successes(self):
        plan = self.make()
        class FailDetailed(MockAdapter):
            async def generate(self, model, request):
                if 'MARKDOWN ROUTING DEFINITIONS' in request['system'] and 'Mattress' in request['user']:
                    raise RuntimeError('fixture failure')
                return await super().generate(model, request)
        self.execute(plan, adapters={'mock': FailDetailed()})
        report = evaluate(plan, self.out, bootstrap=0, charts=False)
        for pair in report['paired']:
            self.assertEqual(pair['paired_n'], 2)
            self.assertEqual(pair['simple_only_n'], 1)
            self.assertEqual(pair['simple_completed'], 2)
            self.assertEqual(pair['detailed_completed'], 2)

    def test_metric_denominators_and_bootstrap(self):
        def record(prediction, flag, category):
            return {'prediction': prediction,
                    'case': {'weak_misroute': flag, 'verified': {'misroute': flag, 'category': category},
                             'inputs': {'ORIGINAL_TYPE': 'Graffiti', 'CURRENT_TYPE': 'Graffiti'},
                             'location': None, 'photos': []},
                    'usage': dict(input_tokens=10, output_tokens=2, cost_usd=0, latency_seconds=.1)}
        simple = [record({'misroute': True, 'recommended_category': 'Graffiti'}, True, 'Graffiti'),
                  record({'misroute': True, 'recommended_category': None}, False, 'Graffiti'),
                  record({'misroute': None, 'recommended_category': None}, True, 'Graffiti')]
        detailed = copy.deepcopy(simple)
        detailed[1]['prediction'] = {'misroute': False, 'recommended_category': 'Graffiti'}
        detailed[2]['prediction'] = {'misroute': True, 'recommended_category': 'Graffiti'}
        sm, dm = metrics(simple, {}), metrics(detailed, {})
        self.assertEqual(sm['verified_f1'], .5)
        self.assertEqual(sm['verified_precision'], .5)
        self.assertEqual(sm['verified_recall'], .5)
        self.assertEqual(sm['verified_category_accuracy'], 1 / 3)
        self.assertEqual(sm['verified_false_positive_rate'], 1)
        self.assertEqual(dm['verified_f1'], 1)
        low, high = paired_ci(simple, detailed, {}, 'verified_category_accuracy', 100, 42)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)

    def test_sampling_and_batch_membership_frozen(self):
        self.rows = [dict(self.rows[2], ID=str(i)) for i in range(205)]
        plan = self.make(limit=201)
        self.assertEqual(len(plan['cases']), 201)
        self.assertEqual([sum(c['batch'] == i for c in plan['cases']) for i in range(3)], [100, 100, 1])
        self.assertEqual(self.execute(plan, batches=[2], model_ids=['a']), {'completed': 8})
        self.assertEqual({r['case_id'] for r in self.query('SELECT * FROM predictions')}, {plan['cases'][-1]['id']})

    def test_pilot_case_filter_expands_without_repeating_predictions(self):
        plan = self.make()
        self.assertEqual(self.execute(plan, case_ids=['one'], modalities=['text', 'text_location']), {'completed': 8})
        self.assertEqual({r['case_id'] for r in self.query('SELECT * FROM predictions')}, {'one'})
        self.assertEqual(self.execute(plan, modalities=['text', 'text_location']), {'unchanged': 8, 'completed': 16})
        self.assertEqual(len(self.query('SELECT * FROM attempts')), 24)

    def test_markdown_and_custom_prompt_version_are_pinned(self):
        definitions = self.root / 'definitions.md'
        definitions.write_text('# Exact existing definitions\nKeep verbatim.', encoding='utf-8')
        plan = self.make(definitions_path=definitions)
        self.assertIn(definitions.read_text(), plan['prompts']['detailed']['system'])
        definitions.write_text('Changed later', encoding='utf-8')
        self.assertEqual(load_plan(self.out), plan)

    def test_custom_template_version_is_recorded_on_each_prediction(self):
        self.make()
        custom_out = self.root / 'custom'
        (self.root / 'simple.txt').write_text('Custom concise prompt. Valid categories: {categories}', encoding='utf-8')
        write_json(self.root / 'custom.json', {'models': self.models, 'prompts': {
            'simple': {'template': 'simple.txt', 'version': 'custom-v9'}}})
        plan = make_plan(self.root / 'cases.csv', self.root / 'custom.json', custom_out,
                         photo_manifest=self.root / 'photos.json')
        asyncio.run(run(plan, custom_out, variants=['simple'], modalities=['text'], retries=0))
        conn = connect(custom_out)
        try:
            rows = list(conn.execute('SELECT prompt_variant,prompt_version,prompt_hash FROM predictions'))
        finally:
            conn.close()
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(tuple(row) == ('simple', 'custom-v9', plan['prompts']['simple']['hash']) for row in rows))

    def test_photo_bytes_and_location_are_sent_only_when_enabled(self):
        plan = self.make()
        case = next(c for c in plan['cases'] if c['id'] == 'one')
        for experiment in experiments(plan):
            request = request_for(plan, experiment, case)
            self.assertEqual(len(request['images']), int('photos' in experiment['modality']))
            if request['images']:
                import base64
                self.assertEqual(base64.b64decode(request['images'][0]), (self.root / 'photo.png').read_bytes())
            self.assertNotIn('fixture://one', request['user'])
            if 'location' in experiment['modality']:
                self.assertEqual(json.loads(request['user'])['LOCATION'], {'latitude': 32.7, 'longitude': -117.1})

    def test_runtime_fingerprint_change_requires_new_plan(self):
        self.make()
        conn = connect(self.out)
        def fake_api(path, body=None):
            return {'/api/tags': {'models': [{'name': 'installed', 'digest': 'v1'}]},
                    '/api/show': {'capabilities': ['completion', 'vision']},
                    '/api/version': {'version': 'fixed'}}[path]
        model = {'id': 'live', 'adapter': 'ollama', 'model': 'installed', 'supports_photos': True}
        try:
            with patch('prompt_benchmark.local_transport.api', side_effect=fake_api):
                asyncio.run(preflight(conn, model))
            with conn:
                conn.execute("UPDATE meta SET value='different' WHERE key='model_fingerprint:live'")
            with patch('prompt_benchmark.local_transport.api', side_effect=fake_api), self.assertRaisesRegex(ValueError, 'changed'):
                asyncio.run(preflight(conn, model))
        finally:
            conn.close()

    def test_duplicate_case_ids_rejected(self):
        self.rows[1]['ID'] = self.rows[0]['ID']
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.make()

    def test_checkpoint_cannot_mix_plans(self):
        plan = self.make()
        self.execute(plan)
        changed = copy.deepcopy(plan)
        changed['id'] = sha('another plan')
        with self.assertRaisesRegex(ValueError, 'different plan'):
            self.execute(changed)

    def test_ollama_adapter_retains_truncation_and_usage(self):
        plan, _, _, _ = self.blinded()
        request = request_for(plan, next(experiments(plan)), plan['cases'][0])
        response = {'done': True, 'done_reason': 'length', 'message': {'content': '{'},
                    'prompt_eval_count': 12, 'eval_count': 5}
        with patch('prompt_benchmark.local_transport.api', return_value=response) as api:
            result = asyncio.run(OllamaAdapter().generate({'model': 'installed'}, request))
        self.assertIn('Truncated', result['transport_error'])
        self.assertEqual(json.loads(result['raw']), response)
        self.assertEqual(result['input_tokens'], 12)
        self.assertEqual(api.call_args.args[0], '/api/chat')
        self.assertEqual(api.call_args.args[1]['format'], plan['schema'])

    def test_preflight_never_pulls_missing_models(self):
        self.make()
        conn = connect(self.out)
        try:
            with patch('prompt_benchmark.local_transport.api', return_value={'models': []}) as api:
                with self.assertRaisesRegex(ValueError, 'never downloads'):
                    asyncio.run(preflight(conn, {'id': 'missing', 'adapter': 'ollama', 'model': 'missing'}))
            self.assertEqual(api.call_count, 1)
            self.assertEqual(api.call_args.args, ('/api/tags',))
        finally:
            conn.close()


if __name__ == '__main__':
    unittest.main()
