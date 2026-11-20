"""Separate weak agreement from verification; compare only identical completed cases."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

from .core import bind_plan, connect, dump, experiments, parse_bool, read_verified, sha, write_json


def ratio(a, b):
    return a / b if b else None


def mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def metrics(records, aliases):
    out = {'completed': len(records)}
    for source in ('weak', 'verified'):
        labeled = [(r['prediction']['misroute'], r['case']['weak_misroute'] if source == 'weak'
                    else r['case']['verified'].get('misroute')) for r in records]
        labeled = [(p, y) for p, y in labeled if y is not None]
        tp = sum(p is True and y is True for p, y in labeled)
        fp = sum(p is True and y is False for p, y in labeled)
        # An abstention on a positive is an unflagged positive in recall/F1.
        fn = sum(p is not True and y is True for p, y in labeled)
        tn = sum(p is False and y is False for p, y in labeled)
        out.update({f'{source}_flag_n': len(labeled), f'{source}_tp': tp, f'{source}_fp': fp,
                    f'{source}_fn': fn, f'{source}_tn': tn,
                    f'{source}_precision': ratio(tp, tp + fp),
                    f'{source}_recall': ratio(tp, tp + fn),
                    f'{source}_f1': ratio(2 * tp, 2 * tp + fp + fn),
                    f'{source}_false_positive_rate': ratio(fp, sum(y is False for _, y in labeled)),
                    f'{source}_flag_accuracy': ratio(tp + tn, len(labeled)),
                    f'{source}_negative_abstention_n': sum(p is None and y is False for p, y in labeled),
                    f'{source}_flag_coverage': ratio(sum(p is not None for p, _ in labeled), len(labeled))})
    verified_categories = [(r['prediction']['recommended_category'], r['case']['verified'].get('category'))
                           for r in records if r['case']['verified'].get('category') is not None]
    out['verified_category_n'] = len(verified_categories)
    out['verified_category_accuracy'] = ratio(sum(p == y for p, y in verified_categories), len(verified_categories))
    out['category_abstention_rate'] = ratio(sum(r['prediction']['recommended_category'] is None for r in records), len(records))
    out['flag_abstention_rate'] = ratio(sum(r['prediction']['misroute'] is None for r in records), len(records))
    out['any_abstention_rate'] = ratio(sum(r['prediction']['misroute'] is None or
                                         r['prediction']['recommended_category'] is None for r in records), len(records))
    for field, name in [('ORIGINAL_TYPE', 'original'), ('CURRENT_TYPE', 'current')]:
        def target(record):
            case = record['case']
            return case.get('evaluation_current_type', case['inputs'].get(field, '')) if field == 'CURRENT_TYPE' else case['inputs'][field]
        usable = [r for r in records if target(r).strip()]
        out[f'{name}_agreement_n'] = len(usable)
        out[f'{name}_agreement'] = ratio(sum(r['prediction']['recommended_category'] is not None and
            r['prediction']['recommended_category'] == aliases.get(target(r), target(r))
            for r in usable), len(usable))
        out[f'{name}_unmapped_n'] = sum(target(r) not in aliases for r in usable)
    out['with_location_n'] = sum(r['case']['location'] is not None for r in records)
    out['with_photos_n'] = sum(bool(r['case']['photos']) for r in records)
    out['image_cases_sent_n'] = sum('photos' in r.get('modality', '') and bool(r['case']['photos']) for r in records)
    out['location_cases_sent_n'] = sum('location' in r.get('modality', '') and r['case']['location'] is not None for r in records)
    for field in ('input_tokens', 'output_tokens', 'cost_usd', 'latency_seconds'):
        out[f'mean_{field}'] = mean([r['usage'][field] for r in records])
    return out


def difference(a, b):
    return b - a if a is not None and b is not None else None


def paired_ci(simple, detailed, aliases, field, iterations, seed):
    # Resample paired cases, and only labeled cases for the selected metric.
    label = 'category' if field == 'verified_category_accuracy' else 'misroute'
    def target(record):
        return record['case']['weak_misroute'] if field.startswith('weak_') else record['case']['verified'].get(label)
    paired = [(a, b) for a, b in zip(simple, detailed) if target(a) is not None]
    if not paired or iterations < 1:
        return None, None
    # Precompute outcomes to avoid repeatedly evaluating every other metric during
    # bootstrap (important for full 100-case batches across many models).
    if label == 'category':
        outcomes = [(a['prediction']['recommended_category'] == a['case']['verified']['category'],
                     b['prediction']['recommended_category'] == b['case']['verified']['category']) for a, b in paired]
    else:
        def counts(record):
            prediction, truth = record['prediction']['misroute'], target(record)
            return (int(prediction is True and truth is True), int(prediction is True and truth is False),
                    int(prediction is not True and truth is True))
        outcomes = [(counts(a), counts(b)) for a, b in paired]
    rng, values = random.Random(seed), []
    for _ in range(iterations):
        sample = rng.choices(outcomes, k=len(outcomes))
        if label == 'category':
            delta = sum(y - x for x, y in sample) / len(sample)
        else:
            first = [sum(x[0][i] for x in sample) for i in range(3)]
            second = [sum(x[1][i] for x in sample) for i in range(3)]
            delta = difference(ratio(2 * first[0], 2 * first[0] + first[1] + first[2]),
                               ratio(2 * second[0], 2 * second[0] + second[1] + second[2]))
        if delta is not None:
            values.append(delta)
    if len(values) < iterations * 0.9:
        return None, None
    values.sort()
    return values[int((len(values) - 1) * .025)], values[int((len(values) - 1) * .975)]


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def usage_for(attempts):
    out = {'attempts': len(attempts)}
    for field in ('input_tokens', 'output_tokens', 'cost_usd', 'latency_seconds'):
        values = [a[field] for a in attempts]
        out[field] = sum(values) if values and all(v is not None for v in values) else None
    return out


def evaluate(plan, output, *, bootstrap=1000, min_verified=30, min_f1_gain=.01,
             max_extra_tokens=10000, max_extra_latency=10.0, max_extra_cost=.01, charts=True,
             evaluation_path=None, case_ids=None, report_name='report', verified_path=None, raw_export=True):
    if bootstrap < 0 or min_verified < 1:
        raise ValueError('bootstrap must be nonnegative; min_verified must be positive')
    output = Path(output)
    if Path(report_name).name != report_name:
        raise ValueError('Report name must be a single directory name')
    report_dir = output / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    conn = connect(output)
    try:
        bind_plan(conn, plan)
        rows = [dict(r) for r in conn.execute('SELECT * FROM predictions')]
        attempts = [dict(r) for r in conn.execute('SELECT * FROM attempts ORDER BY id')]
    finally:
        conn.close()
    by_attempt = defaultdict(list)
    if case_ids is not None:
        if set(case_ids) - {c['id'] for c in plan['cases']}:
            raise ValueError('Unknown report case IDs')
        rows = [r for r in rows if r['case_id'] in case_ids]
        attempts = [r for r in attempts if r['case_id'] in case_ids]
    for attempt in attempts:
        by_attempt[attempt['experiment_id'], attempt['case_id']].append(attempt)
    # Evaluation-only joins never modify the frozen plan or inference requests.
    cases = {c['id']: dict(c) for c in plan['cases'] if case_ids is None or c['id'] in case_ids}
    if evaluation_path:
        if plan.get('input_policy') != 'blinded':
            raise ValueError('External scoring labels require a blinded plan')
        with Path(evaluation_path).open(encoding='utf-8-sig', newline='') as handle:
            evaluation_rows = list(csv.DictReader(handle))
        lookup = {}
        for row in evaluation_rows:
            case_id = row['ID'].strip()
            if not case_id or case_id in lookup:
                raise ValueError('Evaluation IDs must be nonempty and unique')
            lookup[case_id] = row
        if set(cases) - set(lookup):
            raise ValueError('Evaluation source is missing selected case IDs')
        for case_id, case in cases.items():
            row = lookup[case_id]
            association = (sha({k: row[k] for k in plan['input_columns']})
                           if all(k in row for k in plan['input_columns']) else row.get('input_sha256'))
            if association != case['evaluation_join_hash']:
                raise ValueError('Evaluation/input association hash mismatch')
            case['weak_misroute'] = parse_bool(row.get('label'), 'label')
            case['weak_provenance'] = row.get('LABEL_PROVENANCE', '')
            case['evaluation_current_type'] = row.get('CURRENT_TYPE', '')
    if verified_path:
        adjudications=read_verified(verified_path,plan['categories'])
        if not set(adjudications)&set(cases):
            raise ValueError('No verified case IDs match the selected benchmark')
        for case_id,case in cases.items():
            case['verified']=adjudications.get(case_id,{})
    completed = defaultdict(dict)
    by_experiment = defaultdict(list)
    false_positives = []
    for row in rows:
        by_experiment[row['experiment_id']].append(row)
        if row['status'] != 'completed':
            continue
        record = {'case': cases[row['case_id']], 'prediction': json.loads(row['prediction_json']), 'modality': row['modality'],
                  'usage': usage_for(by_attempt[row['experiment_id'], row['case_id']])}
        completed[row['experiment_id']][row['case_id']] = record
        for source, target in [('weak', record['case']['weak_misroute']),
                               ('verified', record['case']['verified'].get('misroute'))]:
            if target is False and record['prediction']['misroute'] is True:
                false_positives.append({'experiment_id': row['experiment_id'], 'case_id': row['case_id'],
                    'model': row['model_id'], 'modality': row['modality'], 'variant': row['prompt_variant'],
                    'label_source': source, 'reason': record['prediction']['reason']})
    summaries, batch_rows, lookup = [], [], {}
    for experiment in experiments(plan):
        records = list(completed[experiment['id']].values())
        statuses = by_experiment[experiment['id']]
        all_usage = usage_for([a for a in attempts if a['experiment_id'] == experiment['id']])
        summary = {'experiment_id': experiment['id'], 'model': experiment['model']['id'],
                   'adapter': experiment['model']['adapter'], 'modality': experiment['modality'],
                   'variant': experiment['variant'], 'planned': len(cases),
                   'failed': sum(r['status'] == 'failed' for r in statuses),
                   'skipped': sum(r['status'] == 'skipped' for r in statuses),
                   'pending': len(cases) - len(statuses),
                   **metrics(records, plan['aliases']),
                   **{f'all_attempts_{k}': v for k, v in all_usage.items()}}
        summaries.append(summary)
        lookup[experiment['model']['id'], experiment['modality'], experiment['variant']] = experiment
        for batch in sorted({c['batch'] for c in cases.values()}):
            batch_rows.append({'model': summary['model'], 'modality': summary['modality'],
                               'variant': summary['variant'], 'batch': batch,
                               'planned': sum(c['batch'] == batch for c in cases.values()),
                               **metrics([r for r in records if r['case']['batch'] == batch], plan['aliases'])})
    paired, per_case = [], []
    for model in plan['config']['models']:
        for modality in plan['modalities']:
            a = lookup[model['id'], modality, 'simple']
            b = lookup[model['id'], modality, 'detailed']
            simple, detailed = completed[a['id']], completed[b['id']]
            ids = sorted(set(simple) & set(detailed))
            s, d = [simple[i] for i in ids], [detailed[i] for i in ids]
            sm, dm = metrics(s, plan['aliases']), metrics(d, plan['aliases'])
            row = {'model': model['id'], 'modality': modality, 'paired_n': len(ids),
                   'planned': len(cases), 'simple_only_n': len(set(simple) - set(detailed)),
                   'detailed_only_n': len(set(detailed) - set(simple)),
                   'verified_flag_n': sm['verified_flag_n'], 'verified_category_n': sm['verified_category_n'],
                   'weak_flag_n': sm['weak_flag_n']}
            for field in sm:
                row['simple_' + field], row['detailed_' + field] = sm[field], dm[field]
                row['delta_' + field] = difference(sm[field], dm[field])
            for field in ('weak_f1', 'verified_f1', 'verified_category_accuracy'):
                low, high = paired_ci(s, d, plan['aliases'], field, bootstrap,
                                      int(sha([plan['seed'], model['id'], modality, field])[:8], 16))
                row[f'delta_{field}_ci_low'], row[f'delta_{field}_ci_high'] = low, high
            row['category_simple_correct_detailed_wrong'] = 0
            row['category_detailed_correct_simple_wrong'] = 0
            for case_id, first, second in zip(ids, s, d):
                target = first['case']['verified'].get('category')
                if target is not None:
                    x = first['prediction']['recommended_category'] == target
                    y = second['prediction']['recommended_category'] == target
                    row['category_simple_correct_detailed_wrong'] += int(x and not y)
                    row['category_detailed_correct_simple_wrong'] += int(y and not x)
                per_case.append({'model': model['id'], 'modality': modality, 'case_id': case_id,
                    'batch': first['case']['batch'], 'simple': dump(first['prediction']),
                    'detailed': dump(second['prediction']),
                    **{f'delta_{key}': difference(first['usage'][key], second['usage'][key])
                       for key in ('input_tokens', 'output_tokens', 'latency_seconds', 'cost_usd')}})
            extra_tokens = (row['delta_mean_input_tokens'] + row['delta_mean_output_tokens']
                            if row['delta_mean_input_tokens'] is not None and
                            row['delta_mean_output_tokens'] is not None else None)
            row['delta_mean_total_tokens'] = extra_tokens
            if model['adapter'] == 'mock':
                decision = 'synthetic_fixture_only'
            elif len(ids) != len(cases):
                decision = 'incomplete_paired_experiment'
            elif sm['verified_flag_n'] < min_verified:
                decision = 'insufficient_independent_verification'
            elif row['delta_verified_f1_ci_low'] is None:
                decision = 'insufficient_information_for_paired_interval'
            elif row['delta_verified_f1'] < min_f1_gain or row['delta_verified_f1_ci_low'] <= 0:
                decision = 'detailed_gain_not_established'
            elif any(v is None for v in (extra_tokens, row['delta_mean_latency_seconds'], row['delta_mean_cost_usd'])):
                decision = 'overhead_unknown'
            elif (extra_tokens > max_extra_tokens or row['delta_mean_latency_seconds'] > max_extra_latency
                  or row['delta_mean_cost_usd'] > max_extra_cost):
                decision = 'detailed_exceeds_overhead_budget'
            else:
                decision = 'detailed_meets_configured_tradeoff'
            row['tradeoff_assessment'] = decision
            paired.append(row)
    availability = []
    for summary in summaries:
        records = list(completed[summary['experiment_id']].values())
        for present in (True, False):
            availability.append({'model': summary['model'], 'modality': summary['modality'],
                'variant': summary['variant'], 'has_linked_photos': present,
                **metrics([r for r in records if bool(r['case']['photos']) == present], plan['aliases'])})
    comparison_fields = ['weak_precision','weak_recall','weak_f1','weak_flag_accuracy',
        'flag_abstention_rate','category_abstention_rate','original_agreement','current_agreement',
        'mean_input_tokens','mean_output_tokens','mean_latency_seconds','mean_cost_usd']
    def compare(first, second, labels):
        left, right = completed[first['id']], completed[second['id']]
        ids = sorted(set(left) & set(right))
        a, b = metrics([left[i] for i in ids],plan['aliases']), metrics([right[i] for i in ids],plan['aliases'])
        return {**labels,'paired_n':len(ids),**{'delta_'+k:difference(a[k],b[k]) for k in comparison_fields}}
    modality_rows, model_rows = [], []
    for model in plan['config']['models']:
        for variant in ('simple','detailed'):
            for modality in plan['modalities']:
                if modality != 'text':
                    modality_rows.append(compare(lookup[model['id'],'text',variant],lookup[model['id'],modality,variant],
                        {'model':model['id'],'variant':variant,'baseline':'text','comparison':modality}))
    models = plan['config']['models']
    for i, first in enumerate(models):
        for second in models[i+1:]:
            for modality in plan['modalities']:
                for variant in ('simple','detailed'):
                    model_rows.append(compare(lookup[first['id'],modality,variant],lookup[second['id'],modality,variant],
                        {'baseline_model':first['id'],'comparison_model':second['id'],'variant':variant,'modality':modality}))
    write_csv(report_dir / 'photo_availability.csv',availability)
    write_csv(report_dir / 'modality_differences.csv',modality_rows)
    write_csv(report_dir / 'model_comparisons.csv',model_rows)
    write_csv(report_dir / 'performance.csv', summaries)
    write_csv(report_dir / 'by_batch.csv', batch_rows)
    write_csv(report_dir / 'paired_differences.csv', paired)
    write_csv(report_dir / 'paired_cases.csv', per_case)
    write_csv(report_dir / 'false_positives.csv', false_positives)
    write_csv(report_dir / 'predictions.csv', rows)
    # Preserve exact raw strings and request content in portable JSONL as well as SQLite.
    if raw_export:
        with (report_dir / 'raw_attempts.jsonl').open('w', encoding='utf-8') as handle:
            for attempt in attempts:
                handle.write(dump(attempt) + '\n')
    artifact = {'plan_id': plan['id'], 'summary': summaries, 'paired': paired,
                'evaluation_source_sha256': sha(Path(evaluation_path).read_bytes()) if evaluation_path else None,
                'verified_source_sha256': sha(Path(verified_path).read_bytes()) if verified_path else None,
                'modality_comparisons': modality_rows, 'model_comparisons': model_rows,
                'policy': {'primary_metric': 'verified_f1', 'min_verified': min_verified,
                           'min_f1_gain': min_f1_gain, 'max_extra_tokens': max_extra_tokens,
                           'max_extra_latency_seconds': max_extra_latency,
                           'max_extra_cost_usd': max_extra_cost, 'bootstrap_iterations': bootstrap},
                'notes': [plan['evaluation_note'],
                    'Differences are detailed minus simple on identical completed cases only.',
                    'Null denotes unavailable/undefined; unknown cost is never reported as zero.',
                    'Accuracy includes category abstentions as incorrect. Positive flag abstentions count as false negatives.',
                    'Original/current agreement uses explicit reference aliases; unmapped workflow names stay unmapped.',
                    'Raw outputs and resource totals include failed/retried attempts. Latency is per-request wall time including retries, excluding queue wait.',
                    'Cost is API token cost only, excluding hardware/energy; cloud token-rate estimates are not invoices. Mock tokens/timings are synthetic plumbing evidence.',
                    'Flag accuracy counts all abstentions as incorrect; F1 alone does not penalize negative-case abstentions. Inspect coverage and abstentions together.',
                    'Image cases sent counts actual nonempty image inputs. Linked-photo counts describe dataset availability, including text-only configurations.',
                    'Intervals are paired case bootstrap percentiles, descriptive and unadjusted for multiple comparisons.',
                    'Tradeoff decisions use configured thresholds on paired verified F1; inspect category accuracy, completeness and all modality/model rows before choosing a winner.']}
    write_json(report_dir / 'report.json', artifact)
    chart_names = make_charts(summaries, paired, report_dir) if charts else []
    write_html(artifact, report_dir, chart_names)
    return artifact


def make_charts(summaries, paired, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    groups = list(dict.fromkeys((r['model'], r['modality']) for r in summaries))
    x = np.arange(len(groups))
    lookup = {(r['model'], r['modality'], r['variant']): r for r in summaries}
    names = []
    panels = [
        ('performance.png', [('verified_f1', 'Verified misroute F1'), ('weak_f1', 'Weak-label flag F1 agreement'),
                             ('verified_category_accuracy', 'Verified category accuracy')]),
        ('overhead.png', [('mean_input_tokens', 'Input tokens / completed case (with retries)'),
                          ('mean_latency_seconds', 'Latency seconds / completed case (with retries)'),
                          ('mean_cost_usd', 'API USD / completed case (with retries)')]),
    ]
    for filename, fields in panels:
        fig, axes = plt.subplots(len(fields), 1, figsize=(max(10, len(groups) * .7), 10), layout='constrained')
        for ax, (field, title) in zip(axes, fields):
            values_present = False
            for offset, variant in [(-.18, 'simple'), (.18, 'detailed')]:
                values = [lookup[model, modality, variant][field] for model, modality in groups]
                values_present |= any(v is not None for v in values)
                ax.bar(x + offset, [v if v is not None else float('nan') for v in values], .36, label=variant)
            if not values_present:
                ax.text(.5, .5, 'No measured values', transform=ax.transAxes, ha='center')
            ax.set_title(title)
            ax.set_xticks(x, [f'{m}\n{modes}' for m, modes in groups], rotation=35, ha='right', fontsize=8)
            ax.legend()
            ax.grid(axis='y', alpha=.2)
        context = 'synthetic fixtures' if all(r['adapter']=='mock' for r in summaries) else 'real inference; blank bars mean unavailable'
        fig.suptitle('Paired prompt benchmark — '+context,fontsize=11)
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
        names.append(filename)
    fig, ax = plt.subplots(figsize=(max(10, len(groups) * .7), 5), layout='constrained')
    for i, row in enumerate(paired):
        point, low, high = (row[k] for k in ('delta_verified_f1', 'delta_verified_f1_ci_low', 'delta_verified_f1_ci_high'))
        if point is not None:
            ax.plot(i, point, 'o', color='tab:blue')
            if low is not None:
                ax.vlines(i, low, high, color='tab:blue')
    ax.axhline(0, color='gray', linewidth=1)
    ax.set_xticks(x, [f'{m}\n{modes}' for m, modes in groups], rotation=35, ha='right', fontsize=8)
    ax.set_title('Detailed − simple verified F1; paired bootstrap 95% intervals')
    fig.savefig(output / 'paired_f1.png', dpi=140)
    plt.close(fig)
    names.append('paired_f1.png')
    fig, ax = plt.subplots(figsize=(max(10,len(groups)*.7),5),layout='constrained')
    for i,row in enumerate(paired):
        if row['delta_weak_f1'] is not None:
            ax.plot(i,row['delta_weak_f1'],'o',color='tab:blue')
            if row['delta_weak_f1_ci_low'] is not None:
                ax.vlines(i,row['delta_weak_f1_ci_low'],row['delta_weak_f1_ci_high'],color='tab:blue')
    ax.axhline(0,color='gray',linewidth=1)
    ax.set_xticks(x,[f'{m}\n{modes}' for m,modes in groups],rotation=35,ha='right',fontsize=8)
    ax.set_title('Detailed − simple weak-label F1 agreement (not verified accuracy); paired 95% intervals')
    fig.savefig(output/'paired_weak_f1.png',dpi=140)
    plt.close(fig)
    return names+['paired_weak_f1.png']


def write_html(report, output, charts):
    from html import escape
    def fmt(value):
        return '—' if value is None else f'{value:.4g}' if isinstance(value, float) else escape(str(value))
    def table(rows, fields):
        return '<table><thead><tr>' + ''.join(f'<th>{escape(k)}</th>' for k in fields) + '</tr></thead><tbody>' + ''.join(
            '<tr>' + ''.join(f'<td>{fmt(r[k])}</td>' for k in fields) + '</tr>' for r in rows) + '</tbody></table>'
    body = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Prompt benchmark</title>'
    body += '<style>body{font:15px system-ui;margin:2rem;color:#152330}table{border-collapse:collapse;font-size:13px}td,th{padding:.5rem;border:1px solid #ccd5dc}th{background:#eef3f7}img{max-width:100%}.scroll{overflow:auto}</style>'
    body += '<h1>Simple vs. detailed prompt benchmark</h1>'
    totals = {k: sum(r[k] for r in report['summary']) for k in ('completed', 'failed', 'skipped', 'pending')}
    body += '<p>Saved predictions: ' + ', '.join(f'{v:,} {k}' for k, v in totals.items()) + '.</p>'
    diagnostics = [(name, label) for name, label in [
        ('execution_summary.json', 'Execution times and gates'), ('postrun_validation.json', 'Resume and remaining errors'),
        ('independent_audit.json', 'Independent audit'), ('abstention_investigation.json', 'Abstention probes')]
        if (output.parent / name).exists()]
    if diagnostics:
        body += '<p>' + ' · '.join(f'<a href="../{name}">{label}</a>' for name, label in diagnostics) + '</p>'
    exports = [
            ('performance.csv', 'Performance CSV'), ('paired_differences.csv', 'Prompt comparisons'),
            ('modality_differences.csv', 'Modality comparisons'), ('model_comparisons.csv', 'Model comparisons'),
            ('photo_availability.csv', 'Photo availability strata'), ('false_positives.csv', 'False positives'),
            ('predictions.csv', 'Predictions'), ('report.json', 'Report JSON')]
    if (output/'raw_attempts.jsonl').exists():
        exports.append(('raw_attempts.jsonl', 'Raw attempts (large file)'))
    else:
        exports.append(('../checkpoint.sqlite', 'Raw requests and responses (SQLite)'))
    body += '<p>Download: ' + ' · '.join(
        f'<a href="{name}">{label}</a>' for name, label in exports) + '</p>'
    body += '<p>' + escape(' '.join(report['notes'])) + '</p>'
    body += '<p>Tradeoff policy: ' + escape(dump(report['policy'])) + '</p>'
    body += '<h2>Performance by model and modality</h2><div class="scroll">' + table(report['summary'],
        ['model', 'adapter', 'modality', 'variant', 'planned', 'completed', 'failed', 'skipped', 'pending',
         'verified_flag_n', 'verified_f1', 'weak_f1', 'verified_category_n', 'verified_category_accuracy',
         'weak_precision','weak_recall','weak_fp','weak_flag_accuracy','original_agreement','current_agreement','image_cases_sent_n',
         'flag_abstention_rate', 'category_abstention_rate', 'mean_input_tokens', 'mean_output_tokens',
         'mean_latency_seconds', 'mean_cost_usd']) + '</div>'
    body += '<h2>Paired differences (detailed minus simple)</h2><div class="scroll">' + table(report['paired'],
        ['model', 'modality', 'paired_n', 'verified_flag_n','delta_weak_f1','delta_weak_f1_ci_low','delta_weak_f1_ci_high', 'delta_verified_f1', 'delta_verified_f1_ci_low',
         'delta_verified_f1_ci_high', 'delta_verified_category_accuracy', 'delta_mean_total_tokens',
         'delta_mean_latency_seconds', 'delta_mean_cost_usd', 'tradeoff_assessment']) + '</div>'
    for name in charts:
        body += f'<p><img src="{name}" alt="{name}"></p>'
    body += '<p>Full precision/recall/F1, agreement, false positives and batch summaries are in the accompanying CSV and JSON files. Exact raw requests/responses remain in the checkpoint SQLite database and any supplied raw JSONL export.</p></html>'
    (output / 'index.html').write_text(body, encoding='utf-8')
