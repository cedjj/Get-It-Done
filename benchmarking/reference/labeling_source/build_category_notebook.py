"""Build and execute the self-contained, filtering-only review notebook."""

from pathlib import Path
from textwrap import dedent
import os

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/category_mapping_and_resolution_filtering.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md("""
# Get It Done: category mapping and resolution filtering

**Scope: deterministic filtering only. No LLM calls, description classification,
misroute labels, or semantic assessment of action text.**

The main population is `Get_it_done_data/cases_2026_11_16.csv`. The existing
`cases_3_plus_internal_referrals.csv` is processed separately as a companion
population, not joined into or added to the full population. Its prior extraction
uses **at least three**, not strictly more than three, internal referrals.

This notebook preserves raw fields, applies the mapping proposed in the review
conversation, and retains uncertain routing combinations for later semantic
review. A candidate is a **category difference requiring review**, not a verified
misroute. A positive closure reason is not proof of completed work.

Run all cells from the workspace root, or set `PROJECT_ROOT` below. Only Python's
standard library plus pandas/IPython for display are used. Source CSVs are never
modified. Outputs are written to `outputs/category_filtering` and replaced on rerun.
""")

code("""
from pathlib import Path
from collections import Counter
from functools import lru_cache
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
import re
import time
import unicodedata

import pandas as pd
from IPython.display import display

PROJECT_ROOT = next(p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / 'Get_it_done_data').is_dir())
assert (PROJECT_ROOT / 'Get_it_done_data').is_dir(), 'Set PROJECT_ROOT to the data workspace.'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'category_filtering'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'full_export': PROJECT_ROOT / 'Get_it_done_data' / 'cases_2026_11_16.csv',
    'three_plus_referrals': PROJECT_ROOT / 'outputs/category_filtering/cases_3_plus_internal_referrals.csv',
}
OUTPUT_NAMES = {
    'full_export': 'suspicious_cases.csv',
    'three_plus_referrals': 'suspicious_cases_3plus_referrals.csv',
}
APPROVED_REASONS = ('Problem resolved', 'Issue Resolved', 'Compliance Achieved')
MAPPING_VERSION = '2026-11-17-reviewed-proposal-v1'
csv.field_size_limit(100_000_000)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 100)
print('Approved closure reasons:', ', '.join(APPROVED_REASONS))
print('Outputs:', OUTPUT_DIR)
""")

md("""
## Mapping policy and sources

1. Normalize Unicode, capitalization, dash variants, and repeated whitespace.
   Apply only the explicit aliases below. Preserve the original text in the CSV.
2. Apply **directional** routine-refinement pairs. Do not infer that every pair
   sharing a destination is equivalent. No substring or fuzzy matching is used.
3. Keep conditional asset-, location-, enforcement-, and department-based routes
   in the review queue. Their plausibility cannot be established from category
   names alone. An unlisted pair is also retained, not automatically labeled wrong.
4. Exclude `Other` as an original category and missing categories from the core
   queue, with separate counts. Greater specificity does not establish an error.
   A specific original category changing to `Other` stays in the review queue.
5. All category relations are structural bookkeeping, not case outcome labels.
   Even equivalent/refinement pairs may conceal an inappropriate initial category;
   a future false-negative audit should sample those excluded records.

Provenance:

- The category mapping is the **proposal reviewed in this conversation**, not an
  official City taxonomy or an empirically validated classifier.
- `FieldsForSnowflake_Updated (2).xlsx`, Final Fields: `Original Type` is described
  as the original report category. `Current Type` is not documented as the actual
  resolving department. Initial selection may also have been made by staff.
- `FY27 ROW Support Matrix 9-11-2026 FINAL DRAFT.xlsx`: supports conditional
  ownership/location and multi-department routes. It contains draft revisions and
  unresolved language, and is not a complete historical routing authority.
- [City service guide](https://www.sandiego.gov/sites/default/files/get-it-done-guidelines.pdf)
  and [City guidance on interpreting reports](https://data.sandiego.gov/help/articles/tips-getitdone-311-requests/).
  Historical menus and department responsibilities may differ from current guidance.

The previous proposal called some routes compatible **when the description fits**.
Here those conditional routes are retained; the notebook does not silently assume
that prerequisite has been met.
""")

code("""
@lru_cache(maxsize=100_000)
def normalize_text(value):
    text = unicodedata.normalize('NFKC', value or '')
    text = re.sub(r'[\\u2010-\\u2015\\u2212]', '-', text)
    return re.sub(r'\\s+', ' ', text).strip().casefold()

# These aliases denote explicit naming relationships, not common departments.
ALIAS_GROUPS = {
    'Bicycle-Related Issue': ['Bicycle Related Issue', 'Bicycle-Related Issue'],
    'Container Left Out': ['Container Left Out', 'Containers Left Out Beyond Collection Hours'],
    'Dumpster Encroachment': ['Dumpster Encroachment', 'Dumpster Encroaching on Public Right of Way'],
    'Missed Collection': ['Missed Collection', 'Missed Trash'],
    'Parking': ['Parking', 'Parking Issue'],
    'Homeless Outreach': ['Outreach', 'Homeless Outreach'],
    'Street Light Maintenance': ['Street Light', 'Street Light Maintenance'],
    'Traffic Sign Maintenance': ['Traffic Sign', 'Traffic Sign Maintenance'],
    'Vegetation Encroachment': ['Vegetation Encroachment', 'Vegetation Encroachments'],
}
ALIASES = {
    normalize_text(alias): normalize_text(canonical)
    for canonical, variants in ALIAS_GROUPS.items() for alias in variants
}

@lru_cache(maxsize=100_000)
def canonical_category(value):
    normalized = normalize_text(value)
    return ALIASES.get(normalized, normalized)

ORIGINAL_VALUES = [
    'ADA Assisted Collection', 'ADA Compliance', 'Bicycle Related Issue',
    'Bicycle-Related Issue', 'Biohazard', 'COVID-19', 'Container Left Out',
    'Container Request', 'Containers Left Out Beyond Collection Hours', 'Curb',
    'Damaged Guardrail', 'Dead Animal', 'Dumpster Encroaching on Public Right of Way',
    'Dumpster Encroachment', 'Dumpster Overflow', 'ESD Code', 'EV Charger',
    'Encampment', 'Faded Striping', 'Graffiti', 'HAZMAT', 'Hazardous Waste',
    'Illegal Discharge', 'Illegal Dumping', 'Missed Collection', 'Missed Trash',
    'Other', 'Outreach', 'Over Irrigation', 'Parking', 'Parking Issue',
    'Parking Meter', 'Parks Issue', 'Pothole', 'STRO Violation',
    'Scavenging of Blue Recycling or Black Trash Containers', 'Shared Mobility Device',
    'Sidewalk', 'Storm Drain', 'Street Flooded', 'Street Light', 'Street Sweeping',
    'Traffic Sign', 'Traffic Signal', 'Tree Maintenance', 'Vegetation Encroachment',
    'Vegetation Encroachments', 'Waste on Private Property',
]
CURRENT_VALUES = [
    'ADA Compliance', 'Bicycle-Related Issue', 'Biohazard',
    'Building and Land Use Enforcement', 'COVID-19', 'Construction and Demolition Deposits',
    'Container Request', 'Dead Animal', 'Development Services - Code Enforcement',
    'ESD Collections', 'ESD Recycling', 'EV Charger', 'Encampment',
    'Environmental Services Code Compliance', 'Graffiti', 'Graffiti - Code Enforcement',
    'Graffiti - Private', 'Graffiti - Public', 'HAZMAT', 'Homeless Outreach',
    'Illegal Dumping', 'Missed Collection', 'Other', 'Oversized Vehicle',
    'PUD Code Compliance', 'Parking', 'Parking - 72-Hours', 'Parking Violation',
    'Parks Issue', 'Pavement Maintenance', 'Pothole', 'ROW Maintenance',
    'Right-of-Way Code Enforcement', 'STRO Violation', 'Shared Mobility Device',
    'Sidewalk Repair Issue', 'Stormwater', 'Stormwater Code Enforcement',
    'Street Flooded', 'Street Light Maintenance', 'Street Sweeping',
    'Traffic Engineering', 'Traffic Sign Maintenance', 'Traffic Signal Issue',
    'Traffic Signal Timing', 'Trash/Recycling Collection', 'Tree Maintenance',
    'UUP', 'Vegetation Encroachment', 'Waste on Private Property', 'Weed Cleanup',
]

# Canonical original -> current targets. All are directional, never transitive.
ROUTINE = {
    'Graffiti': ['Graffiti - Public', 'Graffiti - Private', 'Graffiti - Code Enforcement'],
    'Parking': ['Parking - 72-Hours', 'Parking Violation', 'Oversized Vehicle'],
    'Pothole': ['Pavement Maintenance'],
    'Sidewalk': ['Sidewalk Repair Issue'],
    'Traffic Signal': ['Traffic Signal Issue', 'Traffic Signal Timing'],
}

# Do not discard these records: the proposal explicitly depends on case context.
CONDITIONAL = {
    'ADA Assisted Collection': ['ESD Collections', 'Trash/Recycling Collection', 'Environmental Services Code Compliance'],
    'ADA Compliance': ['Traffic Engineering', 'Pavement Maintenance', 'Sidewalk Repair Issue', 'Right-of-Way Code Enforcement'],
    'Bicycle-Related Issue': ['ROW Maintenance', 'Street Sweeping', 'Traffic Engineering', 'Pavement Maintenance', 'Parks Issue', 'Stormwater', 'Vegetation Encroachment', 'Weed Cleanup'],
    'Biohazard': ['Illegal Dumping', 'Environmental Services Code Compliance'],
    'Container Request': ['ESD Collections', 'Trash/Recycling Collection'],
    'Container Left Out': ['Environmental Services Code Compliance', 'Right-of-Way Code Enforcement', 'ESD Collections'],
    'Curb': ['Pavement Maintenance', 'Traffic Engineering', 'ADA Compliance', 'Sidewalk Repair Issue', 'ROW Maintenance', 'Right-of-Way Code Enforcement'],
    'Damaged Guardrail': ['ROW Maintenance', 'Pavement Maintenance', 'Parks Issue', 'Traffic Engineering'],
    'Dead Animal': ['Environmental Services Code Compliance', 'Parks Issue', 'Stormwater'],
    'Dumpster Encroachment': ['Environmental Services Code Compliance', 'Right-of-Way Code Enforcement'],
    'Dumpster Overflow': ['Environmental Services Code Compliance', 'ESD Collections', 'Trash/Recycling Collection', 'ESD Recycling'],
    'ESD Code': ['Environmental Services Code Compliance'],
    'EV Charger': ['Parking Violation'],
    'Encampment': ['Homeless Outreach', 'Environmental Services Code Compliance', 'Illegal Dumping', 'Parks Issue', 'Stormwater'],
    'Homeless Outreach': ['Encampment'],
    'Faded Striping': ['Traffic Engineering', 'ROW Maintenance', 'Parks Issue'],
    'Graffiti': ['Environmental Services Code Compliance', 'Building and Land Use Enforcement', 'Development Services - Code Enforcement', 'Parks Issue', 'Stormwater', 'ROW Maintenance'],
    'Hazardous Waste': ['Environmental Services Code Compliance', 'HAZMAT', 'Biohazard', 'Illegal Dumping'],
    'Illegal Discharge': ['Stormwater Code Enforcement', 'Stormwater', 'PUD Code Compliance', 'Building and Land Use Enforcement', 'Development Services - Code Enforcement'],
    'Over Irrigation': ['PUD Code Compliance', 'Stormwater Code Enforcement', 'Stormwater', 'Street Flooded', 'Parks Issue'],
    'Illegal Dumping': ['Waste on Private Property', 'Environmental Services Code Compliance', 'Parks Issue', 'Stormwater', 'ROW Maintenance', 'ESD Recycling'],
    'Missed Collection': ['Missed Collection', 'ESD Collections', 'Trash/Recycling Collection'],
    'Parking': ['Traffic Engineering', 'Right-of-Way Code Enforcement', 'Development Services - Code Enforcement', 'Building and Land Use Enforcement'],
    'Parking Meter': ['Parking', 'Parking Violation', 'Traffic Engineering'],
    'Parks Issue': ['Encampment', 'Tree Maintenance', 'Vegetation Encroachment', 'Weed Cleanup', 'Street Light Maintenance', 'Pavement Maintenance', 'ROW Maintenance', 'Graffiti - Public', 'Illegal Dumping', 'Dead Animal', 'Street Sweeping', 'Stormwater'],
    'Scavenging of Blue Recycling or Black Trash Containers': ['ESD Collections', 'ROW Maintenance'],
    'Shared Mobility Device': ['ROW Maintenance', 'Traffic Engineering'],
    'Sidewalk': ['Pavement Maintenance', 'Right-of-Way Code Enforcement', 'ROW Maintenance', 'Vegetation Encroachment', 'Weed Cleanup', 'Tree Maintenance', 'Illegal Dumping', 'PUD Code Compliance', 'Stormwater'],
    'Storm Drain': ['Stormwater', 'Stormwater Code Enforcement', 'Building and Land Use Enforcement', 'Development Services - Code Enforcement', 'PUD Code Compliance'],
    'Street Flooded': ['Stormwater', 'Stormwater Code Enforcement', 'PUD Code Compliance'],
    'Street Sweeping': ['Stormwater', 'ROW Maintenance', 'Parks Issue'],
    'Traffic Sign Maintenance': ['Traffic Engineering'],
    'Traffic Signal': ['Traffic Engineering'],
    'Tree Maintenance': ['Parks Issue', 'Vegetation Encroachment', 'Weed Cleanup', 'ROW Maintenance', 'Stormwater', 'Right-of-Way Code Enforcement'],
    'Vegetation Encroachment': ['Weed Cleanup', 'ROW Maintenance', 'Right-of-Way Code Enforcement', 'Parks Issue', 'Stormwater', 'Tree Maintenance'],
    'Waste on Private Property': ['Environmental Services Code Compliance', 'Building and Land Use Enforcement', 'Development Services - Code Enforcement'],
}

def pair_set(mapping):
    return {(canonical_category(o), canonical_category(c))
            for o, targets in mapping.items() for c in targets
            if canonical_category(o) != canonical_category(c)}

ROUTINE_PAIRS = pair_set(ROUTINE)
CONDITIONAL_PAIRS = pair_set(CONDITIONAL)
assert not ROUTINE_PAIRS & CONDITIONAL_PAIRS
KNOWN_ORIGINAL = {canonical_category(x) for x in ORIGINAL_VALUES}
KNOWN_CURRENT = {canonical_category(x) for x in CURRENT_VALUES}

@lru_cache(maxsize=100_000)
def mapping_relation(original, current):
    '''Structural pair relationship only; never a semantic misroute decision.'''
    if not original or not current:
        return 'missing_category'
    if original == 'other':
        return 'original_other'
    if original == current:
        return 'equivalent'
    if (original, current) in ROUTINE_PAIRS:
        return 'routine_refinement'
    if current == 'other':
        return 'current_other'
    if original not in KNOWN_ORIGINAL or current not in KNOWN_CURRENT or current == 'uup':
        return 'undocumented_category'
    if (original, current) in CONDITIONAL_PAIRS:
        return 'conditional_routing'
    return 'unexplained_category_difference'

REVIEW_RELATIONS = {
    'conditional_routing', 'unexplained_category_difference',
    'current_other', 'undocumented_category',
}
APPROVED_REASON_KEYS = {normalize_text(x) for x in APPROVED_REASONS}

mapping_rows = []
for original in ORIGINAL_VALUES:
    canonical = canonical_category(original)
    mapping_rows.append({
        'original_value': original,
        'canonical_original': canonical,
        'routine_current_targets': '; '.join(sorted(c for o, c in ROUTINE_PAIRS if o == canonical)),
        'conditional_current_targets_RETAIN_FOR_REVIEW': '; '.join(sorted(c for o, c in CONDITIONAL_PAIRS if o == canonical)),
        'policy': 'exclude generic original' if canonical == 'other' else 'same canonical value compatible; unlisted changes retained',
        'mapping_version': MAPPING_VERSION,
    })
mapping_table = pd.DataFrame(mapping_rows)
display(mapping_table)
print('Canonical values are comparison keys; raw source values remain unchanged.')
""")

md("""
## Resolution filter and review-queue definition

Eligibility is an exact normalized match to `STATUS = Closed` **and** one of the
three approved `REASON` values. No other reasons, including other apparently
positive reasons, are added. `ISCLOSED` and public status fields do not override
the requested `STATUS` filter.

After mapping, two scopes are reported:

- **Unexplained differences:** eligible, specified original/current categories
  without an explicit compatible/refinement/conditional relationship.
- **Inclusive semantic-review queue (exported):** unexplained differences **plus**
  conditional routing, undocumented categories, and specific originals changing
  to `Other`. Conditional combinations are retained to avoid prematurely deciding
  whether the case-specific prerequisite holds.

Original `Other` and missing categories are counted separately. They are not
treated as negative examples or included in the core queue. Description and action
fields are checked only for presence; no text is interpreted. Blank action text
does not remove a record at this stage, and nonblank text is not proof of resolution.

All source columns are preserved in the candidate export. Added fields describe
normalization, mapping relation, and text availability only. No misroute label or
LLM prediction field is generated.
""")

code("""
# Meaningful boundary checks: prevent broad aliases or transitive mappings.
assert canonical_category('Bicycle Related Issue') == canonical_category('Bicycle-Related Issue')
assert canonical_category('  STREET LIGHT  ') == canonical_category('Street Light Maintenance')
assert canonical_category('Vegetation Encroachments') == canonical_category('Vegetation Encroachment')
assert canonical_category('Parking Meter') != canonical_category('Parking Issue')
assert canonical_category('ADA Assisted Collection') != canonical_category('ADA Compliance')
assert canonical_category('Hazardous Waste') != canonical_category('HAZMAT')
assert mapping_relation(canonical_category('Pothole'), canonical_category('Pavement Maintenance')) == 'routine_refinement'
assert mapping_relation(canonical_category('Curb'), canonical_category('Pavement Maintenance')) == 'conditional_routing'
assert mapping_relation(canonical_category('Pothole'), canonical_category('Curb')) not in {'equivalent', 'routine_refinement'}
assert mapping_relation(canonical_category('Street Light'), canonical_category('Traffic Signal Issue')) == 'unexplained_category_difference'
assert mapping_relation('other', 'pothole') == 'original_other'
assert mapping_relation('', 'pothole') == 'missing_category'
assert normalize_text('Externally Referred') not in APPROVED_REASON_KEYS
assert normalize_text('Issue Resolved') in APPROVED_REASON_KEYS
print('Mapping boundaries and exact reason membership verified.')
""")

code("""
def write_csv_atomic(path, columns, rows):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def source_fingerprint(path):
    '''Cheap rerun identity: size, mtime and a header/sample hash, not a full-file hash.'''
    stat = path.stat()
    with path.open('rb') as stream:
        sample_hash = hashlib.sha256(stream.read(1_048_576)).hexdigest()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'first_megabyte_sha256': sample_hash}

def scan_population(population, source_path):
    '''Stream a source once, collect counts and atomically export review candidates.'''
    started = time.monotonic()
    counts = Counter()
    relations_all = Counter()
    relations_eligible = Counter()
    inventories = {'ORIGINAL_TYPE': Counter(), 'CURRENT_TYPE': Counter()}
    pairs_all = Counter()
    pairs_eligible = Counter()
    reasons_closed = Counter()
    seen_ids = set()  # IDs only, not complete case records; used to verify case counts.
    candidate_ids = set()
    metadata = source_fingerprint(source_path)
    target = OUTPUT_DIR / OUTPUT_NAMES[population]
    assert target.resolve() != source_path.resolve()
    temporary = target.with_suffix('.csv.partial')
    extras = [
        'FILTER_ORIGINAL_CANONICAL', 'FILTER_CURRENT_CANONICAL',
        'FILTER_MAPPING_RELATION', 'FILTER_MAPPING_VERSION',
        'FILTER_HAS_DESCRIPTION', 'FILTER_HAS_ACTION_TEXT',
    ]
    with source_path.open('r', encoding='utf-8-sig', newline='') as source, \\
         temporary.open('w', encoding='utf-8-sig', newline='') as output:
        reader = csv.reader(source)
        header = next(reader)
        assert len(header) == len(set(header)), 'Duplicate source column names.'
        assert not set(extras) & set(header), 'Derived columns already exist in source.'
        required = {'ID', 'CASENUMBER', 'STATUS', 'REASON', 'ORIGINAL_TYPE',
                    'CURRENT_TYPE', 'DESCRIPTION', 'ACTION_TAKEN__C', 'ACTION_TAKEN_OTHER__C'}
        assert required <= set(header), f'Missing fields: {required - set(header)}'
        ix = {name: header.index(name) for name in required}
        writer = csv.writer(output)
        writer.writerow(header + extras)
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f'{population}: malformed record ending at physical line {reader.line_num}')
            counts['source_rows'] += 1
            key = row[ix['ID']].strip()
            if not key:
                counts['missing_case_id'] += 1
            elif key in seen_ids:
                counts['duplicate_case_id_rows'] += 1
            else:
                seen_ids.add(key)
            original_raw, current_raw = row[ix['ORIGINAL_TYPE']], row[ix['CURRENT_TYPE']]
            original = canonical_category(original_raw)
            current = canonical_category(current_raw)
            relation = mapping_relation(original, current)
            relations_all[relation] += 1
            inventories['ORIGINAL_TYPE'][original_raw] += 1
            inventories['CURRENT_TYPE'][current_raw] += 1
            raw_pair = (original_raw, current_raw)
            pairs_all[raw_pair] += 1
            if original_raw != current_raw:
                counts['raw_different_all'] += 1
            if original and current and original != current:
                counts['canonical_different_all'] += 1
            if original and current and original != 'other':
                counts['specified_original_and_current_all'] += 1
                if relation in REVIEW_RELATIONS:
                    counts['mapping_review_before_resolution'] += 1
            if normalize_text(row[ix['STATUS']]) != 'closed':
                continue
            counts['closed'] += 1
            reasons_closed[row[ix['REASON']]] += 1
            if relation in REVIEW_RELATIONS:
                counts['mapping_review_closed'] += 1
            if normalize_text(row[ix['REASON']]) not in APPROVED_REASON_KEYS:
                continue
            counts['closed_approved_reason'] += 1
            relations_eligible[relation] += 1
            pairs_eligible[raw_pair] += 1
            if relation not in REVIEW_RELATIONS:
                continue
            has_description = bool(row[ix['DESCRIPTION']].strip())
            has_action = bool(row[ix['ACTION_TAKEN__C']].strip() or row[ix['ACTION_TAKEN_OTHER__C']].strip())
            counts['exported_review_queue'] += 1
            counts['queue_has_description'] += int(has_description)
            counts['queue_missing_description'] += int(not has_description)
            counts['queue_has_action_text'] += int(has_action)
            counts['queue_missing_action_text'] += int(not has_action)
            counts['queue_has_description_and_action_text'] += int(has_description and has_action)
            counts['queue_has_parent_reference'] += int(any(
                row[header.index(field)].strip() for field in ('PARENTID', 'PARENT_CASE_CASE_NUMBER') if field in header
            ))
            if key:
                candidate_ids.add(key)
            writer.writerow(row + [original, current, relation, MAPPING_VERSION,
                                   has_description, has_action])
            if counts['exported_review_queue'] % 25_000 == 0:
                print(f'{population}: {counts["source_rows"]:,} read; {counts["exported_review_queue"]:,} retained', flush=True)

    counts['distinct_case_ids'] = len(seen_ids)
    counts['distinct_candidate_ids'] = len(candidate_ids)
    assert sum(relations_all.values()) == counts['source_rows']
    assert sum(relations_eligible.values()) == counts['closed_approved_reason']
    assert sum(relations_eligible[r] for r in REVIEW_RELATIONS) == counts['exported_review_queue']
    assert sum(pairs_all.values()) == counts['source_rows']
    assert sum(pairs_eligible.values()) == counts['closed_approved_reason']
    assert counts['queue_has_description'] + counts['queue_missing_description'] == counts['exported_review_queue']
    assert counts['queue_has_action_text'] + counts['queue_missing_action_text'] == counts['exported_review_queue']
    # Case counts are only presented as such after verifying unique, present IDs.
    assert counts['missing_case_id'] == 0, 'Missing IDs: inspect before presenting case counts.'
    assert counts['duplicate_case_id_rows'] == 0, 'Repeated IDs: reconcile snapshots before presenting case counts.'
    assert counts['distinct_candidate_ids'] == counts['exported_review_queue']
    after = source_fingerprint(source_path)
    assert metadata == after, 'Source changed during processing.'
    temporary.replace(target)
    elapsed = time.monotonic() - started
    print(f'{population}: completed {counts["source_rows"]:,} cases in {elapsed:.1f}s; retained {counts["exported_review_queue"]:,}', flush=True)
    return {'counts': dict(counts), 'relations_all': dict(relations_all),
            'relations_eligible': dict(relations_eligible), 'inventories': inventories,
            'pairs_all': pairs_all, 'pairs_eligible': pairs_eligible,
            'closed_reasons': reasons_closed, 'candidate_ids': candidate_ids,
            'source': metadata, 'output': str(target), 'seconds': elapsed}

results = {}
for population, source_path in SOURCES.items():
    assert source_path.is_file(), source_path
    results[population] = scan_population(population, source_path)

assert results['three_plus_referrals']['candidate_ids'] <= results['full_export']['candidate_ids'], 'Companion candidates must also appear in the full export.'
""")

md("""
## Counts after each step

The cumulative view below follows the requested sequence: source, normalization,
category screening, Closed status, approved resolution reason, then retained
category differences. Normalization itself removes no records.

The companion diagnostic view starts with closure eligibility and partitions it
by mapping relation. These are two views of the same filters, not extra filtering
or additional populations to add together.
""")

code("""
flow_rows = []
for population, result in results.items():
    c, rel = result['counts'], result['relations_eligible']
    stages = [
        ('1. Source cases', c['source_rows']),
        ('2. Normalized categories (no rows removed)', c['source_rows']),
        ('3. Specified original/current; original is not Other', c.get('specified_original_and_current_all', 0)),
        ('4. Category-review queue before closure filters', c.get('mapping_review_before_resolution', 0)),
    ]
    stages += [
        ('5. Category-review queue with STATUS = Closed', c.get('mapping_review_closed', 0)),
        ('6. Above + approved resolution reason (exported)', c.get('exported_review_queue', 0)),
    ]
    previous = None
    for stage, number in stages:
        flow_rows.append({'population': population, 'stage': stage, 'cases_remaining': number,
                          'removed_since_previous': None if previous is None else previous - number})
        previous = number

flow = pd.DataFrame(flow_rows)
flow['removed_since_previous'] = pd.array(flow['removed_since_previous'], dtype='Int64')
display(flow)

summary_rows = []
for population, result in results.items():
    c = result['counts']
    summary_rows.append({'population': population, **{
        name: c.get(name, 0) for name in [
            'source_rows', 'distinct_case_ids', 'closed', 'closed_approved_reason',
            'raw_different_all', 'canonical_different_all', 'exported_review_queue',
            'queue_has_description', 'queue_missing_description',
            'queue_has_action_text', 'queue_missing_action_text',
            'queue_has_description_and_action_text', 'queue_has_parent_reference',
            'missing_case_id', 'duplicate_case_id_rows',
        ]}})
summary = pd.DataFrame(summary_rows).set_index('population').T
display(summary)

relation_rows = []
for population, result in results.items():
    for relation in sorted(set(result['relations_all']) | set(result['relations_eligible'])):
        relation_rows.append({'population': population, 'mapping_relation': relation,
            'all_cases': result['relations_all'].get(relation, 0),
            'closed_approved_reason_cases': result['relations_eligible'].get(relation, 0),
            'retained_for_review': relation in REVIEW_RELATIONS})
relation_counts = pd.DataFrame(relation_rows)
display(relation_counts)

reason_rows = [{'population': population, 'raw_reason': reason, 'closed_cases': count,
                'approved': normalize_text(reason) in APPROVED_REASON_KEYS}
               for population, result in results.items()
               for reason, count in result['closed_reasons'].most_common()]
reason_counts = pd.DataFrame(reason_rows)
display(reason_counts[reason_counts['approved']])
""")

code("""
inventory_rows, pair_rows = [], []
for population, result in results.items():
    for field, frequencies in result['inventories'].items():
        for raw_value, number in sorted(frequencies.items()):
            inventory_rows.append({'population': population, 'field': field,
                'raw_value': raw_value, 'canonical_value': canonical_category(raw_value), 'cases': number})
    for (original, current), number in sorted(result['pairs_all'].items()):
        o, c = canonical_category(original), canonical_category(current)
        pair_rows.append({'population': population, 'original_type': original, 'current_type': current,
            'original_canonical': o, 'current_canonical': c, 'mapping_relation': mapping_relation(o, c),
            'all_cases': number, 'closed_approved_reason_cases': result['pairs_eligible'][(original, current)]})
inventory = pd.DataFrame(inventory_rows)
pair_counts = pd.DataFrame(pair_rows)
display(inventory.groupby(['population', 'field']).agg(
    distinct_values_including_blank=('raw_value', 'size'), cases=('cases', 'sum')))
display(pair_counts[pair_counts['closed_approved_reason_cases'] > 0].sort_values(
    ['population', 'closed_approved_reason_cases'], ascending=[True, False]).head(30))

# All catalog values must have been observed in the complete source; new values are surfaced.
full_inventory = results['full_export']['inventories']
for field, expected in [('ORIGINAL_TYPE', ORIGINAL_VALUES), ('CURRENT_TYPE', CURRENT_VALUES)]:
    observed = set(full_inventory[field]) - {''}
    print(field, 'nonblank distinct:', len(observed),
          'new values:', sorted(observed - set(expected)),
          'catalog values absent:', sorted(set(expected) - observed))
""")

md("""
## Verify and save the outputs

Re-read each exported CSV to verify its row count, unique IDs, exact closure
eligibility, and mapping relationship. This check does not interpret descriptions
or action text. No description samples or provisional semantic decisions are shown.

Candidate files are written to `.partial` files and renamed only after the scan's
checks pass. An interruption leaves the prior completed CSV intact. Restarting
this initial deterministic stage re-scans the source; no row-offset resume is
claimed. Each population's completed outputs can be validated independently.
""")

code("""
for population, result in results.items():
    verified = 0
    ids = set()
    with Path(result['output']).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            verified += 1
            assert normalize_text(row['STATUS']) == 'closed'
            assert normalize_text(row['REASON']) in APPROVED_REASON_KEYS
            original, current = canonical_category(row['ORIGINAL_TYPE']), canonical_category(row['CURRENT_TYPE'])
            assert original and current and original != 'other' and original != current
            assert row['FILTER_MAPPING_RELATION'] == mapping_relation(original, current)
            assert row['FILTER_MAPPING_RELATION'] in REVIEW_RELATIONS
            assert row['FILTER_ORIGINAL_CANONICAL'] == original
            assert row['FILTER_CURRENT_CANONICAL'] == current
            assert row['ID'] not in ids
            ids.add(row['ID'])
    assert verified == result['counts']['exported_review_queue']
    assert ids == result['candidate_ids']
    print(f'{population}: independently verified {verified:,} exported cases.')

tables = {
    'filter_counts.csv': flow,
    'population_summary.csv': pd.DataFrame(summary_rows),
    'mapping_relation_counts.csv': relation_counts,
    'category_mapping.csv': mapping_table,
    'category_inventory.csv': inventory,
    'category_pair_counts.csv': pair_counts,
    'closed_reason_counts.csv': reason_counts,
}
for filename, table in tables.items():
    write_csv_atomic(OUTPUT_DIR / filename, list(table.columns), table.to_dict('records'))

mapping_payload = {'aliases': ALIAS_GROUPS, 'routine': ROUTINE, 'conditional': CONDITIONAL,
                   'approved_reasons': APPROVED_REASONS, 'mapping_version': MAPPING_VERSION}
mapping_hash = hashlib.sha256(json.dumps(mapping_payload, sort_keys=True).encode()).hexdigest()
manifest = {
    'completed_utc': datetime.now(timezone.utc).isoformat(),
    'python': platform.python_version(), 'mapping_version': MAPPING_VERSION,
    'mapping_sha256': mapping_hash, 'mapping_policy': mapping_payload,
    'llm_calls': 0, 'semantic_labels_generated': False,
    'populations': {population: {key: result[key] for key in [
        'counts', 'relations_all', 'relations_eligible', 'source', 'output', 'seconds'
    ]} for population, result in results.items()},
}
manifest_path = OUTPUT_DIR / 'run_manifest.json'
temporary = manifest_path.with_suffix('.json.partial')
temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
temporary.replace(manifest_path)
print('Saved mapping, counts, audit tables, two candidate CSVs, and run manifest.')
print('LLM calls: 0. Semantic misroute labels: none.')
""")

code("""
workload_rows = []
for population, result in results.items():
    c, r = result['counts'], result['relations_eligible']
    available = c.get('queue_has_description', 0)
    workload_rows.append({
        'population': population,
        'exported_candidates': c.get('exported_review_queue', 0),
        'unexplained_differences': r.get('unexplained_category_difference', 0),
        'conditional_routes_retained': r.get('conditional_routing', 0),
        'current_Other_or_undocumented': r.get('current_other', 0) + r.get('undocumented_category', 0),
        'DESCRIPTION_available': available,
        'DESCRIPTION_and_action_text_available': c.get('queue_has_description_and_action_text', 0),
        'illustrative_human_hours_at_1_min_per_description': round(available / 60, 2),
        'yield_needed_for_100_positives_percent_NOT_A_FORECAST': round(10000 / available, 3) if available else None,
    })
display(pd.DataFrame(workload_rows).set_index('population').T)
print('Human time is an illustrative assumption; complex cases can take longer.')
print('The yield threshold is arithmetic, not an estimate of genuine misroute prevalence.')
print('Action-text availability does not demonstrate completed resolution.')
""")

md("""
## Proposed semantic review: not executed

Every retained record with a usable `DESCRIPTION` would receive an individual
semantic assessment. Category-name rules would not decide whether it is a genuine
misroute. Records without descriptions would require other original evidence or
remain unassessable, not be forced into an outcome.

**Separate the assessments to limit hindsight bias:**

1. Give the model the description, relevant original-report context, and a
   versioned service-definition reference. Hide `CURRENT_TYPE`, resolution text,
   and referral count. First identify the reported issue and all reasonable
   resident-facing categories, allowing several acceptable choices.
2. Compare `ORIGINAL_TYPE` with those acceptable choices and the actual inclusion/
   exclusion rules. Return reasonable selection, clearly inappropriate selection,
   or insufficient evidence. Require a brief evidence explanation, exact supporting
   description excerpts, and service-definition references. A better category does
   not make the original category wrong.
3. In a separate assessment, reveal the current classification and routing/closure
   fields. Decide whether the current classification fits, whether completed work
   or compliance is supported, and whether the handling department is established.
   Do not assume `CURRENT_TYPE` itself proves department ownership or completion.
4. A potential benchmark positive requires all three: inappropriate original,
   appropriate current classification, and corroborated handling. Human review
   should adjudicate every proposed positive and all reviewer disagreements.

**Grounding and uncertainty:** Build the reference from official service scope,
exceptions, location/asset ownership, and effective dates. Keep draft matrix rules
separate. Missing historical rules, edited descriptions, vague wording, multiple
issues, and absent asset ownership should produce abstentions. Neither model
self-reported confidence nor agreement between two model calls proves correctness.

**Pilot and scale:** After approval, use a stratified 150-200-case pilot, including
conditional pairs and samples of excluded equivalents, refinements, Other, and
missing-category cases. Obtain independent human decisions, adjudicate conflicts,
and reserve a held-out validation set. Measure positive precision, exclusion errors,
and abstention rates before choosing a model or claiming reliability.

For production, one case per API request prevents records from being mixed.
Requests can be submitted in batches of 250-1,000 while keeping case decisions
independent. Store a stable case/request key, input hash, prompt/reference/model
versions, per-stage result, token usage, and validation state. Retry only missing or
failed requests. Set a cost ceiling after measuring real pilot token usage. Nothing
in this notebook authenticates to an API or submits a request.

**Benchmark expectations:** This queue favors coverage over precision. Conditional
routes, broad final queues, symptom-to-cause diagnosis, and generic positive closure
messages can all create false positives. Equivalent/refinement exclusions, original
Other, missing categories, and the strict closure-reason list can hide genuine
misroutes. The three-plus-referrals subset is enriched and cannot estimate overall
misroute prevalence. Keep parent/duplicate families together in any future
train/test split. Yield must be measured by the pilot; a large candidate queue alone
does not establish that enough genuine misroutes exist.

**Stop here. No LLM analysis or misroute labeling has been performed.**
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
notebook.metadata['language_info'] = {'name': 'python', 'version': '3'}
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'notebook-cell-{index}', 'exec')
nbf.write(notebook, NOTEBOOK)
print(f'Created {NOTEBOOK}', flush=True)

runtime = ROOT / 'outputs' / 'category_filtering' / '.jupyter_runtime'
runtime.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
client = NotebookClient(notebook, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
try:
    client.execute()
finally:
    nbf.write(notebook, NOTEBOOK)
print(f'Executed and saved {NOTEBOOK}', flush=True)
