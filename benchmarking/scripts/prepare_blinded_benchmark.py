"""Separate allowlisted model inputs from evaluation-only labels; never run inference."""
import argparse
import csv
from pathlib import Path

from project_paths import ROOT
from prompt_benchmark.core import allowed_blinded_column, read_cases, sha, write_json


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory; existing datasets are not overwritten')
    rows = read_cases(source)
    fields = [k for k in rows[0] if allowed_blinded_column(k)]
    inputs = [{k: row[k] for k in fields} for row in rows]
    labels = [{'ID': row['ID'], 'CURRENT_TYPE': row['CURRENT_TYPE'], 'label': row.get('label', ''),
               'LABEL_PROVENANCE': row.get('LABEL_PROVENANCE', ''), 'input_sha256': sha(case)}
              for row, case in zip(rows, inputs)]
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in [('model_inputs.csv', inputs), ('evaluation_labels.csv', labels)]:
        with (output / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    # Read back through the strict blinded loader, not just the serializer.
    assert read_cases(output / 'model_inputs.csv', blinded=True) == inputs
    manifest = {'cases': len(inputs), 'source': str(source.resolve()),
                'source_sha256': sha(source.read_bytes()), 'input_columns': fields,
                'input_sha256': sha((output / 'model_inputs.csv').read_bytes()),
                'evaluation_sha256': sha((output / 'evaluation_labels.csv').read_bytes()),
                'join_key': 'ID; require matching input hash',
                'note': 'ID is retained for joining and never sent to the model. '
                        'Labels/current routing are evaluation-only. Existing labels remain proxies.'}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'results/category_matched_common_support_with_photos.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/blinded_benchmark')
    args = parser.parse_args()
    import json
    print(json.dumps(prepare(args.source, args.output), indent=2))
