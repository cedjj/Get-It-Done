# Paired prompt and modality benchmark

For the photo-enabled, multiple-model run with corrected abstention instructions and optional API adapters, see the [multimodal guide](MULTIMODAL.md). Its output is separate from the preserved first blinded run.

Entry point: `python scripts/run_prompt_benchmark.py`. This is a separate experiment namespace from the stopped semantic review. `semantic_labeling.py`, its request contract, and its production checkpoint remain unchanged. The new Ollama adapter reuses its fixed-loopback API transport. The inspected workspace has no prior multimodal model-adapter, async benchmark or comparative evaluation framework, so those components are supplied here rather than replacing the pinned run.

## Blinded routing evaluation (recommended)

For evaluating original routing from submission-time evidence, hide `CURRENT_TYPE` as well as all misroute labels. The earlier current-visible run is a diagnostic comparison, not a blinded routing-performance estimate. Its high proxy agreement can reflect downstream information. Use physically separate input and evaluation files:

```powershell
# Creates results/blinded_benchmark/model_inputs.csv and evaluation_labels.csv.
# Does not modify the original CSV or run inference.
python scripts/prepare_blinded_benchmark.py

python scripts/run_prompt_benchmark.py plan --blinded --cases results/blinded_benchmark/model_inputs.csv --config scripts/prompt_benchmark/models.blinded.json --output outputs/prompt_benchmark/qwen_blinded_100 --limit 100

# Explicit local inference; run only when intended.
python scripts/run_prompt_benchmark.py run --output outputs/prompt_benchmark/qwen_blinded_100 --modality text --modality text_location --workers 2

# Only scoring reads the original CSV (the held-out evaluation_labels.csv also works).
python scripts/run_prompt_benchmark.py report --output outputs/prompt_benchmark/qwen_blinded_100 --evaluation results/category_matched_common_support_with_photos.csv
```

The input file contains ID for joining, DESCRIPTION, ORIGINAL_TYPE, optional coordinates and public photo URLs. ID/URLs are never model text; enabled images are supplied as bytes. All closure fields, referral counts, model reviews, proxy labels and current routing stay out of the inference CSV. `--blinded` rejects extra CSV columns and prompt references to current routing. Its frozen plan contains no target values; strict request validation rejects unexpected model-input keys. Both blinded prompt variants retain the same JSON schema.

Scoring joins original or held-out labels by unique ID and verifies an input SHA-256 association for every selected case. Joining affects only evaluation memory, never the saved inference plan or requests. Original/current agreement remains an agreement metric, not correct-category accuracy. The original labels remain proxies and require independently reviewed category labels before claims about true category accuracy. Removing current routing does not resolve unverified description/photo/location timing.

The default seed and ID-based selection preserve the same first 100 cases as the current-visible run. Blinded prompts have separate version names and checkpoint identities; old predictions cannot be reused as blinded predictions. The remaining sections describe the original current-visible configuration where applicable.

## Offline validation

```powershell
python -m unittest discover -s tests -v
python scripts/validate_semantic_contract.py
python scripts/run_prompt_benchmark.py smoke
```

Smoke uses two synthetic adapters, three fixture cases, local fixture image bytes, all eight prompt/modality cells, SQLite resume, evaluation, CSV/JSON/JSONL exports, and PNG/HTML reports. It makes **no network or model calls**. Its labels, tokens and performance are fixtures, not findings about model quality. Rerunning smoke reuses the frozen plan and completed predictions.

## Freeze the experiment before inference

```powershell
python scripts/run_prompt_benchmark.py plan --output outputs/prompt_benchmark/trial --limit 100
python scripts/run_prompt_benchmark.py list --output outputs/prompt_benchmark/trial
```

Planning reads the photo-linked development CSV, deterministically orders case IDs with seed 42, selects cases **once**, and partitions them into batches of 100 (the final batch may be smaller). Every configured model and both prompts use those same cases in all four modalities. Omitting `--limit` plans the source file but does not run inference. Duplicate IDs are rejected.

`plan.json` freezes case inputs, labels, batch assignments, model configuration, pricing, category list, common JSON schema, rendered prompts, template versions and SHA-256 hashes. A plan cannot be overwritten; use a new directory when changing cases, photos, definitions, templates, pricing, options or models. Resume and report validate its content hash. Each checkpoint prediction records its prompt variant, version, hash, input hash, model, modality and batch. The key is `(experiment_id, case_id)`; the experiment ID incorporates the complete plan and prompt configuration.

The shipped configuration is [models.local.json](models.local.json). It includes the existing `qwen3:8b`, which is text-only. Its photo cells are recorded as **skipped with a reason**, never evaluated as text-only substitutes. To exercise photographs, add an **already installed** vision-capable Ollama model with a unique `id`, its exact `model` name, `adapter: "ollama"`, and `supports_photos: true`. Any number of installed Ollama models can share a plan. The runner checks installation and vision capability, pins the model digest/runtime on first use and rejects changes during resume. It never pulls models. There are no cloud/paid adapters in this workspace implementation.

Each model may provide `options` and `pricing` with `input_per_million`/`output_per_million` in USD. Local API cost is explicitly zero; hardware and electricity are excluded. Missing prices or usage produce **unknown**, not zero. Both prompts use identical deterministic sampling/context/output settings from that model's configuration. Live inference uses the existing `requests` dependency; charts use the already installed Matplotlib/NumPy. Core planning, checkpoints and evaluation otherwise use the standard library.

## Configure definitions and prompts

```powershell
python scripts/run_prompt_benchmark.py plan --output outputs/prompt_benchmark/city_reference --limit 100 --config path/to/models.json --definitions path/to/routing_definitions.md
```

Simple provides the two tasks, a valid category list, minimal output/abstention instructions and the enabled case inputs. Detailed adds independent evaluation of original/current routing, category distinctions, overlap rules, uncertainty handling, and careful treatment of photographs and coordinates. Both always receive `ORIGINAL_TYPE`, `CURRENT_TYPE` and `DESCRIPTION`, and exactly the same output schema:

```json
{"misroute": true, "recommended_category": "Illegal Dumping", "reason": "Discarded mattress is visible in the alley."}
```

`misroute` is `true`, `false`, or `null`. `recommended_category` is a valid category or `null`; the two decisions can abstain independently. Extra keys, invalid categories and empty reasons fail validation. The schema is supplied through Ollama's structured output format on every request.

No standalone Markdown routing definitions were found in the inspected workspace. By default, the detailed prompt includes a Markdown rendering of **all existing definitions and definition statuses** in `outputs/semantic_review/service_definitions.json`. `--definitions` embeds your Markdown verbatim instead; `--reference` selects a different JSON category inventory. The resolved prompt is saved as `simple_prompt.txt`/`detailed_prompt.txt`. Configure `prompts.simple` and `prompts.detailed` in the model config with a `version` and optional `template` path relative to that config file. Templates require `{categories}`; detailed also requires `{definitions}`. The frozen prompt, not the subsequently edited source template, controls a resumed run.

## Photo and location inputs

Photo modes pass **image bytes**, not URLs masquerading as images. Public photo URLs remain audit metadata. Cache photos before planning and supply `--photo-manifest` pointing to JSON of this shape:

```json
{"https://example.org/public-photo": "photos/cached-photo.jpg"}
```

Paths are relative to the manifest (absolute paths also work). The planner hashes cached bytes; execution checks them before each request. All numbered `PUBLIC_PHOTO_URL_n` fields are included in numeric order, with duplicate URLs removed. The runner never downloads photos. A case with linked but uncached photos **fails** its photo cells with a repair message; a case with no exported photo links receives explicit unavailable-photo metadata. Known missing links are not proof the original case had no photos. Plan after caching; repairing missing cache associations requires a new plan. Missing files at a previously frozen path can be restored with the original bytes and retried.

Location modes supply the exported latitude/longitude or `null`; text modes omit it. Invalid or incomplete coordinates fail planning. No reverse geocoding is performed. Missing photos/location remain the same across variants and models, avoiding modality-specific resampling. Reports include counts of completed cases with photos and location so absence is visible.

## Explicitly run or resume selected cells

These commands initiate **local model inference**; planning and reporting do not:

```powershell
python scripts/run_prompt_benchmark.py run --output outputs/prompt_benchmark/trial --batch 0 --workers 4

# Select by model/prompt/modality; multiple values can be repeated.
python scripts/run_prompt_benchmark.py run --output outputs/prompt_benchmark/trial --model qwen3-8b-local --variant detailed --modality text

# Retry one failed experiment (get the exact ID from list).
python scripts/run_prompt_benchmark.py run --output outputs/prompt_benchmark/trial --experiment EXPERIMENT_ID --failed-only
```

The bounded async runner calls the blocking local transport in threads. `--retries 1` means up to two attempts per selected pending/failed prediction per invocation. Each attempt commits separately, preserving raw responses, full requests, error messages, latency, tokens and cost, including validation failures. Successful predictions are committed once and always skipped on resume; failed predictions are updated under the same primary key while attempt history grows. An interrupted attempt with no final prediction remains pending. Avoid simultaneous runner processes against the same plan: the checkpoint guarantees resume, not cross-process work leasing.

Use repeated `--case CASE_ID` options to validate a subset of a frozen batch before expanding to the rest; completed subset predictions are reused. `scripts/execute_local_prompt_benchmark.py` is an explicit local-inference entry point for the requested ten-case gate followed by the same 100-case checkpoint, using only simple/detailed × text/text_location. It records wall times and verifies real Ollama response metadata, the output schema, saved hashes, database integrity, unchanged checkpoint/attempt contents on resume, and report artifacts before allowing the larger stage. Stage verification JSON and the ten-case report snapshot are saved alongside the final report.

## Labels and comparative evaluation

```powershell
python scripts/run_prompt_benchmark.py report --output outputs/prompt_benchmark/trial
```

Open `report/index.html` for comparative tables and charts. Exports include:

| Artifact | Contents |
| --- | --- |
| `performance.csv` | By model, modality and prompt: planned/completed/failed/skipped/pending; verified and weak-label precision, recall, F1, false positives/rates; verified category accuracy; original/current agreement; abstention rates; tokens, latency and cost |
| `by_batch.csv` | The same performance metrics for each fixed batch |
| `paired_differences.csv` | Detailed minus simple on the **intersection of completed identical cases**, paired label counts, discordant category outcomes, overhead deltas and paired bootstrap intervals |
| `paired_cases.csv` | Individual paired predictions and resource differences for inspection |
| `false_positives.csv` | Case-level false positives, explicitly identified as weak or verified labels |
| `predictions.csv` | Prediction/checkpoint status, prompt identity and errors |
| `raw_attempts.jsonl` | All attempts with unchanged raw response strings and request content |
| `report.json` | Machine-readable results, thresholds and interpretation notes |
| `performance.png`, `overhead.png`, `paired_f1.png` | Comparative performance, overhead and paired differences by model/modality |

The existing `label` is **weak/proxy misroute supervision**, never a verified correct-category label. `CURRENT_TYPE` and `ORIGINAL_TYPE` are agreement targets only. Supply independently adjudicated labels with `--verified` **at planning time**:

```csv
ID,independently_verified,verification_provenance,verified_misroute,verified_category
case-id,true,Blinded City-informed review 2026-11-19,true,Illegal Dumping
```

At least one target is required per row; blanks mean unavailable. Every row must explicitly assert independent verification and provide provenance. Category labels must be in the valid inventory; the pipeline does not invent a misroute flag from them. Independent review should be blind to proxy labels, model outputs and downstream fields. This file is an assertion by the label provider, not a verification process performed by the software. No evaluation labels, IDs, status, closure notes or referral counts are sent to models.

The new experiment deliberately exposes `CURRENT_TYPE`, as requested. The older `results/model_input_policy.json` excludes that field for its original blinded-review/training purpose and remains unchanged. These new results measure routing **with downstream current-category information available**; weak-label agreement can be inflated by that information and does not establish submission-time or independently verified accuracy. Existing source limitations for description timing and photo/location provenance still apply.

Flag abstention on a verified/weak positive counts as a false negative for recall/F1. Category abstention counts as incorrect where a verified category exists. Coverage and abstention rates are reported separately, and failed/skipped predictions remain distinct from model abstentions. Undefined metrics have null/blank values. Category agreement uses only explicit aliases in the service reference; other downstream workflow names remain unmapped and are counted as such.

Paired bootstrap intervals resample identical labeled cases, with deterministic seeds; they are descriptive 95% intervals, not corrected for testing multiple models/modalities. Aggregate resource totals include every attempt, including failures and retries. Per-completed-case resource means and paired differences include all attempts for those cases. Latency measures the adapter/model request wall time, summed over retries and excluding input preparation, queue waits, reporting and preflight. Unknown usage/cost propagates as unknown.

Tradeoff assessment uses verified F1 as its primary metric. Defaults require a complete pair of experiments, at least 30 independently verified flag labels, a F1 gain of at least .01, a paired interval strictly above zero, and extra per-case overhead within 10,000 total tokens, 10 seconds and $0.01. Configure `--min-verified`, `--min-f1-gain`, `--max-extra-tokens`, `--max-extra-latency`, `--max-extra-cost` and `--bootstrap` for the intended workload. These are explicit decision thresholds, not universal cost-effectiveness claims. All synthetic adapters are marked fixture-only. Review verified category accuracy, false positives, modality availability and completeness alongside the assessment; with no independent labels, the report cannot select an empirically justified winner.
