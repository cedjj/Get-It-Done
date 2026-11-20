# Blinded multimodal benchmark

This extends the existing async runner, SQLite checkpoint and evaluator. All real adapters now require a blinded plan and allowlisted case fields. Legacy current-visible fixture plans remain usable only with mock adapters. Previous results under `outputs/prompt_benchmark/qwen_blinded_100` remain intact.

## Executed configuration

The frozen batch is anchored to the prior blinded plan: exactly the same 100 case IDs, order, descriptions, original types and locations. No model/modality-specific resampling is allowed. The new prompt versions and image cache produce separate experiment IDs and a separate checkpoint under `outputs/prompt_benchmark/multimodal_v2`.

| Model | Local tag | Image support |
| --- | --- | --- |
| Qwen3 8B baseline | `qwen3:8b` | No; photo slots are explicitly skipped |
| Qwen3-VL 8B Instruct | `qwen3-vl:8b-instruct-q4_K_M` | Yes |
| Gemma 3 4B | `gemma3:4b` | Yes |
| Qwen3-VL 4B Instruct | `qwen3-vl:4b-instruct-q4_K_M` | Yes |

All models use temperature 0, seed 42, context 16,384, and up to 384 output tokens. Simple/detailed prompts share the exact output schema. The server uses one GPU request slot, q8 KV cache, flash attention and one loaded model. Requests and model groups execute sequentially on the GPU to avoid VRAM conflicts. Photo downloads use bounded async concurrency; optional API requests use provider/key-specific RPM and concurrency controls with shared Retry-After backoff.

Official local model references: [Qwen3-VL 8B](https://ollama.com/library/qwen3-vl:8b-instruct-q4_K_M), [Gemma 3 4B](https://ollama.com/library/gemma3:4b), [Qwen3-VL 4B](https://ollama.com/library/qwen3-vl:4b-instruct-q4_K_M). Local tags/digests and runtime details are pinned by preflight in SQLite; updating a model/runtime requires a new checkpoint plan.

## Photos

`cache_benchmark_photos.py` reads `PUBLIC_ATTACHMENT_URLS_JSON` for the anchored cases. It cross-checks the attachment ID and filename arrays against the original public attachment export, requiring agreement on both `ParentId` and `Parent.Name` case number. Conflicting ownership or array associations stop preparation. In this batch all 134 attachments are valid images, distributed across 85 cases; the other 15 have no exported photo links.

Downloads follow only HTTPS redirects on the official City/Salesforce hosts, have bounded sizes/timeouts/retries, and are verified with Pillow. Original bytes are preserved. Model inputs are normalized uniformly to EXIF-oriented RGB JPEGs with longest dimension at most 1024 pixels, quality 90, and embedded metadata removed. Each original and normalized file has a SHA-256. The runner checks normalized bytes against the immutable plan before inference. Cached image bytes, not public URLs or attachment IDs, are sent to models.

Cache: `outputs/prompt_benchmark/photo_cache_100/`. `download_audit.json` records each case/photo association, source/final URL, original and normalized hash, dimensions and retrieval time. `manifest.json` is the runner-compatible URL-to-cache mapping. Missing/corrupt linked images fail photo predictions rather than silently becoming text-only requests. Cases without links carry explicit unavailable-photo metadata; they remain in every configuration.

Ownership linkage is verified. The export has no upload timestamp or before/after role, so image availability at initial submission is **unverified**. The report makes no claim that these are exclusively initial-report photographs. Closure fields, referral counts, current routing, labels and evaluation metadata are never sent to models.

## Detailed-prompt abstention investigation

The old detailed prompt emphasized positive evidence for a misroute and uncertainty but never explicitly defined the `false` decision. Its raw responses often explained that the original category fit while emitting `null`. The new `detailed-blinded-v2` prompt defines true/false/null separately and gives brief reasonable-route, incompatible-route and insufficient-evidence examples. It preserves legitimate overlaps and allows the flag and category decisions to abstain independently. The simple v2 prompt also states the three output meanings, with no category definitions or detailed examples.

No postprocessing changes null to false and no forced labels are applied. `diagnose_abstention.py` runs separate old/new prompt probes on clear-fit, clear-mismatch and genuinely vague fixture complaints. These are calibration fixtures, not independently verified dataset labels, and their results are not included in the 100-case scores. `abstention_investigation.json` preserves their real raw responses and the diagnosis. Compare the new baseline model's false/null counts against the preserved old run to inspect the effect on the same actual cases.

## Commands

These setup commands perform authorized network downloads; the inference runner itself never pulls models:

```powershell
python scripts/download_benchmark_models.py
python scripts/cache_benchmark_photos.py
```

Run the model-specific ten-case smoke gates, verify saved schema-valid predictions and unchanged resume, then expand each passing model to the complete anchored batch:

```powershell
python scripts/execute_multimodal_benchmark.py
```

Completed predictions are reused. Logs/progress and `model_execution.json` record model/stage times and gates. Models that fail their smoke gate are retained with errors and do not silently proceed. The driver does not start Ollama; run the workspace-installed server with the workspace model directory, `OLLAMA_NO_CLOUD=1`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1`, and `OLLAMA_KV_CACHE_TYPE=q8_0`.

To start that server again, run this from the workspace root in a separate PowerShell terminal, then use another terminal for the runner. Requests set their context size to 16,384 explicitly.

```powershell
$env:OLLAMA_MODELS = Join-Path (Get-Location) '.local_runtime/models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NOPRUNE = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
& ./.local_runtime/ollama/ollama.exe serve
```

Individual failed cells can be retried without overwriting successful predictions:

```powershell
python scripts/run_prompt_benchmark.py list --output outputs/prompt_benchmark/multimodal_v2
python scripts/run_prompt_benchmark.py run --output outputs/prompt_benchmark/multimodal_v2 --model MODEL_ID --experiment EXPERIMENT_ID --failed-only
```

After inference, report generation alone reads evaluation labels from the original CSV, joined by ID with per-case input-hash verification:

```powershell
python scripts/run_prompt_benchmark.py report --output outputs/prompt_benchmark/multimodal_v2 --evaluation results/category_matched_common_support_with_photos.csv
python scripts/audit_multimodal_benchmark.py
```

`report/index.html` and `performance.csv` compare precision, recall, F1, flag accuracy, original/current category agreement, abstentions, latency, tokens and API cost. `paired_differences.csv` includes simple/detailed differences and paired bootstrap intervals. `modality_differences.csv` and `model_comparisons.csv` use intersections of identical completed cases. `photo_availability.csv` separates cases with and without linked photos. `predictions.csv` and `raw_attempts.jsonl` preserve checkpoint statuses and raw audits.

Existing labels are **weak proxies**, including previous model-informed screening. Current-category agreement is not verified category accuracy. No independent category labels are available for this run, so verified accuracy is unavailable. F1 does not penalize abstentions on negative cases; inspect flag accuracy, coverage, negative abstentions and false positives too. Local API charges are zero, while hardware/energy costs are not measured. Latency includes request/model processing, cold loads and cache effects, not queue waits; serial scheduling prevents concurrent benchmark requests from adding queue latency.

When independent adjudications become available, supply `report --verified path/to/verified.csv`. Required columns are `ID,independently_verified,verification_provenance,verified_misroute,verified_category`; every row must explicitly assert independent verification and identify its provenance, with at least one target present. This file is read only by scoring and never becomes an inference input. The evaluator records its source hash and reports verified metrics separately. Token counters use each model/provider's own accounting and are not a shared tokenizer unit across models.

## Optional cloud APIs (not executed without authorization)

`models.apis.example.json` demonstrates Gemini Flash, Claude Sonnet and OpenAI vision adapters. Choose an authorized available model ID and set its API key in the indicated environment variable. Set token rates in `pricing` for cost estimates; missing prices remain unknown. Cloud costs are flat configured-rate token estimates, not invoices or cache-discount-aware billing.

Cloud adapters support image bytes and the same output schema via OpenAI `response_format.json_schema`, Gemini `responseJsonSchema`, and Anthropic `output_config.format`. Official references: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat), [OpenAI vision](https://developers.openai.com/api/docs/guides/images-vision), [Gemini structured outputs](https://ai.google.dev/gemini-api/docs/structured-output), [Gemini images](https://ai.google.dev/gemini-api/docs/image-understanding), [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Anthropic vision](https://platform.claude.com/docs/en/build-with-claude/vision).

An API key is **not** authorization. The runner refuses selected paid models unless the caller explicitly passes `--allow-paid` after user approval. The current task has no paid authorization; none are invoked. Tests exercise payloads, schema handling, token accounting, 429 backoff and the authorization block using mocked HTTP only. Keys/authorization headers are never stored in request audits. Provider rate limits vary; configure RPM/concurrency for the actual account and honor 429 backoff.

The Gemini adapter uses the documented [generateContent REST endpoint](https://ai.google.dev/api/generate-content) and [GenerationConfig fields](https://docs.cloud.google.com/nodejs/docs/reference/generativelanguage/latest/generativelanguage/protos.google.ai.generativelanguage.v1beta.igenerationconfig); it does not use the newer Interactions transport. Choose a model supporting that endpoint. Provider-echoed credentials in HTTP errors are redacted before auditing; successful model outputs are preserved unchanged.
