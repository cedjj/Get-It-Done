# Six-model final evaluation

Canonical package: `outputs/prompt_benchmark/final_evaluation/`.

The final notebook is `notebook/Final_Evaluation.ipynb`; its executed, code-hidden HTML counterpart is `notebook/Final_Evaluation.html`. The HTML is intended for presentation, and the notebook retains executable analysis cells. Both are built entirely from real saved predictions; notebook execution never calls a model or the internet.

## Scope and preservation

The four models from `multimodal_v2` are reused with the exact existing simple/detailed v2 prompts, common schema, and frozen 100-case cohort. Only `llama3.1:8b` and `ministral-3:8b` are newly evaluated. Llama's image configurations are explicitly skipped; Ministral supports all four modalities. Existing failed predictions are retained, not retried or filled in during consolidation.

The final checkpoint is a separate six-model plan namespace. Its `prediction_origins` and `attempt_origins` tables preserve the original experiment/attempt IDs and exact source-row hashes. Only experiment IDs are remapped; raw requests, responses, counters, predictions, and timestamps remain unchanged. `provenance/source_files.json` fingerprints the entire previous four-model result directory. The previous files remain as frozen historical records so old links and checkpoints continue to work.

Raw requests and model outputs have one authoritative representation in the final package: `checkpoint.sqlite`, table `attempts`. A second large raw-request JSONL export is deliberately not generated. Normalized input photos in `assets/photos` are hard linked to the verified cache where supported, so they do not require duplicate image storage. Original downloaded bytes and earlier reports remain in the preserved source directories.

## Package layout

| Path | Contents |
|---|---|
| `notebook/` | Executed notebook and standalone presentation HTML |
| `report/` | Standard metric tables, false positives, paired comparisons, and HTML report |
| `analysis/` | Notebook-derived paired modality ablations, photo-available subset, selected configuration |
| `figures/` | Presentation charts in PNG and SVG |
| `checkpoint.sqlite` | All six models' predictions/statuses, exact raw requests/responses, counters, origin mappings |
| `plan.json`, `config/`, `*_prompt.txt` | Frozen cases, schema, model options, and exact prompts |
| `data/` | Blinded cases, separate evaluation-only labels, input/source hashes |
| `assets/` | Case-owned normalized photos and ownership provenance |
| `runtime/` | Local server, downloads, new-model execution and resume evidence |
| `provenance/` | Prior-run timing, diagnostics, fingerprints, source import records |
| `reproducibility/` | Analysis helper, source snapshots/hashes, environment and dependencies |
| `validation/` | Independent raw-output/metric audit and notebook execution evidence |

## Reproduction

From the workspace root, prepare/import existing results once:

```powershell
python scripts/consolidate_final_benchmark.py
python scripts/prepare_final_assets.py
```

Start the workspace Ollama server with the settings in [MULTIMODAL.md](provenance/existing_four/MULTIMODAL.md). Downloads are explicit and limited to the requested additions:

```powershell
python scripts/download_benchmark_models.py --model llama3.1:8b --model ministral-3:8b --output outputs/prompt_benchmark/final_evaluation/runtime/model_downloads.json
python scripts/execute_final_benchmark.py
```

That driver selects only the two new models, performs ten-case schema/checkpoint gates, expands to the same 100 cases, unloads between model phases, and checks the imported predictions/attempts are unchanged. It uses one GPU request at a time and no paid APIs. Completed predictions and unsupported skips are reused on resume. Failed cells retain all attempts; the driver does not change options to force them to succeed.

To retry an individual failed experiment intentionally, use the existing CLI with an exact final-plan experiment ID from `list`:

```powershell
python scripts/run_prompt_benchmark.py list --output outputs/prompt_benchmark/final_evaluation
python scripts/run_prompt_benchmark.py run --output outputs/prompt_benchmark/final_evaluation --model MODEL_ID --experiment EXPERIMENT_ID --failed-only --workers 1
```

Generate all consolidated reports, snapshot the code, execute the notebook from top to bottom, and export HTML:

```powershell
python scripts/finalize_final_evaluation.py
```

This command performs offline analysis only. It preserves partial notebook outputs on an execution failure, checks consecutive execution counts and absence of error outputs, and records artifact hashes. The standalone analysis helper is included with the notebook; opening and running the saved notebook does not require an active Ollama server. Existing absolute image paths remain frozen for checkpoint identity; packaged image bytes allow offline audit without the original cache location. Re-running inference on a different filesystem layout requires a new plan rather than editing old hashes.

## Interpretation

The primary comparison holds the simple prompt and description inputs constant across all six models. The notebook also presents every supported strategy and each model's best fully completed configuration. Its provisional recommendation uses an explicit post-hoc heuristic: configurations within 0.01 weak-label F1 of the highest fully completed score are practical ties; then prefer fewer input modalities, simpler instructions, and fewer reported tokens. This is exploratory selection, not a preregistered study or verified deployment recommendation.

All performance targets are weak proxies. Earlier Qwen3 8B screening influenced label/cohort construction and may favor Qwen-family agreement. Original/current category agreement is not independently verified category accuracy. Structured current-routing, label, downstream-outcome, ID, and evaluation fields are excluded from requests. The original routing category is intentionally visible because judging that routing is the first task. Exported description/location are not verified immutable submission snapshots; descriptions can contain later annotations. Photo ownership is checked, while upload timing and initial/closure role remain unknown. Numerical coordinates are not supplemented by GIS lookups.

Paired analyses use intersections of the same completed cases. Positive-case abstentions count as false negatives, while all abstentions fail flag accuracy. Undefined metrics remain unavailable, not zero. Failed configurations remain visible and are excluded from final selection. Bootstrap intervals use 1,000 paired case resamples and are exploratory, without multiple-comparison correction. No independently verified accuracy is available.

Local API charges are zero. Energy, hardware ownership, and review labor are not measured. Token counters use model-specific accounting. Observed latency depends on caching, cold loads, quantization, backend, and desktop load; it is not a controlled production-throughput measurement. All-attempt resource totals retain failed/retried request consumption.
