# Six-model evaluation pipeline

The preserved historical run is under `outputs/prompt_benchmark/final_evaluation/`. Open its [presentation HTML](outputs/prompt_benchmark/final_evaluation/notebook/Final_Evaluation.html) to read the completed report, including failures, comparisons and limitations.

The active entry point is [scripts/run_prompt_benchmark.py](scripts/run_prompt_benchmark.py), backed by `scripts/prompt_benchmark/`: immutable plans and input schemas, resumable SQLite execution, model adapters, photo validation and evaluation. `local_transport.py` provides the local Ollama HTTP transport without dataset-labeling logic. Nothing downloads or invokes a model merely by importing these files.

## Prepare a portable rerun

From this project directory:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/prepare_capstone_benchmark.py
```

Preparation verifies the frozen input hashes, labels and photo ownership/bytes; it freezes exactly the same 100 cases and order, six models, model options, schema and v2 prompts. It uses local packaged photos and creates a new plan in `outputs/prompt_benchmark/reproduced/`. It does not contact a model or the internet. A new filesystem location produces a new plan identity; the historical plan and checkpoint stay frozen.

The matrix contains 6 models × 2 prompts × 4 input configurations = 48 planned configurations. Qwen3 8B and Llama 3.1 8B skip photo configurations, leaving 40 supported configurations. Every case gets a saved completion, failure or unsupported status. Completed cells are reused on resume. Use another empty output path with `--output` for a separate run.

## Execute the six models

Install and start local Ollama, then explicitly install the tags recorded in [the frozen model config](outputs/prompt_benchmark/final_evaluation/config/models.json). These commands download model weights:

```powershell
ollama pull qwen3:8b
ollama pull qwen3-vl:8b-instruct-q4_K_M
ollama pull gemma3:4b
ollama pull qwen3-vl:4b-instruct-q4_K_M
ollama pull llama3.1:8b
ollama pull ministral-3:8b
```

Run one GPU request at a time across the six-model plan:

```powershell
python scripts/run_prompt_benchmark.py run --output outputs/prompt_benchmark/reproduced --workers 1
```

Repeat that command to resume. Historical options use temperature 0, seed 42, a 16,384-token context and a 384-token output limit. Installed tags may evolve; the runner records model/runtime fingerprints and rejects changes within a checkpoint. Exact historical digests and environment records are preserved under the saved run's `runtime/` and `reproducibility/` folders. Running newer weights/hardware can change predictions and timing.

## Score and generate the rerun report

```powershell
python scripts/run_prompt_benchmark.py report --output outputs/prompt_benchmark/reproduced --evaluation outputs/prompt_benchmark/final_evaluation/data/scoring_labels.csv
```

Labels are attached only during scoring. The report writes performance tables, paired prompt/model/modality comparisons, false positives and HTML to the new run's `report/`. Positive-case abstentions count as false negatives, all abstentions fail flag accuracy, and undefined metrics remain unavailable. Scores measure agreement with proxy targets. This command generates the standard benchmark report; the preserved presentation notebook contains the additional historical analysis and charts.

## Reexecute historical analysis offline

The historical `final_evaluation/checkpoint.sqlite` is distributed separately and excluded from Git because it is about 674 MB. On another computer, restore that file from the separately shared research artifact to `outputs/prompt_benchmark/final_evaluation/checkpoint.sqlite`. With dependencies installed, open `notebook/Final_Evaluation.ipynb` in that same evaluation folder and run all cells. It locates the bundle relative to the notebook, recomputes metrics, audits raw requests/responses and validates packaged images without model calls. The saved notebook/HTML can be read without this database.

The historical `reproducibility/source/` files show the original staged experiment: four models were imported and two models added. Those scripts expect prior research directories. For a clean six-model rerun from a repo checkout, use the portable preparation command above.
