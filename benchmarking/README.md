# Get It Done routing capstone

This folder collects the preprocessing method, six-model benchmark pipeline, and completed evaluation report. Copy this whole folder into the capstone repository as one project directory, preserving its internal paths. Run commands from this directory.

| Part | Start here |
| --- | --- |
| Preprocessing original/current-type mismatches | [Preprocessing notebook](notebooks/category_mapping_and_resolution_filtering.ipynb) |
| How the benchmark labels were constructed | [Labeling method and lineage](LABELING.md) |
| Testing the same 100 cases across six models | [Pipeline and rerun commands](PIPELINE.md) |
| Completed report | [Presentation HTML](outputs/prompt_benchmark/final_evaluation/notebook/Final_Evaluation.html) |
| Executed analysis notebook | [Final_Evaluation.ipynb](outputs/prompt_benchmark/final_evaluation/notebook/Final_Evaluation.ipynb) |
| Full metric tables | [Report](outputs/prompt_benchmark/final_evaluation/report/index.html) and [performance CSV](outputs/prompt_benchmark/final_evaluation/report/performance.csv) |

Models: Qwen3 8B, Qwen3-VL 8B, Qwen3-VL 4B, Gemma 3 4B, Llama 3.1 8B, and Ministral 3 8B. Each uses simple/detailed prompts across description, description + location, description + photos, and all three inputs. Unsupported photo configurations for the two text-only models remain explicit skips.

The cohort has 100 cases from a category-matched development dataset of 1,634 cases (817 per proxy class). These are exploratory weak-label results. Original/current-type differences identify candidates for review; they do not establish routing errors. The report explains the label construction, Qwen screening bias, failed configurations, and limits on generalization.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/prepare_capstone_benchmark.py
```

The last command prepares a separate portable run without downloads or inference. See [PIPELINE.md](PIPELINE.md) for executing all six models and scoring the saved outputs.

## Add to the capstone repository

Keep code, notebooks with saved outputs, small lineage tables, frozen configuration, 100-case inputs/labels, normalized photo assets, figures, HTML, and CSV reports together. The included `.gitignore` excludes model weights, raw exports, generated reruns, and SQLite files.

The **674 MB `outputs/prompt_benchmark/final_evaluation/checkpoint.sqlite` is distributed separately and ignored by Git**. It preserves exact requests, raw responses, attempts and prediction origins. A Git checkout can read the saved report/notebook outputs and rerun inference using the packaged inputs/photos. Reexecuting the historical analysis notebook and its raw-request audit requires this checkpoint; distribute it separately as a research artifact and restore it to that path. No model server is needed for historical analysis.

The historical preprocessing notebook requires `Get_it_done_data/cases_2026_11_16.csv`, kept outside Git. The large semantic caches and full historical exports needed to rebuild the proxy dataset remain in the original research workspace; [LABELING.md](LABELING.md) identifies that boundary.

`package_manifest.json` records source paths and hashes. Source research files, labels, historical plans and report values were preserved. The preprocessing copy only adds support for an absent optional referral subset; its saved outputs still describe the original run. Original paths inside frozen provenance describe the historical workspace.

Dataset-screening implementations are excluded from publication, including historical copies of the description/category review and LLM-derived subset builders. The six-model benchmark uses a separate HTTP transport and evaluates the unchanged saved dataset. Historical methodology and limitations remain documented; some historical source-hash entries refer to intentionally undistributed files.
