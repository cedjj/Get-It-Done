# Preprocessing and proxy-label construction

Start with [category_mapping_and_resolution_filtering.ipynb](notebooks/category_mapping_and_resolution_filtering.ipynb). This is the simple deterministic preprocessing notebook for original/current-type mismatches. Its saved outputs document the historical run. Rerunning requires the historical cases export; it performs no model inference.

## 1. Find category differences requiring review

The notebook preserves the raw fields, normalizes Unicode, capitalization, whitespace and dash variants, and applies explicit category aliases. It removes listed directional routine refinements, such as Sidewalk becoming Sidewalk Repair Issue. Conditional and unlisted category combinations remain for review. Missing categories and an original category of Other are counted separately.

The core queue requires `STATUS = Closed` and exactly one of `Problem resolved`, `Issue Resolved`, or `Compliance Achieved`. Differences remaining after these rules become review candidates. Closure status and category changes do not prove that the initial routing was wrong. The mapping is a reviewed proposal, rather than an official complete historical taxonomy.

Outputs include `suspicious_cases.csv`, stepwise filter counts and category mappings. The separate three-plus-referrals cohort is a companion population, not extra cases added to the full export. In this package copy, that companion is optional when its CSV is absent. Place the full cases export in [Get_it_done_data/](Get_it_done_data/README.md), then run the notebook from this folder or `notebooks/`.

## 2. Construct positive proxies

The later Qwen semantic review assessed descriptions and original categories against service guidance. The semantic run was stopped and incomplete: its retained progress records 57,895 of 84,077 candidate cases. Within the recorded results, 2,777 mismatch predictions whose suggested category exactly matched current type were retained as positive proxies. These are model-assisted labels, not independently adjudicated errors. The preprocessing notebook alone did not produce these labels.

## 3. Construct negative proxies and restrict to common category support

Negatives require nonempty normalized original/current equality, an observed explicit zero internal-referral count, Closed status, an approved successful resolution reason, conservative semantic acceptance, and duplicate/clarity screens. Missing referral values do not count as zero.

The full historical rescan found 21 of 32 positive category strata without strictly eligible negatives. The corrected builder caps each category at its own quota and never transfers unavailable quota to another category. It retains all 2,777 positives with 883 negatives in a 3,660-case audit collection. This collection has substantial category imbalance.

A separate common-support subset takes equal class counts within each supported category: **817 positives + 817 negatives = 1,634 cases** across 11 categories. The final benchmark freezes 100 cases from this subset. This addresses category-count imbalance within supported categories, while narrowing coverage and preserving the limitations of proxy labels.

See the original [detailed labeling report](results/labeling_summary.md), [category counts](results/common_support_category_distribution.csv), and [eligibility/QA evidence](reference/labeling_evidence/). The detailed report's workspace-relative reproduction paths refer to the original research workspace.

## Reproduction and source lineage

[reference/labeling_source/](reference/labeling_source/) preserves the deterministic preprocessing builder and photo-link enrichment code. The implementations that screen descriptions against categories or select the LLM-derived labeling subset are excluded from this publication package, including embedded historical source copies. The existing dataset, report and methodology disclosures are retained. Complete dataset reconstruction requires the excluded labeling code, full exports, cached semantic reviews, QA records, earlier positive collection, and prior-run archive in the original research workspace. The labeling report records the historical reconstruction commands; those commands are not a runnable labeling workflow in this handoff.

The executable six-model pipeline uses the frozen 100-case inputs and separate scoring labels shipped in `outputs/prompt_benchmark/final_evaluation/data/`; it does not need the historical labeling caches.

## Inference boundary and interpretation

Model requests use description and original category, with optional coordinates and case-owned photos. Current category, proxy labels, referrals, status, closure reason, IDs, semantic decisions and audit fields are excluded from model requests. IDs are used locally to join predictions and scoring labels, with input hashes checked.

Qwen screening influenced cohort/label selection, so agreement may favor related models. Exported descriptions and location are not verified immutable submission snapshots; photo ownership is checked, but upload timing is unknown. Independent blinded review is needed to establish real-world routing accuracy. These results describe a proxy development benchmark, rather than an independently validated held-out test set.
