# Routing dataset correction: strict category matching is infeasible

## Result and dataset roles

The full **4,337,284-case** historical export was rescanned for all **32** positive ORIGINAL_TYPE strata. **21 strata have no strictly eligible negatives**, covering **1956 / 2,777 positives (70.44%)**. Preserving all 2,777 positives and strict original/current equality makes an adequately matched full dataset **impossible**, at any nonzero negative sample size. Fewer negatives alone cannot supply absent category support.

- `correctly_routed.csv`: **883** verified negatives, capped at each original proportional quota. **No missing quota is transferred.** This is a partial negative collection, not a matched distribution against all positives.
- `labeled_routing_dataset.csv`: **3660** rows: 883 negatives plus **all 2,777 unchanged positive cases**. This is an incomplete-coverage audit/development collection and **still has substantial category imbalance**. Do not use it as a category-matched benchmark.
- `category_matched_common_support.csv`: a separate **1634-row** proxy development subset, **817 negatives and 817 positives**, with **identical category counts and percentages for every included category**. No categories are mapped or substituted. The main dataset retains the full positive set; only this explicitly separate subset downsamples positives.
- `common_support_excluded_positives.csv`: **1960** positives excluded from the paired subset, with reasons. They remain in the main dataset. This restriction changes the target population; the paired subset cannot establish performance on the 21 uncovered categories.
- `category_distribution.csv`: side-by-side prior/corrected counts, percentages, shortages and imbalance flags. `common_support_category_distribution.csv` describes the paired subset.

Seed **42** is used for proportional rounding and seeded SHA-256 random priorities. All cases remain **proxy labels**, not independently established ground truth.

## What caused the original imbalance

The previous implementation calculated the intended quotas correctly, but then used `capped_allocation` to transfer unavailable category slots repeatedly to categories with remaining candidates until it reached 3,000 negatives. This overfilled Illegal Dumping (1,122), Tree Maintenance (739), Encampment (608), and other categories. That quantity-first policy was inappropriate for a category-controlled benchmark. It has been removed from the active builder; the previous builder and outputs are archived under `archive/labeled_routing/previous_imbalanced_v1/`.

The zero categories were **not caused by capitalization, whitespace, the semantic model, or the negative keyword screens**. For all positive categories, literal and normalized equality counts are identical in the new audit. Twenty strata have no original/current equality at all; Parking has 23 equal-category cases but none survive zero-referral/closed/successful-resolution requirements. Hence 21 strict-eligibility shortages.

Original reporting-menu names and historical CURRENT_TYPE service/workflow names often use different vocabularies. Examples from the full rescan:

| STRATUM | CURRENT_TYPE | cases |
| --- | --- | --- |
| dumpster overflow | Environmental Services Code Compliance | 2984 |
| dumpster overflow | Trash/Recycling Collection | 626 |
| dumpster overflow | Illegal Dumping | 490 |
| graffiti | Graffiti - Public | 90739 |
| graffiti | Graffiti - Code Enforcement | 50581 |
| graffiti | Graffiti - Private | 26231 |
| illegal discharge | Stormwater Code Enforcement | 5213 |
| illegal discharge | Illegal Dumping | 1023 |
| illegal discharge | Parks Issue | 219 |
| parking | Parking - 72-Hours | 115056 |
| parking | Parking Violation | 34744 |
| parking | Oversized Vehicle | 8606 |
| sidewalk | Sidewalk Repair Issue | 31238 |
| sidewalk | Vegetation Encroachment | 2216 |
| sidewalk | Illegal Dumping | 1612 |

For example, Sidewalk commonly becomes Sidewalk Repair Issue; Graffiti becomes Graffiti - Public/Private/Code Enforcement. Illegal Discharge commonly becomes Stormwater Code Enforcement, and Dumpster Overflow commonly becomes Environmental Services Code Compliance. These names might describe appropriate downstream handling, but **they do not satisfy the required equality**. Treating them as equivalent would change the label policy and is not done here. The inventory is diagnostic, not a mapping used for eligibility. No unrelated categories replace missing cases.

## Full historical eligibility investigation

The source is `Get_it_done_data/cases_2026_11_16.csv`, not the 84,077-case suspicious queue. Internal referrals are joined to the November 2 referral export by normalized case number. Only an observed, valid, explicit numeric zero qualifies; missing joins/counts are not zeros. Both category fields must be nonempty and equal after Unicode NFKC normalization, case folding and collapsed/trimmed whitespace. Closed status and exactly Problem resolved, Issue Resolved or Compliance Achieved are retained. These are the same conservative successful outcomes used previously; broader reasons were not added to manufacture capacity.

The following counts precede semantic/duplicate filtering, so zeros cannot be explained by overly restrictive semantic filtering:

| ORIGINAL_TYPE | all_original_rows | literal_equal | normalized_equal | equal_explicit_zero_referrals | equal_zero_closed | equal_zero_closed_successful |
| --- | --- | --- | --- | --- | --- | --- |
| ADA Compliance | 3502 | 1524 | 1524 | 1384 | 746 | 130 |
| Bicycle Related Issue | 2487 | 0 | 0 | 0 | 0 | 0 |
| Bicycle-Related Issue | 720 | 567 | 567 | 504 | 416 | 3 |
| Container Left Out | 12563 | 0 | 0 | 0 | 0 | 0 |
| Containers Left Out Beyond Collection Hours | 3805 | 0 | 0 | 0 | 0 | 0 |
| Curb | 15340 | 0 | 0 | 0 | 0 | 0 |
| Damaged Guardrail | 2477 | 0 | 0 | 0 | 0 | 0 |
| Dead Animal | 18095 | 18055 | 18055 | 17079 | 16885 | 13706 |
| Dumpster Encroaching on Public Right of Way | 494 | 0 | 0 | 0 | 0 | 0 |
| Dumpster Encroachment | 2440 | 0 | 0 | 0 | 0 | 0 |
| Dumpster Overflow | 4909 | 0 | 0 | 0 | 0 | 0 |
| Encampment | 226583 | 221280 | 221280 | 215298 | 198352 | 19280 |
| Faded Striping | 6454 | 0 | 0 | 0 | 0 | 0 |
| Graffiti | 183003 | 0 | 0 | 0 | 0 | 0 |
| Illegal Discharge | 7220 | 0 | 0 | 0 | 0 | 0 |
| Illegal Dumping | 389001 | 362033 | 362033 | 97174 | 91392 | 74066 |
| Missed Collection | 41676 | 41323 | 41323 | 41321 | 40960 | 5173 |
| Missed Trash | 107355 | 0 | 0 | 0 | 0 | 0 |
| Outreach | 40275 | 0 | 0 | 0 | 0 | 0 |
| Over Irrigation | 8747 | 0 | 0 | 0 | 0 | 0 |
| Parking | 165422 | 23 | 23 | 2 | 2 | 0 |
| Parking Issue | 234376 | 0 | 0 | 0 | 0 | 0 |
| Pothole | 98489 | 94356 | 94356 | 93674 | 91167 | 63365 |
| Sidewalk | 46902 | 0 | 0 | 0 | 0 | 0 |
| Storm Drain | 8050 | 0 | 0 | 0 | 0 | 0 |
| Street Flooded | 9730 | 4746 | 4746 | 4401 | 2707 | 788 |
| Street Light | 49061 | 0 | 0 | 0 | 0 | 0 |
| Street Sweeping | 9640 | 8371 | 8371 | 8121 | 7311 | 5347 |
| Traffic Sign | 30684 | 0 | 0 | 0 | 0 | 0 |
| Traffic Signal | 42219 | 0 | 0 | 0 | 0 | 0 |
| Tree Maintenance | 36957 | 29335 | 29335 | 28419 | 24114 | 17155 |
| Vegetation Encroachment | 12351 | 10683 | 10683 | 10369 | 7954 | 7133 |

`full_category_eligibility_audit.csv` also records successful zero-referral closures with equality deliberately ignored, showing that many resolved cases exist but fail the required equality. For example, the audit finds 12,736 such Sidewalk rows, 789 Illegal Discharge rows, 2,741 Dumpster Overflow rows and 77,088 Graffiti rows; **none** qualify once exact category equality is applied. `original_current_category_inventory.csv` and `category_normalization_inventory.csv` retain the detailed naming evidence. `full_category_audit_manifest.json` records the full scan, source hashes and the fact that no semantic/keyword filters or category mappings were applied to that audit.

The rare Bicycle-Related Issue stratum has exactly three historically eligible cases. All three descriptions were inspected: two explicitly say they are test reports (3397475 and 4540708) and were excluded; case 4097107 describes a branch obstructing a bicycle lane and is retained. Thus one usable negative remains versus a quota of five. The unhyphenated Bicycle Related Issue stratum has no strict matches and is kept separate, as required by capitalization/whitespace-only normalization. This exhausts the only nonzero historical stratum unable to fill its requested quota; the other supported categories have enough verified negatives.

## Corrected sampling and semantic quality

The 3,000-case target quotas are computed directly from the unchanged 2,777 positive rows using largest remainders of `3000 * category_positive_count / 2777`, with alphabetical ties. For each category, select `min(requested_quota, verified_negative_capacity)` with seeded priorities and **stop**. The result is 883 negatives and **2117 unfilled slots**, without quota transfers. Sampling and case ranking are deterministic with seed 42.

Verified cases reuse the earlier local Ollama `qwen3:8b` checks: both conservative text-only semantic screens must accept the unchanged description/original category and City reference, with exact nonempty evidence. Those inputs and reference hashes were checked unchanged against the full rescan and previous manifest. Reusing the cached semantic judgments avoids new inference on the same reports; **no new paid API or LLM calls** were needed. Semantic inputs are only DESCRIPTION, ORIGINAL_TYPE and the versioned service reference, never CURRENT_TYPE, referrals or closure data. Full historical fields are used solely to construct/audit labels.

The previous conservative clarity screens, parent-link exclusions, exact/numeric-template deduplication and near-text-duplicate exclusions remain in force for this verified pool; no semantic criterion was weakened. The pool is not a claim that every historical case has been semantically reviewed. This is sufficient here because every nonzero-quota category has ample verified capacity except the three-case bicycle stratum, which was exhaustively inspected.

Service guidance is the unchanged `outputs/semantic_review/service_definitions.json`, with curated paraphrases and provenance in the previous report and `semantic_config.json`. The reference includes the official [City reporting guide](https://www.sandiego.gov/sites/default/files/get-it-done-guidelines.pdf) and [City service FAQ](https://getitdone.sandiego.gov/TSWCommunityFAQ). Historical boundaries are incomplete, and Qwen screening is not independent City validation.

## Side-by-side category distribution and remaining imbalance

Percentages use each class's actual denominator: 883 negatives versus 2,777 positives. A descriptive imbalance flag means missing negative support OR an absolute class-percentage difference of at least **1 percentage point**; it is not a statistical significance test. All flagged rows remain unsuitable for claiming full category balance. Requested quotas retain the original 3,000 denominator to make every unfilled slot visible.

| ORIGINAL_TYPE | positive_count | target_negative_count_at_3000 | previous_negative_count | negative_count | unfilled_negative_quota | positive_percent | negative_percent | difference_percentage_points | significantly_imbalanced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADA Compliance | 20 | 22 | 30 | 22 | 0 | 0.72 | 2.492 | 1.771 | True |
| Bicycle Related Issue | 13 | 14 | 0 | 0 | 14 | 0.468 | 0.0 | -0.468 | True |
| Bicycle-Related Issue | 5 | 5 | 1 | 1 | 4 | 0.18 | 0.113 | -0.067 | False |
| Container Left Out | 43 | 46 | 0 | 0 | 46 | 1.548 | 0.0 | -1.548 | True |
| Containers Left Out Beyond Collection Hours | 7 | 8 | 0 | 0 | 8 | 0.252 | 0.0 | -0.252 | True |
| Curb | 111 | 120 | 0 | 0 | 120 | 3.997 | 0.0 | -3.997 | True |
| Damaged Guardrail | 8 | 9 | 0 | 0 | 9 | 0.288 | 0.0 | -0.288 | True |
| Dead Animal | 2 | 2 | 7 | 2 | 0 | 0.072 | 0.227 | 0.154 | False |
| Dumpster Encroaching on Public Right of Way | 22 | 24 | 0 | 0 | 24 | 0.792 | 0.0 | -0.792 | True |
| Dumpster Encroachment | 144 | 156 | 0 | 0 | 156 | 5.185 | 0.0 | -5.185 | True |
| Dumpster Overflow | 212 | 229 | 0 | 0 | 229 | 7.634 | 0.0 | -7.634 | True |
| Encampment | 163 | 176 | 608 | 176 | 0 | 5.87 | 19.932 | 14.062 | True |
| Faded Striping | 11 | 12 | 0 | 0 | 12 | 0.396 | 0.0 | -0.396 | True |
| Graffiti | 111 | 120 | 0 | 0 | 120 | 3.997 | 0.0 | -3.997 | True |
| Illegal Discharge | 412 | 445 | 0 | 0 | 445 | 14.836 | 0.0 | -14.836 | True |
| Illegal Dumping | 301 | 325 | 1122 | 325 | 0 | 10.839 | 36.806 | 25.967 | True |
| Missed Collection | 13 | 14 | 49 | 14 | 0 | 0.468 | 1.586 | 1.117 | True |
| Missed Trash | 2 | 2 | 0 | 0 | 2 | 0.072 | 0.0 | -0.072 | True |
| Outreach | 138 | 149 | 0 | 0 | 149 | 4.969 | 0.0 | -4.969 | True |
| Over Irrigation | 27 | 29 | 0 | 0 | 29 | 0.972 | 0.0 | -0.972 | True |
| Parking | 13 | 14 | 0 | 0 | 14 | 0.468 | 0.0 | -0.468 | True |
| Parking Issue | 6 | 7 | 0 | 0 | 7 | 0.216 | 0.0 | -0.216 | True |
| Pothole | 20 | 22 | 75 | 22 | 0 | 0.72 | 2.492 | 1.771 | True |
| Sidewalk | 537 | 580 | 0 | 0 | 580 | 19.337 | 0.0 | -19.337 | True |
| Storm Drain | 29 | 31 | 0 | 0 | 31 | 1.044 | 0.0 | -1.044 | True |
| Street Flooded | 16 | 17 | 59 | 17 | 0 | 0.576 | 1.925 | 1.349 | True |
| Street Light | 30 | 32 | 0 | 0 | 32 | 1.08 | 0.0 | -1.08 | True |
| Street Sweeping | 69 | 75 | 258 | 75 | 0 | 2.485 | 8.494 | 6.009 | True |
| Traffic Sign | 68 | 73 | 0 | 0 | 73 | 2.449 | 0.0 | -2.449 | True |
| Traffic Signal | 12 | 13 | 0 | 0 | 13 | 0.432 | 0.0 | -0.432 | True |
| Tree Maintenance | 198 | 214 | 739 | 214 | 0 | 7.13 | 24.236 | 17.106 | True |
| Vegetation Encroachment | 14 | 15 | 52 | 15 | 0 | 0.504 | 1.699 | 1.195 | True |

Main-dataset category total variation distance: **0.705025**; previous value: **0.705824**. The theoretical lower bound for any negatives under these strict rules while retaining every positive is **0.704357**, because that much positive mass has no eligible negative counterpart. Removing forced substitution therefore does **not** magically balance the full-positive dataset; it makes the infeasibility explicit and avoids fabricating coverage.

For the separate paired subset, each included category gets `min(positive_count, selected_negative_count)` cases **from each class**. Every one of its 11 included categories has identical class counts; total variation distance is **0.000000** and no included category is imbalanced. The excluded 21 strata have zero cases in both classes of this subset, not invented negatives. Four extra positives in the scarce hyphenated bicycle stratum are also excluded from that subset. This sacrifices coverage and some positive cases only in the companion dataset; it does not alter or delete any positives from the requested main file.

## QA and data leakage

- No duplicate case IDs, duplicate normalized case numbers, class-overlapping IDs or case-level conflicting labels in either output dataset. All original positive values/fields are preserved in the main file.
- All 883 negatives satisfy strict equality, observed zero referrals, Closed status, successful reason and both cached semantic accepts. Negative normalized descriptions and numeric templates are unique; no exact normalized negative/positive description overlap exists.
- Character-within-word 3–5-gram hashing cosine checks confirm maximum negative-pair similarity **0.872107**, and maximum cross-class similarity **0.897527**, both below 0.90. As before, these heuristics do not detect every duplicate physical incident.
- A new deterministic random-priority sample of **50** negatives was inspected using full DESCRIPTION and ORIGINAL_TYPE, reusing earlier assistant observations only when the exact ID/text/category matched. `v2_qa_50_inputs.csv` and `v2_qa_50_review.csv` retain the case-by-case findings. All final 50 were judged consistent. This is assistant QA, not independent human adjudication or an accuracy estimate.
- This correction's spot review rejected six candidates (ambiguous dump/camp wording, mixed landscaping/irrigation complaints, unstated service problems, uncertain discarded-item context, and a missed-collection report with unrelated issues). They were replaced within their original strata, never with another category. `v2_qa_exclusions.csv` and `v2_qa_review_history.csv` preserve these findings and replacements.
- The positives retain their existing repeated descriptions: the prior audit found 508 rows in 70 repeated-description groups (438 excess descriptions). Distinct positive IDs are preserved as requested. Their exact-agreement labels remain provisional.
- Full source SHA-256 hashes were reverified unchanged. Original datasets and notebooks were not modified. Previous outputs and code were backed up before regeneration.

**Prediction-time allowlist:** DESCRIPTION and ORIGINAL_TYPE; add only original photos, verified submission-time location and official definitions with established provenance. Do not use CURRENT_TYPE, referral counts/flags, STATUS, REASON, action/resolution fields, Qwen outputs, semantic judgments, labels, sampling/audit fields, IDs or any other downstream fields as model features. The wide CSVs are audit tables, not feature matrices; `model_input_policy.json` and `field_roles.csv` retain that boundary.

The exported description/location are not verified immutable submission snapshots. Some descriptions contain dispatch/work-order annotations; verify text history and remove downstream additions before training. Keep original photos separate from after-work photos. Labels are asymmetric proxies selected using Qwen agreement versus historical stability and semantic clarity; class-specific clarity/length and repeated-text shortcuts remain even in the category-matched subset.

Evaluate a final model on a **separate independently reviewed sample**, with City-informed reviewers initially blind to Qwen, labels and downstream outcomes. Group development splits by duplicate/near-duplicate text and by incident, location and time where established. A holdout of either proxy dataset alone cannot establish real-world routing accuracy. The common-support dataset addresses the category-count shortcut only, not label validity or deployment coverage.

## Reproduction

Python dependencies are in `requirements.txt`; semantic caches and the archived prior outputs are required to reproduce this correction without rerunning inference.

```powershell
python scripts/audit_category_eligibility.py
python scripts/build_labeled_routing.py prepare
# Inspect v2_qa_50_inputs.csv and record exact case-level findings in v2_qa_50_review.csv.
# Add failures to v2_qa_exclusions.csv and rerun prepare if needed.
python scripts/build_labeled_routing.py record_qa
python scripts/build_labeled_routing.py finalize
```

The active builder has no redistribution branch. It fails if sources changed, cached semantic inputs differ, required outcomes are missing, positives changed, IDs duplicate, quotas are exceeded, or final QA does not match the selected cases. Reproduction reuses the saved exact QA judgments; changed samples require new inspection. `run_manifest.json` declares the requested full match **infeasible** and records the paired subset as a separate artifact.
