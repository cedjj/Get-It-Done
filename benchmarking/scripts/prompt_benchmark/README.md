# Routing benchmark modules

Start with the project [pipeline guide](../../PIPELINE.md) to prepare and run the six-model capstone benchmark.

| Module | Role |
| --- | --- |
| `core.py` | Immutable plans, allowlisted cases, common schema, request construction and checkpoints |
| `runner.py` | Model adapters, bounded execution, raw attempts and resume behavior |
| `photos.py` | Photo ownership, download and normalization helpers |
| `cloud.py` | Optional API adapters, separate from the local six-model configuration |
| `evaluate.py` | Proxy metrics, paired comparisons, CSV and HTML reports |
| `templates/` | Versioned prompt templates |

The original [benchmark guides](../../reference/benchmark_guides/) are preserved as historical documentation; their commands can depend on files outside this handoff. Use the project pipeline guide for a fresh checkout.
