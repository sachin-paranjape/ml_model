# Business Entity Resolution — ML Challenge 2026

## Overview

Three-stage pipeline for matching business entities across three
independent, noisy data sources:

1. **Blocking** — TF-IDF cosine similarity on normalised business names
   (per-country) to generate a high-recall candidate set.
2. **Matching** — LightGBM binary classifier trained on 12 pairwise
   string-similarity features (rapidfuzz + Jaccard) with a threshold
   optimised for **F₀.₅**.
3. **Inference** — score all test candidates, apply the threshold, and
   write the final `matching_results.tsv`.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. Tested on Python 3.13.

## Reproducing results

Run from the `student_resource/` directory:

```bash
# Full pipeline (train + test + validate):
python code/business_entity_resolution/run_pipeline.py --base-dir .

# If you already have a trained model and just want to re-run test inference:
python code/business_entity_resolution/run_pipeline.py --base-dir . --skip-train

# Adjustable parameters:
#   --sample-size N   Number of S1 training entities to sample (default 25000)
#   --top-k K         Max candidates per S1 entity (default 50)
```

Output files are written to `output/`:

| File | Description |
|---|---|
| `matching_results.tsv` | Final entity matches (scored on leaderboard) |
| `candidate_pairs.tsv` | Blocking candidate set before classification |

## Project structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── utils.py        # Data loading, text normalisation, lookups
│   ├── blocking.py     # TF-IDF cosine blocking (per-country)
│   ├── matcher.py      # Feature engineering, LightGBM training
│   └── inference.py    # Test scoring + output formatting
├── model/              # Trained model + threshold (auto-created)
├── run_pipeline.py     # Master runner
├── requirements.txt
└── README.md
```

## Validation

The pipeline automatically runs `utils/validate_submission.py` at the end.
You can also run it manually:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

## Packaging Final Submission

Create the competition-compliant zip file with code, outputs, and documentation:

```bash
python package_submission.py --team-name EntityResolvers
```

This generates `EntityResolvers_submission.zip` matching the required structure:
```
EntityResolvers_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```


