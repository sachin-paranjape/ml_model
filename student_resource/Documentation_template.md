# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Machine Learning & Data Science Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We developed an end-to-end, high-precision machine learning pipeline for large-scale multi-source business entity resolution in the Amazon ML Challenge 2026. The solution combines country-partitioned TF-IDF sparse cosine similarity blocking, high-throughput RapidFuzz string similarity feature engineering (12 features), and a LightGBM gradient boosted decision tree calibrated specifically for macro $F_{0.5}$ score optimization. Running on ~1.73M test reference entities and ~10M Source 2/3 records, our pipeline completed end-to-end inference in 31.8 minutes, producing 969,047 matched entities and 763,497 singletons while passing all official validation checks (including full ID existence verification) with zero errors.

---

## 2. Methodology

### 2.1 Problem Analysis
Analysis of the 12.5M training records and 11.6M test records across the three independent data sources revealed key structural characteristics:
- **Severe Scale Constraints:** A naive cross-source Cartesian product between 1.73M Source 1 records and ~10M Source 2 + Source 3 records yields $\approx 1.73 \times 10^{13}$ pairs. This necessitated a memory-efficient, chunked sparse blocking mechanism capable of pruning $>99.999\%$ of non-matching pairs without discarding genuine links.
- **Heterogeneous Noise Patterns:** Business names frequently differ by legal abbreviations (`Corp` vs. `Corporation`, `Pvt Ltd` vs. `Private Limited`, `LLC`), punctuation (`&` vs. `and`), word permutations, and phonetic misspellings.
- **Sparse and Noisy Addresses:** Over 170,000 records lack addresses entirely, while populated addresses exhibit landmark-based references (`Near SBI ATM`), missing postal codes, and differing regional token orders.
- **Open-Set Country Partitioning:** While training data comprises `US` and `India`, the test set introduces `France` (~259K S1 entities). Hardcoding country lists would cause catastrophic failure; the pipeline was engineered to handle arbitrary country labels dynamically.
- **Metric Asymmetry ($F_{0.5}$):** The evaluation metric penalizes false merges (false positives) twice as heavily as missed links (false negatives), and singletons that are correctly left unmatched score a full 1.0. A conservative decision boundary is essential.

### 2.2 Solution Strategy
- **Approach Type:** Country-Partitioned TF-IDF Sparse Blocking + RapidFuzz Feature Engineering + LightGBM Classifier with Macro $F_{0.5}$ Threshold Search.
- **Core Innovation:** A chunked sparse matrix cosine similarity blocking engine utilizing sublinear TF scaling and vocabulary frequency trimming (`max_df=0.25`), followed by C++-accelerated string metric extraction and entity-grouped negative downsampling that directly optimizes the macro $F_{0.5}$ validation score.

---

## 3. Candidate Generation (Blocking)

To reduce the $17$ trillion pair search space into a tractable candidate set:
- **Blocking keys used:**
  1. **Primary Partition:** Exact match on `country` (open-set string grouping, allowing separate isolated blocks for `France`, `India`, `US`).
  2. **Secondary Blocking:** Normalized business name TF-IDF word vectors (`sublinear_tf=True`, `min_df=2`, `max_df=0.25`). Pairwise similarities are computed via sparse matrix dot products in chunks of 5,000 Source 1 queries against the combined S2 + S3 sparse feature matrix.
- **Candidate pairs generated:**
  - Total candidate pairs: **59,081,798** across the test set.
  - Active S1 entities with candidates: **1,255,601** (averaging 34.1 candidates per entity, bounded by $top\_k = 50$).
  - Clean singletons identified at blocking: **476,943** entities had zero TF-IDF token overlap and were safely routed directly to singleton status.
- **How true matches were not lost:**
  - Standardized domain-specific legal entity abbreviations (expanding 21 corporate abbreviations like `pvt`, `corp`, `ltd`, `inc`) prior to n-gram tokenization.
  - TF-IDF max document frequency filtering (`max_df=0.25`) removed non-discriminative ubiquitous stopwords without pruning entity-defining keywords.
  - Retaining up to the top 50 sparse cosine candidates per S1 entity ensured high candidate coverage while keeping downstream scoring fast.

---

## 4. Matching Model

### 4.1 Features Used (12 Features Total)
All string similarity computations use `rapidfuzz` C++ backend (~1 $\mu$s per call):
- **Name Features (6):**
  - `name_ratio`: Levenshtein similarity ratio
  - `name_token_sort`: Token-sorted Levenshtein ratio (robust to word order permutations)
  - `name_token_set`: Token-set Levenshtein ratio (handles subset/superset names)
  - `name_partial`: Best partial substring match ratio
  - `name_jaccard`: Word-level token Jaccard set similarity
  - `name_len_ratio`: Normalized character length ratio $\min(len_1, len_2) / \max(len_1, len_2)$
- **Address Features (4):**
  - `addr_ratio`: Full address Levenshtein ratio
  - `addr_token_sort`: Token-sorted address ratio
  - `addr_token_set`: Token-set address ratio
  - `addr_jaccard`: Word-level token Jaccard similarity of address tokens
- **Metadata & Indicator Features (2):**
  - `both_have_addr`: Binary indicator equal to 1.0 if both records provide an address, and 0.0 if either is missing (allowing the tree model to treat missing addresses cleanly).
  - `source_is_s3`: Binary flag indicating whether the candidate is from Source 3 (vs. Source 2).

### 4.2 Model Type & Hyperparameters
- **Model:** LightGBM Binary Classifier (`lightgbm.train`)
  - Objective: `binary`, metric: `binary_logloss`
  - Max leaves: 63, learning rate: 0.05, feature fraction: 0.85
  - Early stopping: 20 rounds (converged at round 363 with validation logloss 0.0395)
  - Class balancing: `scale_pos_weight = 5.25` based on positive-to-negative candidate pair ratio
- **Threshold Selection Method:**
  - Evaluated on a held-out entity validation set with true singletons included.
  - Grid search over candidate thresholds $\in [0.10, 0.90]$ calculating the exact competition macro $F_{0.5}$ metric.
  - Optimal threshold selected: **$0.82$**. The conservative threshold heavily penalizes false merges and ensures maximum precision on the macro evaluation.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score:** **0.6457** (with LightGBM threshold at 0.82).
- **Test Set Predictions:**
  - Total S1 entities: **1,732,544**
  - Matched entities: **969,047** (55.9%)
  - Clean singletons: **763,497** (44.1%)
  - Total candidate pairs scored: **59,081,798**
- **Validation Check:**
  - Passed `utils/validate_submission.py` with 0 errors.
  - Full ID existence verification (`--check-ids`) validated all **9,969,589** Source 2 and Source 3 IDs with 0 missing or invalid references.
- **Common False Positives (Wrong Merges):**
  - Businesses sharing identical common brand names (e.g., chain stores or regional franchises like "Apex Motors") located in different localities when address fields are omitted or truncated.
- **Common False Negatives (Missed Matches):**
  - Candidate entities where business names suffered simultaneously from severe phonetic transliteration noise and heavy character truncation, falling below the TF-IDF cosine blocking threshold.

---

## 6. Conclusion

Our solution demonstrates that combining partitioned TF-IDF sparse cosine blocking with gradient boosted decision trees and precision-calibrated thresholding delivers an effective, scalable entity resolution pipeline. By specifically targeting macro $F_{0.5}$ optimization and preserving clean singleton predictions, the pipeline achieves high matching fidelity across 11.6 million multi-source records within 32 minutes of compute time.

---

## Appendix

### A. Code Artefacts
The complete reproducible pipeline is packaged under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── utils.py          # TSV loading (sep='\t'), abbreviation expansion, lookups
│   ├── blocking.py       # TF-IDF cosine candidate generation per country
│   ├── matcher.py        # 12 RapidFuzz features + LightGBM training + F0.5 grid search
│   └── inference.py      # Batch scoring of candidates and TSV formatting
├── model/
│   ├── model.txt         # Trained LightGBM model weights
│   └── threshold.txt     # Calibrated threshold (0.82)
├── run_pipeline.py       # Master CLI orchestrator
├── requirements.txt      # Pinned dependency environment
└── README.md             # Reproduction instructions
```

**Reproduction Command:**
```bash
python code/business_entity_resolution/run_pipeline.py --base-dir .
```

### B. Additional Results & Country Breakdown

| Country | S1 Entities | S2+S3 Records | TF-IDF Vocab | Blocking Duration |
|:---|:---:|:---:|:---:|:---:|
| **France** | 259,452 | 1,434,993 | 50,366 | 9 seconds |
| **India** | 809,986 | 4,717,565 | 109,763 | 58 seconds |
| **US** | 663,106 | 3,817,031 | 115,733 | 108 seconds |

- Pipeline wall time: 31.8 minutes
- Official Submission Validation: **PASS — no blocking issues found. Safe to submit.**
