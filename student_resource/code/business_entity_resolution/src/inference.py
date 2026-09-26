"""
inference.py — Score test-set candidate pairs and produce final outputs.

For every S1 test entity we:
  1. retrieve its blocking candidates,
  2. compute the 12-dimensional feature vector for each pair,
  3. score with the trained LightGBM model,
  4. apply the F₀.₅-optimised threshold,
  5. write ``matching_results.tsv`` and ``candidate_pairs.tsv``.
"""

import os
import time
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.matcher import compute_features_batch

logger = logging.getLogger(__name__)


def run_inference(
    model: lgb.Booster,
    threshold: float,
    candidates: Dict[str, List[str]],
    s1_lookup: Dict[str, Tuple[str, str]],
    s23_lookup: Dict[str, Tuple[str, str]],
    s1_eids: np.ndarray,
    batch_size: int = 200_000,
) -> Dict[str, List[str]]:
    """
    Score all candidate pairs and return final matches.

    Returns ``{s1_entity_id: [matched_s23_ids …]}`` (empty list = singleton).
    """

    results: Dict[str, List[str]] = {}

    # Flatten pairs for batching
    all_s1: list = []
    all_c:  list = []
    pair_s1_eid: list = []

    for s1_eid in s1_eids:
        cands = candidates.get(s1_eid, [])
        if not cands or s1_eid not in s1_lookup:
            results[s1_eid] = []
            continue
        for c in cands:
            if c in s23_lookup:
                all_s1.append(s1_eid)
                all_c.append(c)

    total = len(all_s1)
    logger.info("Inference — %d candidate pairs to score (threshold %.4f)", total, threshold)

    # Process in batches
    all_preds = np.empty(total, dtype=np.float64)
    t0 = time.time()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_s1 = all_s1[start:end]
        batch_c  = all_c[start:end]

        s1n = [s1_lookup[e][0] for e in batch_s1]
        s1a = [s1_lookup[e][1] for e in batch_s1]
        cn  = [s23_lookup[e][0] for e in batch_c]
        ca  = [s23_lookup[e][1] for e in batch_c]

        X = compute_features_batch(s1n, cn, s1a, ca, batch_c)
        all_preds[start:end] = model.predict(X)

        if (start // batch_size + 1) % 5 == 0 or end == total:
            elapsed = time.time() - t0
            pct = end / total * 100
            logger.info("  scored %d/%d (%.0f%%)  %.0fs elapsed", end, total, pct, elapsed)

    # Assemble per-entity results
    for i in range(total):
        s1_eid = all_s1[i]
        c_eid  = all_c[i]
        if all_preds[i] >= threshold:
            results.setdefault(s1_eid, []).append(c_eid)

    # Entities with candidates but no match above threshold → singleton
    for s1_eid in s1_eids:
        results.setdefault(s1_eid, [])

    matched = sum(1 for v in results.values() if v)
    logger.info(
        "Inference done — %d entities, %d matched, %d singletons",
        len(results), matched, len(results) - matched,
    )
    return results


# ── Output helpers ─────────────────────────────────────────────────────────

def save_matching_results(
    results: Dict[str, List[str]],
    s1_eids: np.ndarray,
    output_path: str,
):
    """Write ``matching_results.tsv`` (tab-separated, one row per S1 entity)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1_eids:
            matches = results.get(eid, [])
            # deduplicate while preserving order
            seen = set()
            unique = []
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    unique.append(m)
            fh.write(f"{eid}\t{','.join(unique)}\n")
    logger.info("Saved matching_results → %s", output_path)
