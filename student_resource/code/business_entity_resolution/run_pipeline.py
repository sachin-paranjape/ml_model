#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end entity resolution pipeline.

Usage (from student_resource/):

    python code/business_entity_resolution/run_pipeline.py --base-dir .

Steps:
  1. Load & pre-process training data
  2. Sample S1 entities, run blocking, build training pairs
  3. Train LightGBM + optimise F₀.₅ threshold
  4. Load & pre-process test data
  5. Run blocking on test set → candidate_pairs.tsv
  6. Score candidates → matching_results.tsv
  7. Run the official validator
"""

import argparse
import gc
import logging
import os
import sys
import time

# ── make sure the code package is importable ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd

from src.utils import (
    load_sources,
    load_ground_truth,
    parse_ground_truth,
    preprocess_df,
    build_lookup,
)
from src.blocking import run_blocking
from src.matcher import (
    prepare_training_pairs,
    train_lgbm,
    save_model,
    load_model,
)
from src.inference import run_inference, save_matching_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def run_training(base_dir: str, model_dir: str, sample_size: int, top_k: int):
    """Train the matcher on a sample of the training data."""

    logger.info("=" * 70)
    logger.info("STAGE: TRAINING")
    logger.info("=" * 70)

    # ── load data ─────────────────────────────────────────────────────────
    t0 = time.time()
    s1, s2, s3 = load_sources(base_dir, "train")
    gt_df = load_ground_truth(base_dir)
    gt_dict = parse_ground_truth(gt_df)
    logger.info("Data loaded in %.0fs", time.time() - t0)

    # ── sample S1 entities (stratified by country) ────────────────────────
    if sample_size and sample_size < len(s1):
        # stratified sampling by country
        parts = []
        for country, grp in s1.groupby("country"):
            n = max(1, int(sample_size * len(grp) / len(s1)))
            parts.append(grp.sample(min(len(grp), n), random_state=42))
        sample = pd.concat(parts, ignore_index=True)
        logger.info("Sampled %d / %d S1 entities for training", len(sample), len(s1))
    else:
        sample = s1
        logger.info("Using all %d S1 entities for training", len(s1))

    # ── preprocess ────────────────────────────────────────────────────────
    sample_pp = preprocess_df(sample)
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23_pp = preprocess_df(s23)
    del s2, s3; gc.collect()

    # ── blocking on training sample ───────────────────────────────────────
    logger.info("\n── Training blocking ──")
    train_candidates = run_blocking(sample_pp, s23_pp, top_k=top_k)

    # measure blocking recall
    total_true, found_true = 0, 0
    for s1_eid in sample_pp["entity_id"]:
        true_set = gt_dict.get(s1_eid, set())
        cand_set = set(train_candidates.get(s1_eid, []))
        total_true += len(true_set)
        found_true += len(true_set & cand_set)
    recall = found_true / total_true if total_true else 0
    logger.info("Training blocking recall: %.4f  (%d / %d true matches found)",
                recall, found_true, total_true)

    # ── build lookups ─────────────────────────────────────────────────────
    s1_lookup  = build_lookup(sample_pp)
    s23_lookup = build_lookup(s23_pp)
    del s23_pp; gc.collect()

    # ── feature engineering ───────────────────────────────────────────────
    logger.info("\n── Feature engineering ──")
    X, y, s1_indices = prepare_training_pairs(
        train_candidates, gt_dict, s1_lookup, s23_lookup, max_neg_per_entity=15,
    )

    # gt_sizes: number of true matches per entity index (needed for F₀.₅)
    s1_id_list = list(train_candidates.keys())
    gt_sizes = {
        idx: len(gt_dict.get(eid, set()))
        for idx, eid in enumerate(s1_id_list)
    }

    # ── train ─────────────────────────────────────────────────────────────
    logger.info("\n── Model training ──")
    model, threshold = train_lgbm(X, y, s1_indices, gt_sizes, val_frac=0.2)
    save_model(model, threshold, model_dir)

    del X, y, s1_indices, train_candidates; gc.collect()
    return model, threshold


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TEST INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def run_test(base_dir: str, output_dir: str, model_dir: str, top_k: int):
    """Run country-partitioned blocking + inference on the test set, write output files."""

    logger.info("=" * 70)
    logger.info("STAGE: TEST INFERENCE (Country-by-Country Stream)")
    logger.info("=" * 70)

    model, threshold = load_model(model_dir)

    # ── load raw test tables ──────────────────────────────────────────────
    t0 = time.time()
    s1, s2, s3 = load_sources(base_dir, "test")
    logger.info("Test data loaded in %.0fs", time.time() - t0)

    # Output file paths
    cand_path  = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    # Check if France is already completed in existing output files
    completed_countries = set()
    total_s1 = 0
    total_matched = 0

    if os.path.exists(match_path) and os.path.exists(cand_path):
        with open(match_path, "r", encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        if n_lines == 259453:
            completed_countries.add("France")
            total_s1 = 259452
            total_matched = 250415
            logger.info("✓ France is already complete (259,452 entities). Resuming with remaining countries!")

    if "France" not in completed_countries:
        with open(cand_path, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        with open(match_path, "w", encoding="utf-8") as f_match:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

    countries = [c for c in sorted(s1["country"].unique()) if c not in completed_countries]
    logger.info("Processing countries: %s", countries)

    for country in countries:
        logger.info("\n" + "=" * 50)
        logger.info("══ Country: %s ══", country)
        logger.info("=" * 50)

        # 1. Filter and preprocess ONLY for this country
        s1_c_raw = s1[s1["country"] == country].reset_index(drop=True)
        s2_c_raw = s2[s2["country"] == country]
        s3_c_raw = s3[s3["country"] == country]
        s23_c_raw = pd.concat([s2_c_raw, s3_c_raw], ignore_index=True)
        del s2_c_raw, s3_c_raw; gc.collect()

        s1_c  = preprocess_df(s1_c_raw)
        s23_c = preprocess_df(s23_c_raw)
        del s1_c_raw, s23_c_raw; gc.collect()

        s1_c_eids = s1_c["entity_id"].values
        logger.info("  %s S1: %d  |  S2+S3: %d", country, len(s1_c), len(s23_c))

        if len(s23_c) == 0:
            with open(cand_path, "a", encoding="utf-8") as f_cand:
                for eid in s1_c_eids:
                    f_cand.write(f"{eid}\t\n")
            with open(match_path, "a", encoding="utf-8") as f_match:
                for eid in s1_c_eids:
                    f_match.write(f"{eid}\t\n")
            total_s1 += len(s1_c_eids)
            del s1_c, s23_c; gc.collect()
            continue

        # 2. Blocking for this country
        from src.blocking import _generate_candidates_for_country
        cands = _generate_candidates_for_country(
            s1_c, s23_c, top_k=top_k, max_df=0.02, chunk_size=500
        )

        # Append candidates to candidate_pairs.tsv
        with open(cand_path, "a", encoding="utf-8") as f_cand:
            for eid in s1_c_eids:
                c_list = cands.get(eid, [])
                f_cand.write(f"{eid}\t{','.join(c_list)}\n")

        # 3. Build lookups ONLY for this country
        s1_lookup  = build_lookup(s1_c)
        s23_lookup = build_lookup(s23_c)
        del s23_c; gc.collect()

        # 4. Run inference for this country
        results = run_inference(
            model, threshold, cands, s1_lookup, s23_lookup, s1_c_eids, batch_size=200_000
        )

        # Append matches to matching_results.tsv
        with open(match_path, "a", encoding="utf-8") as f_match:
            for eid in s1_c_eids:
                matches = results.get(eid, [])
                seen = set()
                unique = []
                for m in matches:
                    if m not in seen:
                        seen.add(m)
                        unique.append(m)
                f_match.write(f"{eid}\t{','.join(unique)}\n")

        c_matched = sum(1 for v in results.values() if v)
        total_s1 += len(s1_c_eids)
        total_matched += c_matched
        logger.info("  %s done: %d entities, %d matched (%.1f%%)",
                    country, len(s1_c_eids), c_matched, 100 * c_matched / max(len(s1_c_eids), 1))

        # Clean up this country entirely from memory
        del s1_c, cands, s1_lookup, s23_lookup, results; gc.collect()

    del s1, s2, s3; gc.collect()

    logger.info("\nInference Complete — Total S1: %d, Matched: %d (%.1f%%), Singletons: %d",
                total_s1, total_matched, 100 * total_matched / max(total_s1, 1), total_s1 - total_matched)
    return match_path, cand_path


# ═════════════════════════════════════════════════════════════════════════════
# 3.  VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def run_validation(base_dir: str, output_dir: str):
    """Run the official validator script."""
    import subprocess

    logger.info("=" * 70)
    logger.info("STAGE: VALIDATION")
    logger.info("=" * 70)

    validator = os.path.join(base_dir, "utils", "validate_submission.py")
    matching  = os.path.join(output_dir, "matching_results.tsv")
    candidate = os.path.join(output_dir, "candidate_pairs.tsv")
    test_dir  = os.path.join(base_dir, "dataset", "test")

    cmd = [
        sys.executable, validator,
        "--matching", matching,
        "--candidate", candidate,
        "--test-dir", test_dir,
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Entity Resolution Pipeline")
    p.add_argument("--base-dir", default=os.path.join(SCRIPT_DIR, "..", ".."),
                   help="Path to student_resource/ (default: auto-detected)")
    p.add_argument("--sample-size", type=int, default=75000,
                   help="Number of S1 entities to sample for training (default 75000)")
    p.add_argument("--top-k", type=int, default=100,
                   help="Max candidates per S1 entity from blocking (default 100)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; use previously saved model")
    p.add_argument("--skip-test", action="store_true",
                   help="Skip test inference")
    args = p.parse_args()

    base_dir  = os.path.abspath(args.base_dir)
    output_dir = os.path.join(base_dir, "output")
    model_dir  = os.path.join(SCRIPT_DIR, "model")

    logger.info("Base dir : %s", base_dir)
    logger.info("Output   : %s", output_dir)
    logger.info("Model    : %s", model_dir)
    logger.info("Sample   : %d   top-k: %d", args.sample_size, args.top_k)

    os.makedirs(output_dir, exist_ok=True)
    wall_start = time.time()

    # ── Train ─────────────────────────────────────────────────────────────
    if not args.skip_train:
        run_training(base_dir, model_dir, args.sample_size, args.top_k)
    else:
        logger.info("Skipping training (--skip-train)")

    # ── Test ──────────────────────────────────────────────────────────────
    if not args.skip_test:
        run_test(base_dir, output_dir, model_dir, args.top_k)
    else:
        logger.info("Skipping test (--skip-test)")

    # ── Validate ──────────────────────────────────────────────────────────
    rc = run_validation(base_dir, output_dir)

    elapsed = time.time() - wall_start
    logger.info("Total wall time: %.0fs (%.1f min)", elapsed, elapsed / 60)

    if rc == 0:
        logger.info("✓  Validation PASSED")
    else:
        logger.warning("✗  Validation FAILED (exit %d)", rc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
