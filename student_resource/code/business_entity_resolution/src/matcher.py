"""
matcher.py — Pairwise feature engineering, LightGBM training, and
             F₀.₅-optimised threshold selection.

Features (20 total):
  Name   : ratio, token_sort_ratio, token_set_ratio, partial_ratio,
           WRatio, Jaro-Winkler, word-Jaccard, char-bigram-Jaccard,
           char-trigram-Jaccard, length-ratio
  Address: ratio, token_sort_ratio, token_set_ratio, partial_ratio,
           word-Jaccard, Jaro-Winkler
  Meta   : both_have_address flag, source_is_S3 flag,
           common_numeric_tokens, exact_word_overlap

All string similarities use *rapidfuzz* (C++ back-end, ~1 µs/call).
"""

import os
import logging
import time
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import lightgbm as lgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    # — Name (10) —
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_wratio",
    "name_jaro_winkler",
    "name_jaccard",
    "name_char2_jaccard",
    "name_char3_jaccard",
    "name_len_ratio",
    # — Address (6) —
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_partial",
    "addr_jaccard",
    "addr_jaro_winkler",
    # — Meta (4) —
    "both_have_addr",
    "source_is_s3",
    "common_numeric_tokens",
    "exact_word_overlap",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _word_jaccard(s1: str, s2: str) -> float:
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    u = len(t1 | t2)
    return len(t1 & t2) / u if u else 0.0


def _char_ngram_jaccard(s1: str, s2: str, n: int) -> float:
    if len(s1) < n or len(s2) < n:
        return 0.0
    ng1 = {s1[i:i + n] for i in range(len(s1) - n + 1)}
    ng2 = {s2[i:i + n] for i in range(len(s2) - n + 1)}
    u = len(ng1 | ng2)
    return len(ng1 & ng2) / u if u else 0.0


import re
_NUM_RE = re.compile(r"\d+")


def _common_numeric_tokens(s1: str, s2: str) -> float:
    """Fraction of shared numeric tokens between two text strings."""
    nums1 = set(_NUM_RE.findall(s1)) if s1 else set()
    nums2 = set(_NUM_RE.findall(s2)) if s2 else set()
    u = len(nums1 | nums2)
    return len(nums1 & nums2) / u if u else 0.0


def _exact_word_overlap(s1: str, s2: str) -> float:
    """Raw count of exact shared words (not normalised)."""
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    return float(len(t1 & t2))


# ── Feature computation ───────────────────────────────────────────────────

def compute_features_batch(
    s1_names: list,
    s23_names: list,
    s1_addrs: list,
    s23_addrs: list,
    s23_eids: list,
) -> np.ndarray:
    """Compute 20 pairwise features for a list of (S1, S23) pairs."""
    n = len(s1_names)
    X = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    for i in range(n):
        n1, n2 = s1_names[i], s23_names[i]
        a1, a2 = s1_addrs[i], s23_addrs[i]

        # — Name similarities (10 features) —
        X[i, 0] = fuzz.ratio(n1, n2) / 100.0
        X[i, 1] = fuzz.token_sort_ratio(n1, n2) / 100.0
        X[i, 2] = fuzz.token_set_ratio(n1, n2) / 100.0
        X[i, 3] = fuzz.partial_ratio(n1, n2) / 100.0
        X[i, 4] = fuzz.WRatio(n1, n2) / 100.0
        X[i, 5] = JaroWinkler.similarity(n1, n2) if (n1 and n2) else 0.0
        X[i, 6] = _word_jaccard(n1, n2)
        X[i, 7] = _char_ngram_jaccard(n1, n2, 2)
        X[i, 8] = _char_ngram_jaccard(n1, n2, 3)
        mx = max(len(n1), len(n2))
        X[i, 9] = min(len(n1), len(n2)) / mx if mx else 0.0

        # — Address similarities (6 features) —
        if a1 and a2:
            X[i, 10] = fuzz.ratio(a1, a2) / 100.0
            X[i, 11] = fuzz.token_sort_ratio(a1, a2) / 100.0
            X[i, 12] = fuzz.token_set_ratio(a1, a2) / 100.0
            X[i, 13] = fuzz.partial_ratio(a1, a2) / 100.0
            X[i, 14] = _word_jaccard(a1, a2)
            X[i, 15] = JaroWinkler.similarity(a1, a2)

        # — Meta features (4) —
        X[i, 16] = 1.0 if (a1 and a2) else 0.0
        X[i, 17] = 1.0 if s23_eids[i].startswith("S3-") else 0.0

        # combined text for numeric & word overlap
        full1 = f"{n1} {a1}".strip()
        full2 = f"{n2} {a2}".strip()
        X[i, 18] = _common_numeric_tokens(full1, full2)
        X[i, 19] = _exact_word_overlap(n1, n2)

    return X


# ── Training-data construction ────────────────────────────────────────────

def prepare_training_pairs(
    candidates: Dict[str, List[str]],
    gt_dict: Dict[str, Set[str]],
    s1_lookup: Dict[str, Tuple[str, str]],
    s23_lookup: Dict[str, Tuple[str, str]],
    max_neg_per_entity: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix + labels from blocking candidates and ground truth.

    Returns (X, y, s1_indices) where *s1_indices* maps each row to its
    S1 entity (needed later for per-entity threshold search).
    """

    buf_s1n: list = []
    buf_s23n: list = []
    buf_s1a: list = []
    buf_s23a: list = []
    buf_eids: list = []
    buf_labels: list = []
    buf_s1idx: list = []

    s1_id_list = list(candidates.keys())
    s1_id_to_int = {eid: idx for idx, eid in enumerate(s1_id_list)}

    for s1_eid, cand_list in candidates.items():
        if s1_eid not in s1_lookup or not cand_list:
            continue
        s1_name, s1_addr = s1_lookup[s1_eid]
        true_matches = gt_dict.get(s1_eid, set())
        s1_int = s1_id_to_int[s1_eid]

        pos_eids = [c for c in cand_list if c in true_matches and c in s23_lookup]
        neg_eids = [c for c in cand_list if c not in true_matches and c in s23_lookup]

        # down-sample negatives
        if len(neg_eids) > max_neg_per_entity:
            rng = np.random.RandomState(abs(hash(s1_eid)) % (2**31))
            neg_eids = list(rng.choice(neg_eids, size=max_neg_per_entity, replace=False))

        for c_eid in pos_eids:
            cn, ca = s23_lookup[c_eid]
            buf_s1n.append(s1_name);  buf_s23n.append(cn)
            buf_s1a.append(s1_addr);  buf_s23a.append(ca)
            buf_eids.append(c_eid);   buf_labels.append(1)
            buf_s1idx.append(s1_int)

        for c_eid in neg_eids:
            cn, ca = s23_lookup[c_eid]
            buf_s1n.append(s1_name);  buf_s23n.append(cn)
            buf_s1a.append(s1_addr);  buf_s23a.append(ca)
            buf_eids.append(c_eid);   buf_labels.append(0)
            buf_s1idx.append(s1_int)

    logger.info(
        "Feature extraction for %d pairs (%d pos / %d neg) …",
        len(buf_labels), sum(buf_labels), len(buf_labels) - sum(buf_labels),
    )
    t0 = time.time()
    X = compute_features_batch(buf_s1n, buf_s23n, buf_s1a, buf_s23a, buf_eids)
    logger.info("  done in %.1fs", time.time() - t0)

    return X, np.asarray(buf_labels, np.int32), np.asarray(buf_s1idx, np.int32)


# ── F₀.₅ helpers ──────────────────────────────────────────────────────────

def _f05(p, r):
    return (1.25 * p * r) / (0.25 * p + r) if (p + r) else 0.0


def _macro_f05(
    scores: np.ndarray,
    labels: np.ndarray,
    s1_indices: np.ndarray,
    gt_sizes: Dict[int, int],
    threshold: float,
) -> float:
    """Per-entity macro-averaged F₀.₅ for a given threshold."""
    preds = scores >= threshold
    entity_tp: Dict[int, int] = {}
    entity_fp: Dict[int, int] = {}

    for i in range(len(preds)):
        idx = s1_indices[i]
        if preds[i]:
            if labels[i] == 1:
                entity_tp[idx] = entity_tp.get(idx, 0) + 1
            else:
                entity_fp[idx] = entity_fp.get(idx, 0) + 1

    unique_entities = set(s1_indices)
    f05_sum = 0.0
    for idx in unique_entities:
        tp = entity_tp.get(idx, 0)
        fp = entity_fp.get(idx, 0)
        total_true = gt_sizes.get(idx, 0)

        predicted = tp + fp
        if total_true == 0 and predicted == 0:
            f05_sum += 1.0
        elif predicted == 0 or total_true == 0:
            pass                       # contributes 0
        else:
            p = tp / predicted
            r = tp / total_true
            f05_sum += _f05(p, r)

    return f05_sum / len(unique_entities) if unique_entities else 0.0


def optimize_threshold(
    model: lgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    s1_indices_val: np.ndarray,
    gt_sizes: Dict[int, int],
) -> Tuple[float, float]:
    """Grid-search over thresholds to maximise macro F₀.₅ on validation set."""
    scores = model.predict(X_val)

    best_t, best_f = 0.5, 0.0
    # Fine-grained search
    for t in np.arange(0.05, 0.96, 0.01):
        f = _macro_f05(scores, y_val, s1_indices_val, gt_sizes, round(t, 2))
        if f > best_f:
            best_f, best_t = f, round(t, 2)
    logger.info("Threshold search → best_t=%.2f  F₀.₅=%.4f", best_t, best_f)
    return float(best_t), float(best_f)


# ── Model training ─────────────────────────────────────────────────────────

def train_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    s1_indices: np.ndarray,
    gt_sizes: Dict[int, int],
    val_frac: float = 0.2,
) -> Tuple[lgb.Booster, float]:
    """
    Train a LightGBM classifier, then find the F₀.₅-optimal threshold.
    Returns (model, best_threshold).
    """

    # entity-level train/val split to avoid leakage
    unique_s1 = np.unique(s1_indices)
    train_ents, val_ents = train_test_split(
        unique_s1, test_size=val_frac, random_state=42
    )
    val_ent_set = set(val_ents)

    train_mask = np.array([s not in val_ent_set for s in s1_indices])
    val_mask   = ~train_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask],   y[val_mask]
    s1_va      = s1_indices[val_mask]

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale = neg / max(pos, 1)

    logger.info(
        "LightGBM — train %d (pos %.1f%%)  val %d  scale_pos_weight %.2f",
        len(X_tr), 100 * pos / len(y_tr), len(X_va), scale,
    )

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    dval   = lgb.Dataset(X_va, label=y_va, feature_name=FEATURE_NAMES, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "scale_pos_weight": scale,
        "verbose": -1,
        "n_jobs": -1,
    }

    cbs = [lgb.log_evaluation(50), lgb.early_stopping(30)]
    model = lgb.train(
        params, dtrain, num_boost_round=1000, valid_sets=[dval], callbacks=cbs,
    )
    logger.info("Best iteration: %d", model.best_iteration)

    best_t, best_f = optimize_threshold(model, X_va, y_va, s1_va, gt_sizes)
    return model, best_t


# ── Persistence ────────────────────────────────────────────────────────────

def save_model(model: lgb.Booster, threshold: float, model_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(os.path.join(model_dir, "model.txt"))
    with open(os.path.join(model_dir, "threshold.txt"), "w") as f:
        f.write(f"{threshold:.6f}\n")
    logger.info("Model → %s  (threshold %.4f)", model_dir, threshold)


def load_model(model_dir: str) -> Tuple[lgb.Booster, float]:
    model = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
    with open(os.path.join(model_dir, "threshold.txt")) as f:
        threshold = float(f.read().strip())
    logger.info("Loaded model from %s  (threshold %.4f)", model_dir, threshold)
    return model, threshold
