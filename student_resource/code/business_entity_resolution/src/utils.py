"""
utils.py — Data loading, text normalisation, and feature helpers.

Handles TSV ingestion (explicit sep='\\t'), missing-value cleanup, and
country-agnostic text normalisation (works for US, India, France, or any
future country label).
"""

import os
import re
import logging
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Abbreviation look-ups ─────────────────────────────────────────────────

NAME_ABBREVS: Dict[str, str] = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "llc": "limited liability company", "pvt": "private",
    "co": "company", "intl": "international", "natl": "national",
    "mfg": "manufacturing", "svcs": "services", "svc": "service",
    "grp": "group", "assoc": "associates", "mgmt": "management",
    "tech": "technology", "sys": "systems", "engg": "engineering",
    "eng": "engineering", "dist": "distribution", "dept": "department",
    "pharma": "pharmaceuticals", "infra": "infrastructure",
    "telecom": "telecommunications",
}

ADDR_ABBREVS: Dict[str, str] = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "pkwy": "parkway", "hwy": "highway", "cir": "circle", "sq": "square",
    "apt": "apartment", "ste": "suite", "fl": "floor", "bldg": "building",
    "no": "number", "dist": "district", "nr": "near", "opp": "opposite",
}


# ── Normalisation ──────────────────────────────────────────────────────────

def _expand(text: str, abbrevs: Dict[str, str]) -> str:
    """Replace known abbreviations (with or without trailing dot)."""
    tokens = text.split()
    return " ".join(abbrevs.get(t.rstrip("."), t) for t in tokens)


def normalize_name(name) -> str:
    """Lowercase, strip punctuation, expand business-name abbreviations."""
    if not isinstance(name, str) or not name.strip():
        return ""
    s = name.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)           # drop all punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return _expand(s, NAME_ABBREVS)


def normalize_address(addr) -> str:
    """Lowercase, strip punctuation, expand address abbreviations."""
    if not isinstance(addr, str) or not addr.strip():
        return ""
    s = addr.lower().strip()
    s = re.sub(r"[^\w\s,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _expand(s, ADDR_ABBREVS)


# ── Data I/O ───────────────────────────────────────────────────────────────

def load_sources(
    base_dir: str, split: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read source1/2/3 TSVs for *split* ('train' or 'test')."""
    prefix = os.path.join(base_dir, "dataset", split)
    frames: list = []
    for i in range(1, 4):
        p = os.path.join(prefix, f"{split}_source{i}.tsv")
        logger.info("Loading %s …", p)
        df = pd.read_csv(p, sep="\t", dtype=str)
        for col in ("business_name", "business_address"):
            df[col] = df[col].fillna("")
        df["country"] = df["country"].fillna("").str.strip()
        frames.append(df)
    return frames[0], frames[1], frames[2]


def load_ground_truth(base_dir: str) -> pd.DataFrame:
    p = os.path.join(base_dir, "dataset", "train", "train_ground_truth.tsv")
    gt = pd.read_csv(p, sep="\t", dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    return gt


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Return ``{s1_id: {matched_ids …}}`` (empty set for singletons)."""
    out: Dict[str, Set[str]] = {}
    for s1, mids in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        out[s1] = set(mids.split(",")) if mids else set()
    return out


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add *name_norm* and *addr_norm* columns."""
    logger.info("  Pre-processing %d records …", len(df))
    df = df.copy()
    df["name_norm"] = df["business_name"].apply(normalize_name)
    df["addr_norm"] = df["business_address"].apply(normalize_address)
    return df


def build_lookup(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """``entity_id → (name_norm, addr_norm)`` dictionary for O(1) access."""
    return dict(zip(df["entity_id"], zip(df["name_norm"], df["addr_norm"])))
