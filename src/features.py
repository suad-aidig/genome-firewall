"""
Genome Firewall — mechanism-aggregate features.

Why this exists: the carbapenemase signal is shattered across many rare columns
(blaKPC-2, blaKPC-3, blaKPC-4, blaKPC-7, blaKPC=PARTIAL, …). Of 324 gene features,
211 appear in fewer than 5 isolates. An L2-regularized logistic regression can't
concentrate weight on "a KPC is present" when that fact is split across a dozen
sparse variants — so the model was diluted and, at the 0.5 threshold, lost to a
one-line rule ("any known carbapenemase present -> resistant").

The fix is principled feature engineering: collapse each carbapenemase FAMILY into
a single presence flag (plus an overall "any carbapenemase" flag). These are
deterministic per-row functions of the existing gene calls — they add no new
information and cannot leak; they just present the signal in a form the linear
model can weight cleanly. With them, the calibrated model matches the mechanism
rule within fold-to-fold noise while keeping calibration, abstention, and the
interpretable coefficient-based evidence layer.

The aggregates are model INPUTS only. Grouping, the out-of-distribution
nearest-neighbour test, and the driving-gene evidence display all stay on the raw
genomic profile, so the honesty story and the leakage-free split are unchanged.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Curated carbapenemase families (mirrors the curated list in decisions.py,
# grouped by enzyme family). Class A: KPC, GES. Class B (MBLs): NDM, VIM, IMP,
# SPM. Class D: OXA-48-like. ESBLs (TEM / SHV / CTX-M / OXA-1) are deliberately
# excluded — they are NOT carbapenemases.
MECH_FAMILIES = [
    ("KPC",       ["kpc"]),
    ("NDM",       ["ndm"]),
    ("VIM",       ["vim"]),
    ("IMP",       ["imp"]),
    ("GES",       ["ges"]),
    ("SPM",       ["spm"]),
    ("OXA48like", ["oxa-48", "oxa-181", "oxa-232"]),
]
MECH_PREFIX = "MECH_"
ANY_NAME = MECH_PREFIX + "ANY_CARBAPENEMASE"


def is_mechanism_feature(name: str) -> bool:
    """True for the engineered aggregate columns (so callers can hide them from
    the gene-level evidence display)."""
    return name.startswith(MECH_PREFIX)


def mechanism_spec(raw_feats: list[str]):
    """Ordered [(mech_name, [raw column indices]), …] for the families actually
    present in this feature set, followed by the overall ANY aggregate. Absent
    families (no matching column) are skipped, so the model feature list stays
    clean. Deterministic in `raw_feats` order → identical for matrix and row."""
    idx = {f: i for i, f in enumerate(raw_feats)}
    spec, any_ids = [], []
    for fam, toks in MECH_FAMILIES:
        cols = [idx[f] for f in raw_feats
                if any(t in f.lower() for t in toks)]
        if cols:
            spec.append((MECH_PREFIX + fam, cols))
            any_ids.extend(cols)
    spec.append((ANY_NAME, sorted(set(any_ids))))
    return spec


def model_feature_names(raw_feats: list[str]) -> list[str]:
    """Full model input order: raw features first, then the aggregates."""
    return list(raw_feats) + [name for name, _ in mechanism_spec(raw_feats)]


def augment_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Return X with the mechanism-aggregate columns appended (int8)."""
    raw = list(X.columns)
    A = X.values
    add = {}
    for name, cols in mechanism_spec(raw):
        add[name] = (A[:, cols].sum(axis=1) > 0).astype(np.int8) if cols \
            else np.zeros(len(X), dtype=np.int8)
    return pd.concat([X, pd.DataFrame(add, index=X.index)], axis=1)


def augment_row(raw_row: np.ndarray, raw_feats: list[str]) -> np.ndarray:
    """Map a raw feature row (aligned to raw_feats) to the model feature row
    (raw values ++ aggregate flags, in model_feature_names order)."""
    raw_row = np.asarray(raw_row, dtype=np.int8)
    extra = [int(raw_row[cols].any()) if len(cols) else 0
             for _, cols in mechanism_spec(raw_feats)]
    return np.concatenate([raw_row, np.array(extra, dtype=np.int8)])
