# Genome Firewall — Klebsiella pneumoniae carbapenem panel

Research prototype for Hack-Nation Challenge 06. Predicts likely-to-fail /
likely-to-work / no-call for 4 carbapenems (doripenem, ertapenem, imipenem,
meropenem) from AMRFinderPlus-style gene presence/absence features.
**Not for clinical use — every result must be confirmed by standard lab testing.**

## Status: checkpoint, mid-build
Done: data layer, grouped split, calibrated per-drug models, no-call logic,
evidence categories, reliability diagrams.
Next: Streamlit demo app, final writeup.

## Data provenance
`data/raw/amr_ast_*_KN.csv` — Klebsiella pneumoniae (KN) isolates, AMRFinderPlus
gene/mutation calls (presence=1) paired with lab-measured AST phenotype (last
column, 1=resistant). Sourced from the public AMR_prediction dataset
(github.com/Janaksunuwar/AMR_prediction), itself built from NCBI Pathogen
Detection / BV-BRC isolates. 418 unique isolates, 324 aligned gene features.

## Pipeline (run in order)
```
pip install -r requirements.txt
python3 src/data.py       # unify CSVs, build homology groups (eps=2), grouped split
python3 src/model.py      # per-drug logistic regression + nested grouped-CV calibration
python3 src/decisions.py  # no-call logic, evidence categories, reliability diagrams
```

## Key design decisions (for the memo/mentor)
- **Grouped, not random, split.** 124/418 isolates share an exact AMR-gene
  profile with another isolate. A random split leaks these across train/test;
  we group by Hamming-distance homology (eps=2, threshold justified by
  sensitivity sweep in `data.py`) so no profile spans two splits.
- **Nested grouped CV.** Outer fold = honest held-out test; inner fold fits
  the Platt calibrator without touching outer-test data.
- **No-call** fires on low confidence, out-of-distribution gene profiles, or
  a known carbapenemase conflicting with a "works" prediction.
- **Evidence categories** separate a detected known resistance gene
  (`known_mechanism`) from a purely statistical model signal
  (`statistical_association`) — a model weight is not proof of biological
  cause.

## Files
```
src/data.py        data loading, homology grouping, leakage-free split
src/model.py        per-drug models, nested grouped-CV calibration, metrics
src/decisions.py    no-call logic, evidence categories, reliability plot
data/raw/           source CSVs
data/processed/     unified matrix, labels, groups, split (pickled)
artifacts/          trained models, metrics, coverage stats, reliability.png
```
