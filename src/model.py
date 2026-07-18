"""
Genome Firewall — predictor + calibration (Modules 02, partial 03).

Design choices, and why:
- One regularized logistic regression per drug (brief's recommended baseline):
  interpretable, fast, coefficients map to genes for the evidence layer.
- class_weight='balanced' because resistant/susceptible ratios vary a lot.
- Honest evaluation via NESTED GROUPED cross-validation:
    * OUTER GroupKFold  -> test isolates are never clonal-related to training
                           (groups = the eps=2 homology groups from data.py)
    * INNER GroupKFold  -> out-of-fold scores feed a Platt (sigmoid) calibrator,
                           so probabilities are calibrated without touching the
                           outer-test fold.
  We pool the outer-test calibrated probabilities across folds to get one honest,
  leakage-free prediction per isolate, and also report fold-to-fold spread.
- Deterministic molecular-target gate: if the drug's target is absent the drug
  cannot act -> force 'likely to fail' regardless of the model. For carbapenems
  in K. pneumoniae the target (penicillin-binding proteins) is intrinsic and
  effectively always present, so the gate defers to the model here; it is
  implemented correctly so the architecture generalizes to drug classes where
  target loss causes intrinsic resistance. We do NOT fabricate gate activity.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (balanced_accuracy_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss)

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DRUGS = ["doripenem", "ertapenem", "imipenem", "meropenem"]
C_REG = 1.0            # L2 strength (tunable; fixed for the baseline)
N_OUTER, N_INNER = 5, 3


def _base_clf():
    return LogisticRegression(C=C_REG, class_weight="balanced",
                              solver="liblinear", max_iter=2000)


def _platt(scores, y):
    """Fit a 1-D sigmoid calibrator on decision scores."""
    cal = LogisticRegression(solver="liblinear")
    cal.fit(scores.reshape(-1, 1), y)
    return cal


def target_present(X_row: pd.Series, drug: str) -> bool:
    """Deterministic target gate. Carbapenem target = PBPs (intrinsic in
    K. pneumoniae) -> always present here. Structured so other drug classes can
    plug in a real target-gene check."""
    return True


def calibrated_oof(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """Return pooled out-of-fold calibrated probs + raw (uncalibrated) probs,
    plus per-fold AUROC for a variance estimate."""
    Xv, yv, gv = X.values, y.values, groups.values
    oof_cal = np.full(len(yv), np.nan)
    oof_raw = np.full(len(yv), np.nan)
    fold_auc = []
    outer = GroupKFold(n_splits=min(N_OUTER, len(np.unique(gv))))
    for tr, te in outer.split(Xv, yv, gv):
        clf = _base_clf().fit(Xv[tr], yv[tr])
        oof_raw[te] = clf.predict_proba(Xv[te])[:, 1]
        # inner grouped CV -> scores to fit the Platt calibrator
        inner = GroupKFold(n_splits=min(N_INNER, len(np.unique(gv[tr]))))
        inner_scores = np.full(len(tr), np.nan)
        for itr, ite in inner.split(Xv[tr], yv[tr], gv[tr]):
            c = _base_clf().fit(Xv[tr][itr], yv[tr][itr])
            inner_scores[ite] = c.decision_function(Xv[tr][ite])
        cal = _platt(inner_scores, yv[tr])
        te_scores = clf.decision_function(Xv[te]).reshape(-1, 1)
        oof_cal[te] = cal.predict_proba(te_scores)[:, 1]
        if len(np.unique(yv[te])) == 2:
            fold_auc.append(roc_auc_score(yv[te], oof_cal[te]))
    return oof_cal, oof_raw, np.array(fold_auc)


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "n": int(len(y)),
        "resistant_frac": round(float(np.mean(y)), 3),
        "balanced_acc": round(balanced_accuracy_score(y, yhat), 3),
        "recall_resistant": round(recall_score(y, yhat, pos_label=1), 3),
        "recall_susceptible": round(recall_score(y, yhat, pos_label=0), 3),
        "f1": round(f1_score(y, yhat, pos_label=1), 3),
        "auroc": round(roc_auc_score(y, p), 3),
        "pr_auc": round(average_precision_score(y, p), 3),
        "brier": round(brier_score_loss(y, p), 3),
    }


def fit_final(X, y, groups):
    """Model + calibrator trained on ALL data, for the demo."""
    clf = _base_clf().fit(X.values, y.values)
    gv = groups.values
    inner = GroupKFold(n_splits=min(N_INNER, len(np.unique(gv))))
    sc = np.full(len(y), np.nan)
    for itr, ite in inner.split(X.values, y.values, gv):
        c = _base_clf().fit(X.values[itr], y.values[itr])
        sc[ite] = c.decision_function(X.values[ite])
    cal = _platt(sc, y.values)
    return clf, cal


if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    X = pd.read_pickle(os.path.join(PROC, "X.pkl"))
    Y = pd.read_pickle(os.path.join(PROC, "Y.pkl"))
    G = pd.read_pickle(os.path.join(PROC, "groups.pkl"))

    all_metrics, oof_store, models = {}, {}, {}
    print(f"{'drug':<11}{'n':>4}{'%R':>5}{'balAcc':>8}{'rec_R':>7}{'rec_S':>7}"
          f"{'F1':>6}{'AUROC':>7}{'PRAUC':>7}{'Brier↓':>8}{'Brier_raw':>10}{'foldAUC':>16}")
    for d in DRUGS:
        m = Y[d].notna()
        Xd, yd, gd = X[m], Y.loc[m, d].astype(int), G[m]
        p_cal, p_raw, fauc = calibrated_oof(Xd, yd, gd)
        mc = metrics(yd.values, p_cal)
        braw = round(brier_score_loss(yd.values, p_raw), 3)
        all_metrics[d] = {**mc, "brier_uncalibrated": braw,
                          "fold_auroc_mean": round(float(fauc.mean()), 3),
                          "fold_auroc_std": round(float(fauc.std()), 3)}
        oof_store[d] = pd.DataFrame({"y": yd.values, "p_cal": p_cal},
                                    index=Xd.index)
        models[d] = fit_final(Xd, yd, gd)
        print(f"{d:<11}{mc['n']:>4}{mc['resistant_frac']*100:>4.0f}%"
              f"{mc['balanced_acc']:>8}{mc['recall_resistant']:>7}"
              f"{mc['recall_susceptible']:>7}{mc['f1']:>6}{mc['auroc']:>7}"
              f"{mc['pr_auc']:>7}{mc['brier']:>8}{braw:>10}"
              f"{f'{fauc.mean():.2f}±{fauc.std():.2f}':>16}")

    json.dump(all_metrics, open(os.path.join(ART, "metrics.json"), "w"), indent=2)
    pd.to_pickle(oof_store, os.path.join(ART, "oof.pkl"))
    pd.to_pickle({"models": models, "features": list(X.columns), "drugs": DRUGS},
                 os.path.join(ART, "models.pkl"))
    print("\nsaved -> artifacts/metrics.json, oof.pkl, models.pkl")
    print("Brier↓ = calibrated (lower is better); compare to Brier_raw to see calibration effect.")
