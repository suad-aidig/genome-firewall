"""
Genome Firewall — decision layer (Module 03).

Turns calibrated probabilities into a report: likely to fail / likely to work /
no-call, each with an evidence category and the genes that drove it.

No-call fires when the system should NOT pretend to know:
  1. low confidence   -> calibrated p inside a band around 0.5
  2. out-of-distribution -> isolate's gene profile far from anything in training
  3. conflict         -> a known carbapenemase is present but the model says "works"
Reporting no-call honestly (rate + accuracy on the calls we DO make) is a
requirement, not a weakness.

Evidence category separates biology from statistics, as the brief demands:
  known_mechanism        -> a known carbapenemase gene drove a "fail"
  statistical_association -> "fail" driven by features that aren't known mechanisms
  no_signal              -> "work" with no resistance markers present
  insufficient_evidence  -> no-call
A model weight is an association, not proof of biological cause — we label it so.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
DRUGS = ["doripenem", "ertapenem", "imipenem", "meropenem"]

# Carbapenem-hydrolysing enzymes (class A KPC/GES, class B MBLs NDM/VIM/IMP/SPM,
# class D OXA-48-like). NOTE: plain blaOXA-1 / blaSHV / blaTEM / blaCTX-M are
# ESBLs, NOT carbapenemases, so they are deliberately excluded here.
CARBAPENEMASE_TOKENS = ["KPC", "NDM", "VIM", "IMP", "GES", "SPM",
                        "OXA-48", "OXA-181", "OXA-232", "OXA-232"]

CONF_BAND = 0.10   # no-call if 0.40 < p < 0.60  (tunable coverage/safety knob)


def is_carbapenemase(col: str) -> bool:
    return any(tok.lower() in col.lower() for tok in CARBAPENEMASE_TOKENS)


def prob(models, feats, drug, x_row):
    clf, cal = models[drug]
    score = clf.decision_function(x_row.reshape(1, -1))
    return float(cal.predict_proba(score.reshape(-1, 1))[0, 1])


def driving_genes(models, feats, drug, x_row, k=5):
    """Signed contribution = coef * presence, for genes actually present."""
    clf, _ = models[drug]
    coef = clf.coef_[0]
    contrib = coef * x_row
    idx = np.argsort(-np.abs(contrib))
    out = []
    for i in idx:
        if x_row[i] == 0:
            continue
        out.append({"gene": feats[i], "weight": round(float(coef[i]), 2),
                    "known_carbapenemase": is_carbapenemase(feats[i])})
        if len(out) >= k:
            break
    return out


def decide(p, x_row, feats, drug, nn_dist, ood_tau):
    present_carbapenemase = any(x_row[i] == 1 and is_carbapenemase(feats[i])
                               for i in range(len(feats)))
    reason = None
    if nn_dist > ood_tau:
        verdict, reason = "no_call", "out_of_distribution"
    elif abs(p - 0.5) < CONF_BAND:
        verdict, reason = "no_call", "low_confidence"
    elif present_carbapenemase and p < 0.5:
        verdict, reason = "no_call", "conflicting_evidence"
    else:
        verdict = "likely_to_fail" if p >= 0.5 else "likely_to_work"

    if verdict == "no_call":
        cat = "insufficient_evidence"
    elif verdict == "likely_to_fail":
        cat = "known_mechanism" if present_carbapenemase else "statistical_association"
    else:
        cat = "no_signal"
    return verdict, cat, reason, present_carbapenemase


def loo_nn_distance(Xd: np.ndarray):
    """Leave-one-out nearest-neighbour Hamming distance for each row."""
    n = len(Xd)
    d = np.full(n, np.nan)
    for i in range(n):
        diff = np.logical_xor(Xd[i], Xd).sum(axis=1).astype(float)
        diff[i] = np.inf
        d[i] = diff.min()
    return d


if __name__ == "__main__":
    X = pd.read_pickle(os.path.join(PROC, "X.pkl"))
    Y = pd.read_pickle(os.path.join(PROC, "Y.pkl"))
    store = pd.read_pickle(os.path.join(ART, "models.pkl"))
    oof = pd.read_pickle(os.path.join(ART, "oof.pkl"))
    models, feats = store["models"], store["features"]
    fidx = {f: i for i, f in enumerate(feats)}

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    coverage, all_reports = {}, {}
    print(f"{'drug':<11}{'no_call%':>9}{'balAcc(called)':>16}{'ood_tau':>9}")
    for ax, drug in zip(axes.ravel(), DRUGS):
        m = Y[drug].notna()
        Xd = X[m]; Xv = Xd.values
        y = Y.loc[m, drug].astype(int).values
        p = oof[drug]["p_cal"].reindex(Xd.index).values
        nn = loo_nn_distance(Xv)
        ood_tau = float(np.percentile(nn, 95))   # novelty beyond 95th pct = OOD

        verdicts, cats = [], []
        for i in range(len(Xd)):
            v, c, r, _ = decide(p[i], Xv[i], feats, drug, nn[i], ood_tau)
            verdicts.append(v); cats.append(c)
        verdicts = np.array(verdicts)
        called = verdicts != "no_call"
        nc_rate = 100 * (~called).mean()
        yhat_called = (p[called] >= 0.5).astype(int)
        bacc = balanced_accuracy_score(y[called], yhat_called) if called.sum() else float("nan")
        coverage[drug] = {"no_call_pct": round(nc_rate, 1),
                          "balanced_acc_on_called": round(float(bacc), 3),
                          "ood_tau": round(ood_tau, 1)}
        print(f"{drug:<11}{nc_rate:>8.1f}%{bacc:>16.3f}{ood_tau:>9.1f}")

        frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.plot(mean_pred, frac_pos, "o-", color="#185FA5")
        ax.set_title(f"{drug}  (Brier {np.mean((p-y)**2):.3f})", fontsize=10)
        ax.set_xlabel("predicted P(resistant)"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Reliability diagrams — calibrated, grouped-CV out-of-fold", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ART, "reliability.png"), dpi=130)
    json.dump(coverage, open(os.path.join(ART, "coverage.json"), "w"), indent=2)
    print("\nsaved -> artifacts/reliability.png, coverage.json")
    print(f"(no-call confidence band = +/-{CONF_BAND}, OOD tau = 95th pct of LOO NN distance)")
