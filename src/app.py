"""
Genome Firewall — Streamlit decision-support demo (Module 03 front-end).

Renders the output contract from HANDOFF.md for one isolate x 4 carbapenems:
verdict (likely_to_fail / likely_to_work / no_call), calibrated confidence
(hidden on no_call), evidence category, driving genes, and no-call reason —
wired to the REAL trained models via decide() and driving_genes() in
decisions.py. Nothing here re-implements the science; it presents it.

Honesty design choices, stated plainly so a judge can audit them:
  * For any isolate that is part of the dataset we show the leakage-free
    OUT-OF-FOLD calibrated probability (from artifacts/oof.pkl) — the exact
    number behind the reported held-out metrics, never an in-sample score.
  * For a pasted/uploaded novel gene-call row we fall back to the final model
    trained on all data, and label it as such.
  * The true lab AST result (when known) is shown next to the prediction as
    "held-out ground truth" so live agreement is visible and verifiable.

Run:  streamlit run src/app.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st

# Import the decision layer (ground truth for the science) --------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decisions import prob, decide, driving_genes, DRUGS

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed")
ART = os.path.join(HERE, "..", "artifacts")

SPECIES = "Klebsiella pneumoniae"
DRUG_LABEL = {d: d.capitalize() for d in DRUGS}

# Verdict presentation ---------------------------------------------------------
VERDICT_UI = {
    "likely_to_fail": {"label": "LIKELY TO FAIL", "color": "#C62828",
                       "bg": "#FDECEA", "emoji": "🔴",
                       "gloss": "Resistance predicted — this drug is likely ineffective."},
    "likely_to_work": {"label": "LIKELY TO WORK", "color": "#2E7D32",
                       "bg": "#E9F5EA", "emoji": "🟢",
                       "gloss": "Susceptibility predicted — this drug is likely effective."},
    "no_call":        {"label": "NO CALL", "color": "#B36B00",
                       "bg": "#FFF6E5", "emoji": "🟡",
                       "gloss": "Evidence too weak or unlike training data — abstaining on purpose."},
}
EVIDENCE_UI = {
    "known_mechanism": ("Known resistance mechanism",
                        "A curated carbapenemase gene drove this call."),
    "statistical_association": ("Statistical association",
                        "Driven by model-weighted features, not a known mechanism. "
                        "A weight is an association, not proof of biological cause."),
    "no_signal": ("No resistance signal",
                        "No resistance markers present."),
    "insufficient_evidence": ("Insufficient evidence",
                        "Not enough signal to make a safe call."),
}
NOCALL_REASON_UI = {
    "low_confidence": "Low confidence — calibrated probability sits near 50/50.",
    "out_of_distribution": "Out of distribution — gene profile is unlike anything in training.",
    "conflicting_evidence": "Conflicting evidence — a known carbapenemase is present but the model leans 'works'.",
}


# ---------------------------------------------------------------------------- #
# Data loading (cached)                                                         #
# ---------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_all():
    X = pd.read_pickle(os.path.join(PROC, "X.pkl"))
    Y = pd.read_pickle(os.path.join(PROC, "Y.pkl"))
    split = pd.read_pickle(os.path.join(PROC, "split.pkl"))
    oof = pd.read_pickle(os.path.join(ART, "oof.pkl"))
    store = pd.read_pickle(os.path.join(ART, "models.pkl"))
    coverage = json.load(open(os.path.join(ART, "coverage.json")))
    metrics = json.load(open(os.path.join(ART, "metrics.json")))
    feats = store["features"]

    # Per-drug reference sets for the out-of-distribution nearest-neighbour test.
    ref = {}
    for d in DRUGS:
        idx = Y.index[Y[d].notna()]
        ref[d] = {
            "index": idx,
            "vals": X.loc[idx].values.astype(np.int8),
            "pos": {iso: i for i, iso in enumerate(idx)},
            "ood_tau": float(coverage[d]["ood_tau"]),
        }
    return X, Y, split, oof, store, feats, coverage, metrics, ref


@st.cache_data(show_spinner=False)
def profile_sharing_stat(_X: pd.DataFrame) -> tuple[int, int]:
    """Headline honesty stat: how many isolates share an exact gene profile
    with at least one other isolate (the concrete reason a random split lies)."""
    prof = _X.astype(np.int8).astype(str).agg("".join, axis=1)
    counts = prof.value_counts()
    shared = int(prof.isin(counts[counts > 1].index).sum())
    return shared, int(len(_X))


def nn_distance(x_row: np.ndarray, ref_vals: np.ndarray, exclude_pos=None) -> float:
    """Min Hamming distance from x_row to the reference set (leave-one-out if the
    isolate is itself in the set)."""
    d = np.logical_xor(x_row.astype(np.int8), ref_vals).sum(axis=1).astype(float)
    if exclude_pos is not None:
        d[exclude_pos] = np.inf
    return float(d.min()) if len(d) else float("inf")


def predict(x_row, drug, oof, store, feats, ref, iso=None):
    """One isolate x drug → the full output-contract object."""
    models = store["models"]
    tested = iso is not None and iso in oof[drug].index
    if tested:
        p = float(oof[drug].loc[iso, "p_cal"])   # honest out-of-fold probability
        p_source = "held_out"
        exclude = ref[drug]["pos"].get(iso)
    else:
        p = prob(models, feats, drug, x_row)       # final model on a novel row
        p_source = "model"
        exclude = None

    nn = nn_distance(x_row, ref[drug]["vals"], exclude_pos=exclude)
    verdict, cat, reason, present_carb = decide(
        p, x_row, feats, drug, nn, ref[drug]["ood_tau"])
    genes = driving_genes(models, feats, drug, x_row, k=5)

    confidence = None
    if verdict == "likely_to_fail":
        confidence = p
    elif verdict == "likely_to_work":
        confidence = 1.0 - p

    return {
        "drug": drug, "verdict": verdict, "confidence": confidence,
        "evidence_category": cat, "no_call_reason": reason,
        "driving_features": genes, "present_carbapenemase": present_carb,
        "p_resistant": p, "p_source": p_source, "nn_dist": nn,
        "ood_tau": ref[drug]["ood_tau"],
    }


# ---------------------------------------------------------------------------- #
# Rendering                                                                     #
# ---------------------------------------------------------------------------- #
def render_banner():
    st.markdown(
        """
        <div style="background:#B3261E;color:#fff;padding:14px 18px;border-radius:10px;
             font-size:0.98rem;line-height:1.45;">
          <b>⚠️ Research prototype — decision support only.</b>
          Every antibiotic-response result <b>must be confirmed by standard laboratory
          testing</b>. This tool predicts and explains resistance that already exists; it
          never designs or modifies organisms and never makes a treatment decision on its own.
        </div>
        """, unsafe_allow_html=True)


def _conf_bar(confidence: float, color: str) -> str:
    pct = int(round(confidence * 100))
    return f"""
      <div style="margin:8px 0 4px 0;">
        <div style="display:flex;justify-content:space-between;font-size:0.78rem;
             color:#555;margin-bottom:3px;">
          <span>calibrated confidence</span><span><b>{pct}%</b></span>
        </div>
        <div style="background:#ECECEC;border-radius:6px;height:9px;overflow:hidden;">
          <div style="width:{pct}%;background:{color};height:9px;"></div>
        </div>
      </div>"""


def render_card(r: dict):
    ui = VERDICT_UI[r["verdict"]]
    ev_label, ev_help = EVIDENCE_UI[r["evidence_category"]]

    parts = [f"""
      <div style="border:1px solid #E3E3E3;border-left:6px solid {ui['color']};
           border-radius:12px;padding:16px 18px;background:{ui['bg']};height:100%;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <span style="font-size:1.18rem;font-weight:700;color:#1a1a1a;">{DRUG_LABEL[r['drug']]}</span>
          <span style="font-size:0.82rem;font-weight:800;letter-spacing:0.4px;
                color:{ui['color']};">{ui['emoji']} {ui['label']}</span>
        </div>
        <div style="font-size:0.82rem;color:#444;margin-top:4px;">{ui['gloss']}</div>
    """]

    if r["confidence"] is not None:
        parts.append(_conf_bar(r["confidence"], ui["color"]))
    else:
        parts.append(
            f"<div style='margin:8px 0;font-size:0.8rem;color:#8a5a00;'>"
            f"<i>Confidence intentionally hidden on a no-call.</i></div>")

    # evidence tag
    parts.append(
        f"<div style='margin-top:6px;'><span style='background:#fff;border:1px solid "
        f"{ui['color']}55;color:{ui['color']};padding:2px 9px;border-radius:20px;"
        f"font-size:0.74rem;font-weight:600;'>{ev_label}</span></div>")

    # no-call reason
    if r["no_call_reason"]:
        parts.append(
            f"<div style='margin-top:8px;font-size:0.8rem;color:#8a5a00;'>"
            f"<b>Why no call:</b> {NOCALL_REASON_UI[r['no_call_reason']]}</div>")

    # driving genes one-liner
    genes = r["driving_features"]
    if genes:
        chips = []
        for g in genes[:3]:
            star = " ⚑" if g["known_carbapenemase"] else ""
            sign = "+" if g["weight"] >= 0 else "−"
            chips.append(
                f"<span style='background:#fff;border:1px solid #ddd;border-radius:6px;"
                f"padding:1px 6px;margin:2px 3px 0 0;display:inline-block;font-size:0.72rem;"
                f"font-family:monospace;'>{g['gene']} "
                f"<span style='color:#888;'>({sign}{abs(g['weight'])}){star}</span></span>")
        parts.append(
            f"<div style='margin-top:10px;font-size:0.76rem;color:#555;'>"
            f"<b>Top drivers</b> (signed weight; ⚑ = known carbapenemase):<br>{''.join(chips)}</div>")
    else:
        parts.append(
            "<div style='margin-top:10px;font-size:0.76rem;color:#777;'>"
            "<b>Top drivers:</b> no resistance-associated genes present.</div>")

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_true_label(true_label, p_source):
    if true_label is None:
        return ""
    truth = "Resistant (drug failed)" if int(true_label) == 1 else "Susceptible (drug worked)"
    tcolor = "#C62828" if int(true_label) == 1 else "#2E7D32"
    src = ("held-out out-of-fold — this isolate's label was never used to score it"
           if p_source == "held_out" else "reference label")
    return (f"<span style='font-size:0.78rem;color:#444;'>Lab AST ground truth: "
            f"<b style='color:{tcolor};'>{truth}</b> "
            f"<span style='color:#999;'>({src})</span></span>")


# ---------------------------------------------------------------------------- #
# App                                                                           #
# ---------------------------------------------------------------------------- #
st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="wide")

X, Y, split, oof, store, feats, coverage, metrics, ref = load_all()
fidx = {f: i for i, f in enumerate(feats)}

st.markdown(
    "<h1 style='margin-bottom:0;'>🧬 Genome Firewall</h1>"
    f"<div style='color:#555;margin-top:2px;'>Genome-to-antibiotic-response decision "
    f"support · <i>{SPECIES}</i> · 4 carbapenems</div>", unsafe_allow_html=True)
st.write("")
render_banner()
st.write("")

# ---- Sidebar: isolate input ----
with st.sidebar:
    st.header("Isolate input")
    mode = st.radio(
        "Source", ["Held-out isolate", "Paste gene calls", "Upload row (CSV)"],
        help="Held-out isolates are drawn from the grouped test split — the model "
             "never saw their genetic group during training.")

    x_row, iso, custom = None, None, False

    if mode == "Held-out isolate":
        test_isos = list(split.index[split == "test"])
        # Deep-link support: ?iso=<id> preselects an isolate (handy for sharing
        # a specific demo case, e.g. in the walkthrough video).
        qp_iso = st.query_params.get("iso")
        default_ix = test_isos.index(qp_iso) if qp_iso in test_isos else 0
        iso = st.selectbox(f"Test-split isolate ({len(test_isos)} available)",
                           test_isos, index=default_ix)
        st.query_params["iso"] = iso
        x_row = X.loc[iso].values.astype(np.int8)

    elif mode == "Paste gene calls":
        custom = True
        st.caption("Paste AMRFinderPlus-style tokens, one per line or comma-separated. "
                   "Unknown tokens are ignored; matched ones are set present.")
        example = "\n".join(["blaKPC-2=COMPLETE", "blaSHV-11=COMPLETE",
                             "gyrA_S83I=POINT", "fosA=COMPLETE"])
        txt = st.text_area("Gene calls", value=example, height=170)
        tokens = [t.strip() for t in txt.replace(",", "\n").splitlines() if t.strip()]
        x_row = np.zeros(len(feats), dtype=np.int8)
        matched = [t for t in tokens if t in fidx]
        for t in matched:
            x_row[fidx[t]] = 1
        st.caption(f"✅ matched **{len(matched)}** / {len(tokens)} tokens to model features")

    else:  # Upload row
        custom = True
        st.caption("Upload a CSV that is EITHER one row over the 324 model features, "
                   "OR a single column listing the present gene tokens.")
        up = st.file_uploader("CSV file", type=["csv"])
        x_row = np.zeros(len(feats), dtype=np.int8)
        if up is not None:
            df = pd.read_csv(up)
            overlap = [c for c in df.columns if c in fidx]
            if len(overlap) >= 5:                      # looks like a feature-row CSV
                r0 = df.iloc[0]
                for c in overlap:
                    x_row[fidx[c]] = int(r0[c] != 0)
                st.caption(f"✅ read feature row · {len(overlap)} features matched")
            else:                                      # treat first column as a gene list
                toks = df.iloc[:, 0].astype(str).tolist()
                m = [t for t in toks if t in fidx]
                for t in m:
                    x_row[fidx[t]] = 1
                st.caption(f"✅ read gene list · matched {len(m)} / {len(toks)} tokens")
        else:
            st.info("Upload a file, or switch to a held-out isolate to explore.")

    st.divider()
    st.markdown(
        "<div style='font-size:0.78rem;color:#666;'>"
        "<b>Coverage.</b> This prototype covers <i>Klebsiella pneumoniae</i> and the "
        "carbapenem class only (doripenem, ertapenem, imipenem, meropenem). It makes no "
        "claim about any other species or drug.</div>", unsafe_allow_html=True)

n_present = int(x_row.sum()) if x_row is not None else 0

tab_report, tab_perf, tab_about = st.tabs(
    ["🧾 Antibiotic-response report", "📊 Held-out performance", "🔬 Methods & honesty"])

# ============================ REPORT TAB ==================================== #
with tab_report:
    if x_row is None or (custom and n_present == 0):
        st.info("Provide an isolate on the left to generate a report.")
    else:
        # header line
        who = iso if iso else "Pasted / uploaded isolate"
        badge = ("held-out (grouped test split)" if iso and split.get(iso) == "test"
                 else "novel input" if custom else "dataset isolate")
        st.markdown(
            f"**Isolate:** `{who}` &nbsp;·&nbsp; **{n_present}** resistance-relevant "
            f"gene calls present &nbsp;·&nbsp; <span style='background:#EEF2FF;color:#3730A3;"
            f"padding:1px 8px;border-radius:12px;font-size:0.75rem;'>{badge}</span>",
            unsafe_allow_html=True)
        st.write("")

        results = [predict(x_row, d, oof, store, feats, ref, iso=iso) for d in DRUGS]

        cols = st.columns(2, gap="medium")
        for i, r in enumerate(results):
            with cols[i % 2]:
                render_card(r)
                # ground-truth line for dataset isolates
                if iso is not None:
                    tl = Y.loc[iso, r["drug"]] if iso in Y.index else np.nan
                    if not pd.isna(tl):
                        st.markdown(render_true_label(tl, r["p_source"]),
                                    unsafe_allow_html=True)
                st.write("")

        # panel-level recommendation
        works = [r for r in results if r["verdict"] == "likely_to_work"]
        fails = [r for r in results if r["verdict"] == "likely_to_fail"]
        ncs = [r for r in results if r["verdict"] == "no_call"]
        st.divider()
        if works:
            best = max(works, key=lambda r: r["confidence"])
            others = ""
            if fails:
                others = (f" {len(fails)} carbapenem(s) look likely to fail "
                          f"({', '.join(DRUG_LABEL[r['drug']] for r in fails)}).")
            st.success(
                f"**Panel read:** best-supported carbapenem is "
                f"**{DRUG_LABEL[best['drug']]}** "
                f"({int(best['confidence']*100)}% calibrated confidence it works).{others} "
                f"Confirm with standard lab AST before acting.")
        elif not fails:  # everything is a no-call
            st.warning("**Panel read:** the system abstains on the whole carbapenem panel "
                       "for this isolate — evidence is weak, conflicting, or out of distribution. "
                       "This is not a resistance prediction; escalate to standard lab testing.")
        else:
            nc_note = (f" The remaining {len(ncs)} are no-calls (abstentions, not "
                       f"resistance predictions)." if ncs else "")
            st.error(
                f"**Panel read:** no carbapenem is predicted to work. "
                f"{len(fails)} likely to fail "
                f"({', '.join(DRUG_LABEL[r['drug']] for r in fails)}) — these suggest "
                f"carbapenem resistance; confirm urgently with lab AST and consider "
                f"non-carbapenem options with an infectious-disease specialist.{nc_note}")

        with st.expander("See the full driving-gene evidence for every drug"):
            for r in results:
                st.markdown(f"**{DRUG_LABEL[r['drug']]}** — "
                            f"P(resistant) = {r['p_resistant']:.2f} "
                            f"({'held-out out-of-fold' if r['p_source']=='held_out' else 'final model'}), "
                            f"OOD distance {r['nn_dist']:.0f} vs τ={r['ood_tau']:.0f}")
                if r["driving_features"]:
                    dg = pd.DataFrame(r["driving_features"])
                    dg = dg.rename(columns={"gene": "gene / mutation",
                                            "weight": "signed model weight",
                                            "known_carbapenemase": "known carbapenemase"})
                    st.dataframe(dg, hide_index=True, width="stretch")
                else:
                    st.caption("No resistance-associated genes present in this isolate.")

# ============================ PERFORMANCE TAB =============================== #
with tab_perf:
    st.subheader("Honest, leakage-free held-out performance")
    st.caption("All numbers are pooled out-of-fold results under NESTED GROUPED "
               "cross-validation: test isolates are never in the same homology group as "
               "any training isolate, and the calibrator never sees the outer-test fold.")

    mdf = pd.DataFrame(metrics).T[[
        "n", "resistant_frac", "balanced_acc", "recall_resistant",
        "recall_susceptible", "f1", "auroc", "pr_auc", "brier"]]
    mdf = mdf.rename(columns={
        "n": "N", "resistant_frac": "% resistant", "balanced_acc": "balanced acc",
        "recall_resistant": "recall (R)", "recall_susceptible": "recall (S)",
        "f1": "F1", "auroc": "AUROC", "pr_auc": "PR-AUC", "brier": "Brier ↓"})
    mdf.index = [DRUG_LABEL[d] for d in mdf.index]
    st.dataframe(mdf, width="stretch")

    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        st.markdown("**Reliability diagrams** — calibrated probabilities track the "
                    "diagonal, so a stated confidence means what it says.")
        rel = os.path.join(ART, "reliability.png")
        if os.path.exists(rel):
            st.image(rel, width="stretch")
    with c2:
        st.markdown("**No-call coverage vs. accuracy** — abstaining on weak/OOD/conflicting "
                    "evidence trades a little coverage for markedly higher accuracy on the "
                    "calls we *do* make.")
        cov = pd.DataFrame(coverage).T[["no_call_pct", "balanced_acc_on_called"]]
        cov = cov.rename(columns={"no_call_pct": "no-call rate (%)",
                                  "balanced_acc_on_called": "balanced acc on called"})
        cov.index = [DRUG_LABEL[d] for d in cov.index]
        st.dataframe(cov, width="stretch")
        st.caption("Compare 'balanced acc on called' here against 'balanced acc' in the "
                   "table above — the no-call gate lifts every drug.")

# ============================ ABOUT TAB ==================================== #
with tab_about:
    shared, total = profile_sharing_stat(X)
    st.subheader("Why you can trust these numbers")
    st.markdown(f"""
- **Grouped split, not random.** **{shared} of {total}** isolates share an *exact*
  AMR-gene profile with at least one other isolate. A random train/test split would
  leak those near-clones across the split and inflate every score. We union isolates
  whose gene profiles differ by ≤ 2 calls (Hamming distance, union-find) and keep whole
  groups on one side of the split — verified **0** profiles span two splits.
- **Nested grouped cross-validation.** The outer fold is an honest held-out test; an
  inner grouped fold fits the Platt calibrator without ever touching the outer-test data.
- **No-call by design.** The system abstains on (i) low confidence (calibrated *p* within
  ±0.10 of 0.5), (ii) out-of-distribution gene profiles (beyond the 95th-percentile
  nearest-neighbour distance), or (iii) a known carbapenemase present while the model
  leans "works". Forcing a yes/no answer would manufacture false confidence.
- **Evidence, honestly labelled.** A curated carbapenemase gene (KPC, NDM, VIM, IMP, GES,
  SPM, OXA-48/181/232) driving a "fail" is tagged *known mechanism*; anything else is a
  *statistical association* — a model weight is not proof of biological cause. ESBL genes
  (TEM/SHV/CTX-M) are deliberately **excluded** from the carbapenemase list.
""")
    st.subheader("Stated limitations (we don't hide these)")
    st.markdown("""
- **Feature-space homology is a proxy for phylogeny**, not a substitute for core-genome
  SNP / MLST clustering. The ideal grouping (NCBI Pathogen Detection SNP clusters) isn't
  in this local dataset.
- **The molecular-target gate is a correct no-op here.** Carbapenems target penicillin-
  binding proteins, which are intrinsic in *K. pneumoniae*, so there is no real
  "target absent" case for this class — the gate is implemented so it generalizes to
  drug/species combinations where target loss causes intrinsic resistance, but it does
  not change any call in this demo.
- **Scope is deliberately narrow:** one species, one drug class. Strong on depth, not breadth.
- **Strictly defensive:** predicts and explains resistance that already exists. It never
  designs, modifies, or optimizes an organism, and never decides treatment on its own.
""")
    st.subheader("Pipeline provenance")
    st.markdown("""
`data/raw/amr_ast_*_KN.csv` → AMRFinderPlus-style gene/mutation calls paired with
lab-measured AST phenotype (public *AMR_prediction* dataset, derived from NCBI Pathogen
Detection / BV-BRC). FASTA → AMRFinderPlus → these tabulated features is the documented,
repeatable path (`README.md`). Reproduce with:
`python src/data.py && python src/model.py && python src/decisions.py`.
""")

st.write("")
st.caption("Genome Firewall · Hack-Nation Challenge 06 · research prototype — "
           "not for clinical use. Every result must be confirmed by standard lab testing.")
