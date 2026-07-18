"""
Genome Firewall — data layer.

Loads the Klebsiella pneumoniae carbapenem panel (AMRFinderPlus-annotated gene
presence/absence + lab AST phenotype), builds a unified feature matrix, and
creates a LEAKAGE-FREE grouped split.

Why grouped, not random: many isolates share near-identical AMR-gene profiles
(clonal / closely related). A random split leaks those across train and test and
inflates every metric. We approximate genetic relatedness with a homology proxy:
isolates whose gene profiles differ by <= EPS gene calls are unioned into one
group (union-find), and whole groups are kept within a single split.

Honest limitation: feature-space similarity is a proxy for phylogeny, not a
substitute for core-genome SNP / MLST clustering. Ideally we'd group by NCBI
Pathogen Detection PDS SNP clusters; that metadata isn't in this local dataset.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

DRUGS = ["doripenem", "ertapenem", "imipenem", "meropenem"]
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Known carbapenem-relevant resistance mechanisms (for the honest "evidence
# category" layer later). Matched as substrings against AMRFinderPlus symbols.
KNOWN_CARBAPENEMASES = ["KPC", "NDM", "OXA-48", "OXA-181", "OXA-232",
                        "OXA-232", "VIM", "IMP", "GES", "SPM"]


def load_unified():
    """Return X (isolate x gene, int8), Y (isolate x drug, with NaN where untested)."""
    frames = {d: pd.read_csv(os.path.join(RAW, f"amr_ast_{d}_KN.csv")) for d in DRUGS}
    feats = sorted(c for c in next(iter(frames.values())).columns
                   if c not in ("Isolate", DRUGS[0]))
    # sanity: identical feature space across drugs
    for d, df in frames.items():
        fs = sorted(c for c in df.columns if c not in ("Isolate", d))
        assert fs == feats, f"feature mismatch in {d}"

    isolates = sorted(set().union(*[set(df["Isolate"]) for df in frames.values()]))
    X = pd.DataFrame(0, index=isolates, columns=feats, dtype=np.int8)
    Y = pd.DataFrame(np.nan, index=isolates, columns=DRUGS, dtype=float)
    for d, df in frames.items():
        df = df.set_index("Isolate")
        X.loc[df.index, feats] = df[feats].astype(np.int8).values
        Y.loc[df.index, d] = df[d].astype(float).values
    return X, Y


def build_groups(X: pd.DataFrame, eps: int = 2):
    """Union-find grouping: link isolates with Hamming distance <= eps.

    Returns an array of group ids aligned to X.index.
    """
    A = X.values.astype(np.int8)
    n = A.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Hamming distance via matrix ops: d(i,j) = popcount(a_i XOR a_j).
    # For 418x324 this is trivially cheap. Compute row-by-row to save memory.
    for i in range(n):
        # distance from i to all j>i
        diff = np.logical_xor(A[i], A[i + 1:]).sum(axis=1)
        for off in np.where(diff <= eps)[0]:
            union(i, i + 1 + off)

    roots = np.array([find(i) for i in range(n)])
    # remap to compact 0..k-1
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def grouped_split(gid, y_index, seed=42, frac=(0.60, 0.15, 0.25)):
    """Assign whole groups to train / calibration / test. No group spans splits."""
    rng = np.random.default_rng(seed)
    groups = np.unique(gid)
    rng.shuffle(groups)
    n = len(groups)
    n_tr = int(round(frac[0] * n))
    n_cal = int(round(frac[1] * n))
    g_train = set(groups[:n_tr])
    g_cal = set(groups[n_tr:n_tr + n_cal])
    g_test = set(groups[n_tr + n_cal:])
    split = np.where(np.isin(gid, list(g_train)), "train",
             np.where(np.isin(gid, list(g_cal)), "calibration", "test"))
    return pd.Series(split, index=y_index, name="split")


if __name__ == "__main__":
    os.makedirs(PROC, exist_ok=True)
    X, Y = load_unified()
    print(f"Unified matrix: {X.shape[0]} isolates x {X.shape[1]} genes")

    # threshold sensitivity — justify the eps we pick
    print("\nHomology-threshold sensitivity (eps = max gene-call differences to group):")
    print(f"  {'eps':>3} {'n_groups':>9} {'largest':>8} {'singletons':>11}")
    for eps in (0, 1, 2, 3, 5):
        g = build_groups(X, eps=eps)
        sizes = pd.Series(g).value_counts()
        print(f"  {eps:>3} {len(sizes):>9} {sizes.max():>8} {(sizes==1).sum():>11}")

    EPS = 2  # near-clonal: profiles differing by <=2 calls treated as one group
    gid = build_groups(X, eps=EPS)
    split = grouped_split(gid, X.index)

    # leakage check: no gene profile string appears in more than one split
    prof = X.astype(str).agg("".join, axis=1)
    leak = (pd.DataFrame({"p": prof, "s": split})
            .groupby("p")["s"].nunique())
    n_leaky = int((leak > 1).sum())

    print(f"\nChosen eps={EPS}: {len(np.unique(gid))} groups")
    print("Split sizes (isolates):")
    print(split.value_counts().to_string())
    print(f"\nGene profiles spanning >1 split (leakage): {n_leaky}  <-- must be 0")

    # per-drug label balance within each split
    print("\nResistant fraction by drug x split (n):")
    for d in DRUGS:
        row = []
        for s in ("train", "calibration", "test"):
            m = (split == s) & Y[d].notna()
            yy = Y.loc[m, d]
            row.append(f"{s}={yy.mean():.2f}(n={len(yy)})" if len(yy) else f"{s}=NA")
        print(f"  {d:<11} " + "  ".join(row))

    pd.to_pickle(X, os.path.join(PROC, "X.pkl"))
    pd.to_pickle(Y, os.path.join(PROC, "Y.pkl"))
    pd.to_pickle(pd.Series(gid, index=X.index, name="group"),
                 os.path.join(PROC, "groups.pkl"))
    pd.to_pickle(split, os.path.join(PROC, "split.pkl"))
    json.dump({"drugs": DRUGS, "n_features": int(X.shape[1]), "eps": EPS,
               "known_carbapenemases": KNOWN_CARBAPENEMASES},
              open(os.path.join(PROC, "meta.json"), "w"), indent=2)
    print("\nsaved -> data/processed/{X,Y,groups,split}.pkl, meta.json")
