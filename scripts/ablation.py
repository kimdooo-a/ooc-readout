"""Ablations of the embedding readouts (run after run_assay.py for both plates).

  * CAES: PCA dimension (8/16/32/64) x token (CLS / mean-patch / both)
  * probe: regularisation C (0.01/0.1/1) and token choice
Each configuration is scored with the same leave-one-control-well-out protocol as the main pipeline;
the metric is Z' (negative control wells vs top-dose wells) per curve.

    python scripts/ablation.py   -> results/ablation.csv, results/ablation.md
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.scoring import zprime

R = Path("results")
TOK = {"cls+patch": slice(0, 768), "cls": slice(0, 384), "patch": slice(384, 768)}


def curves(plate, cells):
    """Yield (name, mask over cells, neg wells, top wells)."""
    if plate == "BBBC013":
        neg = sorted(cells[cells.role == "neg"].well.unique())
        for g in ["Wortmannin", "LY294002"]:
            S = cells[(cells.group == g) & (cells.role == "sample")]
            top = sorted(S[S.dose == S.dose.max()].well.unique())
            yield f"{plate}/{g}", np.ones(len(cells), bool), neg, top
    else:
        for g in ["A549", "MCF7"]:
            m = (cells.group == g).values
            C = cells[m]
            yield (f"{plate}/{g}", m, sorted(C[C.role == "neg"].well.unique()),
                   sorted(C[C.role == "pos"].well.unique()))


def lowo_scores(cells, E, neg, pos, method, **kw):
    """Per-well mean score; control wells scored by a model fitted without them."""
    wells = cells.well.values
    out = {}
    targets = sorted(set(cells.well))
    for hold in [None] + sorted(set(neg) | set(pos)):
        tr_n = np.isin(wells, [w for w in neg if w != hold])
        tr_p = np.isin(wells, [w for w in pos if w != hold])
        if method == "caes":
            sc = StandardScaler().fit(E[tr_n])
            p = PCA(kw["k"], random_state=0).fit(sc.transform(E[tr_n]))
            cov = LedoitWolf().fit(p.transform(sc.transform(E[tr_n])))
            f = lambda X: np.log(cov.mahalanobis(p.transform(sc.transform(X))))
        else:
            X = np.r_[E[tr_n], E[tr_p]]
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(C=kw["C"], max_iter=3000).fit(sc.transform(X), np.r_[np.zeros(tr_n.sum()), np.ones(tr_p.sum())])
            f = lambda X: clf.predict_proba(sc.transform(X))[:, 1]
        todo = [hold] if hold else [w for w in targets if w not in set(neg) | set(pos)]
        for w in todo:
            out[w] = float(f(E[wells == w]).mean())
    return out


rows = []
for plate in ["BBBC013", "BBBC014"]:
    cells = pd.read_parquet(R / plate / "cells.parquet").reset_index(drop=True)
    E_all = np.load(R / plate / "embeddings.npy").astype(np.float32)
    for name, m, neg, top in curves(plate, cells):
        C, E0 = cells[m].reset_index(drop=True), E_all[m]
        pos = sorted(C[C.role == "pos"].well.unique())
        for tok, sl in TOK.items():
            E = E0[:, sl]
            for k in [8, 16, 32, 64]:
                s = lowo_scores(C, E, neg, pos, "caes", k=k)
                rows.append({"curve": name, "readout": "CAES", "tokens": tok, "param": f"PCA k={k}",
                             "zprime": zprime([s[w] for w in neg], [s[w] for w in top])})
            for c in [0.01, 0.1, 1.0]:
                s = lowo_scores(C, E, neg, pos, "probe", C=c)
                rows.append({"curve": name, "readout": "probe", "tokens": tok, "param": f"C={c}",
                             "zprime": zprime([s[w] for w in neg], [s[w] for w in top])})
        print(name, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(R / "ablation.csv", index=False)
piv = df.pivot_table(index=["readout", "tokens", "param"], columns="curve", values="zprime").round(2)
piv["mean"] = piv.mean(1).round(2)
(R / "ablation.md").write_text(piv.to_markdown(), encoding="utf-8")
print(piv.to_string())
