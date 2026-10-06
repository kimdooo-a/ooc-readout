"""Readouts, assay-quality statistics and dose-response fitting.

Three single-cell readouts are compared:
  * `classic`  - log nuclear/cytoplasmic reporter ratio (hand-engineered, assay-specific).
  * `caes`     - Control-Anchored Embedding Score (ours): log Mahalanobis distance of a cell's frozen-DINOv2
                 embedding from the negative-control distribution (PCA space fitted on negative-control
                 cells only, Ledoit-Wolf shrinkage); the well score is the mean over cells. `caes_resp`
                 flags cells beyond the 95th percentile of held-out control cells (used for overlays).
                 Needs NO positive control and NO labels - only knowledge of which wells are untreated.
  * `probe`    - few-shot linear probe on the same embeddings, trained on negative vs positive control
                 cells (the usual "supervised" alternative; needs a positive control).
Control wells are always scored by a model that did not see them (leave-one-well-out cross-fitting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ CAES
class CAES:
    def __init__(self, n_comp: int = 32, q: float = 0.95):
        self.n_comp, self.q = n_comp, q

    def fit(self, E_neg: np.ndarray, E_cal: np.ndarray | None = None):
        k = int(min(self.n_comp, max(2, len(E_neg) // 4), E_neg.shape[1]))
        self.sc = StandardScaler().fit(E_neg)
        self.pca = PCA(k, random_state=0).fit(self.sc.transform(E_neg))
        Z = self.pca.transform(self.sc.transform(E_neg))
        self.cov = LedoitWolf().fit(Z)
        cal = E_neg if E_cal is None else E_cal
        self.tau = float(np.quantile(self.distance(cal), self.q))
        return self

    def distance(self, E: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(self.sc.transform(E))
        return self.cov.mahalanobis(Z)

    def responder(self, E: np.ndarray) -> np.ndarray:
        return (self.distance(E) > self.tau).astype(np.float32)


def _fit_caes_lowo(E_neg_by_well: dict, n_comp: int):
    """Fit on all-but-one control well, calibrate tau on the held-out well's cells is NOT done
    (that would bias the held-out score); tau is calibrated on the training control cells via an
    inner split: half of the control wells fit the space, the other half set tau."""
    wells = list(E_neg_by_well)
    rng = np.random.default_rng(0)
    rng.shuffle(wells)
    half = max(1, len(wells) // 2)
    fit_w, cal_w = wells[:half], wells[half:] or wells[:half]
    m = CAES(n_comp).fit(np.concatenate([E_neg_by_well[w] for w in fit_w]),
                         np.concatenate([E_neg_by_well[w] for w in cal_w]))
    return m


def score_cells(cells: pd.DataFrame, E: np.ndarray, n_comp: int = 32) -> pd.DataFrame:
    """Add `caes` and `probe` per-cell scores. `cells` needs columns well, role (neg/pos/sample)."""
    cells = cells.copy()
    cells["caes"] = np.nan
    cells["probe"] = np.nan
    neg_w = sorted(cells.loc[cells.role == "neg", "well"].unique())
    pos_w = sorted(cells.loc[cells.role == "pos", "well"].unique())
    by = {w: E[(cells.well == w).values] for w in cells.well.unique()}

    # CAES: model per held-out control well, plus one model (all controls) for every other well
    full = _fit_caes_lowo({w: by[w] for w in neg_w}, n_comp)
    for w in cells.well.unique():
        if w in neg_w:
            m = _fit_caes_lowo({v: by[v] for v in neg_w if v != w}, n_comp)
        else:
            m = full
        cells.loc[cells.well == w, "caes"] = np.log(m.distance(by[w]))
        cells.loc[cells.well == w, "caes_resp"] = m.responder(by[w])

    # probe: logistic regression neg vs pos cells (leave-one-control-well-out)
    def probe(excl):
        Xn = np.concatenate([by[v] for v in neg_w if v != excl])
        Xp = np.concatenate([by[v] for v in pos_w if v != excl])
        X = np.concatenate([Xn, Xp])
        y = np.r_[np.zeros(len(Xn)), np.ones(len(Xp))]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X), y)
        return lambda Z: clf.predict_proba(sc.transform(Z))[:, 1]
    if pos_w:
        pf = probe(None)
        for w in cells.well.unique():
            f = probe(w) if (w in neg_w or w in pos_w) else pf
            cells.loc[cells.well == w, "probe"] = f(by[w])
    return cells


# ------------------------------------------------------------------ statistics
def zprime(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = abs(a.mean() - b.mean())
    return float(1 - 3 * (a.std(ddof=1) + b.std(ddof=1)) / d) if d > 0 else float("-inf")


def hill(x, bottom, top, logec50, slope):
    return bottom + (top - bottom) / (1 + 10 ** ((logec50 - x) * slope))


def fit_hill(logdose: np.ndarray, y: np.ndarray):
    lo, hi = np.percentile(y, [5, 95])
    p0 = [lo, hi, np.median(logdose), 1.0]
    bounds = ([-np.inf, -np.inf, logdose.min() - 2, 0.1], [np.inf, np.inf, logdose.max() + 2, 10])
    try:
        p, _ = curve_fit(hill, logdose, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    pred = hill(logdose, *p)
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return {"bottom": p[0], "top": p[1], "logec50": p[2], "slope": p[3], "r2": float(r2)}


def vfactor(logdose, y, fit) -> float:
    df = pd.DataFrame({"x": logdose, "y": y})
    sd = df.groupby("x").y.std(ddof=1).mean()
    rng = abs(fit["top"] - fit["bottom"])
    return float(1 - 6 * sd / rng) if rng > 0 else float("-inf")


def curve_stats(wells: pd.DataFrame, col: str, n_boot: int = 300, seed: int = 0) -> dict:
    """wells: one drug/cell-line group with columns dose, role, <col>. Positive-dose wells are fitted."""
    rng = np.random.default_rng(seed)
    s = wells[wells.dose > 0]
    x, y = np.log10(s.dose.values), s[col].values
    fit = fit_hill(x, y)
    out = {"readout": col, "n_wells": len(s)}
    if fit is None:
        return out
    out.update(fit)
    out["ec50"] = float(10 ** fit["logec50"])
    out["vfactor"] = vfactor(x, y, fit)
    sp = pd.Series(y).corr(pd.Series(x), method="spearman")
    out["spearman_logdose"] = float(sp)
    boots = []
    groups = [g for _, g in s.groupby("dose")]
    for _ in range(n_boot):
        b = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31))) for g in groups])
        f = fit_hill(np.log10(b.dose.values), b[col].values)
        if f is not None:
            boots.append(f["logec50"])
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["ec50_ci95"] = [float(10 ** lo), float(10 ** hi)]
        out["logec50_ci_width"] = float(hi - lo)
    return out


def well_table(cells: pd.DataFrame, cols=("log_nc", "caes", "probe")) -> pd.DataFrame:
    agg = {c: "mean" for c in cols if c in cells}
    agg["label"] = "count"
    g = cells.groupby(["plate", "well", "group", "dose", "unit", "role"], as_index=False).agg(agg)
    return g.rename(columns={"label": "n_cells"})
