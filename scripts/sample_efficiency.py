"""Organ-on-a-chip regime simulation: how many chips (replicate wells) and cells per chip are needed?

Organ-on-a-chip experiments yield far fewer replicates and fields than a 96-well screen. We emulate that by
repeatedly sub-sampling the BBBC013 plate:
  * m "chips" per condition (m in 2..4) and k cells per chip (k in 5..all);
  * the models (CAES, probe) are RE-FITTED on the sub-sampled control chips only (m neg + m pos chips),
    and evaluated on m DIFFERENT control chips, so no well is both training and evaluation data;
  * per readout we record Z' (held-out neg chips vs top-dose chips), and the EC50 error
    |log10(EC50_sub / EC50_full)| from a 4-parameter Hill fit on m chips per dose.

    python scripts/sample_efficiency.py --reps 60
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.scoring import CAES, fit_hill, zprime

ap = argparse.ArgumentParser()
ap.add_argument("--plate", default="results/BBBC013")
ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved summary")
a = ap.parse_args()
P = Path(a.plate)
cells = pd.read_parquet(P / "cells.parquet").reset_index(drop=True)
E = np.load(P / "embeddings.npy").astype(np.float32)
full = json.load(open(P / "curves.json"))
rng = np.random.default_rng(a.seed)
neg_w = sorted(cells[cells.role == "neg"].well.unique())
pos_w = sorted(cells[cells.role == "pos"].well.unique())
idx_by_well = {w: np.flatnonzero(cells.well.values == w) for w in cells.well.unique()}
KS = [5, 10, 20, 40, 80, 10_000]
MS = [2, 3, 4]
ALL_X = int(cells.groupby("well").size().median())  # x position used for "all cells" in the figure
rows = []


def take(well, k):
    ix = idx_by_well[well]
    return ix if len(ix) <= k else rng.choice(ix, k, replace=False)


for m in (MS if not a.plot_only else []):
    for k in KS:
        for rep in range(a.reps):
            nw = rng.permutation(neg_w)
            pw = rng.permutation(pos_w)
            tr_neg, ev_neg, tr_pos = nw[:m], nw[m:2 * m], pw[:m]
            tr_n = np.concatenate([take(w, k) for w in tr_neg])
            tr_p = np.concatenate([take(w, k) for w in tr_pos])
            caes = CAES(n_comp=32).fit(E[tr_n])
            sc = StandardScaler().fit(E[np.r_[tr_n, tr_p]])
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(E[np.r_[tr_n, tr_p]]),
                                                               np.r_[np.zeros(len(tr_n)), np.ones(len(tr_p))])

            def well_score(w, kk):
                ix = take(w, kk)
                return {"log_nc": cells.log_nc.values[ix].mean(),
                        "caes": np.log(caes.distance(E[ix])).mean(),
                        "probe": clf.predict_proba(sc.transform(E[ix]))[:, 1].mean()}

            ev_n = [well_score(w, k) for w in ev_neg]
            for g in ["Wortmannin", "LY294002"]:
                S = cells[(cells.group == g) & (cells.role == "sample")]
                doses = sorted(S.dose.unique())
                per_dose = {}
                for d in doses:
                    ws = sorted(S[S.dose == d].well.unique())
                    per_dose[d] = [well_score(w, k) for w in rng.permutation(ws)[:m]]
                top = per_dose[doses[-1]]
                for r in ["log_nc", "caes", "probe"]:
                    zp = zprime([s[r] for s in ev_n], [s[r] for s in top])
                    x = np.log10([d for d in doses if d > 0 for _ in per_dose[d]])
                    y = np.array([s[r] for d in doses if d > 0 for s in per_dose[d]])
                    f = fit_hill(x, y)
                    ref = full["groups"][g][r].get("logec50")
                    err = abs(f["logec50"] - ref) if (f and ref is not None) else np.nan
                    rows.append({"m_chips": m, "k_cells": k, "rep": rep, "group": g, "readout": r,
                                 "zprime": zp, "logec50_err": err})
    print("m", m, "done", flush=True)

if a.plot_only:
    summ = pd.read_csv(P / "sample_efficiency_summary.csv")
else:
  df = pd.DataFrame(rows)
  df.to_csv(P / "sample_efficiency_raw.csv", index=False)
  summ = (df.groupby(["readout", "group", "m_chips", "k_cells"])
          .agg(zprime_median=("zprime", "median"),
               p_zprime_gt_0_5=("zprime", lambda z: float(np.mean(np.asarray(z) > 0.5))),
               logec50_err_median=("logec50_err", "median"))
          .reset_index())
  summ.to_csv(P / "sample_efficiency_summary.csv", index=False)
  print(summ.to_string())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

names = {"log_nc": "classic N/C ratio", "caes": "CAES (label-free)", "probe": "few-shot probe"}
cols = {"log_nc": "#4C72B0", "caes": "#DD8452", "probe": "#55A868"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for r in names:
    for m, ls in zip(MS, [":", "--", "-"]):
        s = summ[(summ.readout == r) & (summ.m_chips == m)].groupby("k_cells").mean(numeric_only=True)
        xs = [min(k, ALL_X) for k in s.index]
        ax[0].plot(xs, s.p_zprime_gt_0_5, ls, color=cols[r], marker="o", ms=3,
                   label=f"{names[r]}, {m} chips" if m == 4 or r == "probe" else None)
        ax[1].plot(xs, s.logec50_err_median, ls, color=cols[r], marker="o", ms=3)
for i, t in enumerate(["P(Z' > 0.5)  (held-out control chips)", "median |log10 EC50 - full-data EC50|"]):
    ax[i].set_xscale("log"); ax[i].set_xlabel(f"cells per chip (rightmost = all, median {ALL_X})"); ax[i].set_title(t)
    ax[i].grid(alpha=0.3)
ax[0].legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(P / "fig_sample_efficiency.png", dpi=130)
