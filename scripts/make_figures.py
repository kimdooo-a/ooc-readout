"""Dose-response figures and the results table used in the technical report.

    python scripts/make_figures.py
Reads results/<plate>/{wells.csv,curves.json}; writes results/fig_dose_response.png and results/summary.md
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.scoring import hill

R = Path("results")
NAMES = {"log_nc": "classic N/C ratio", "caes": "CAES (label-free)", "probe": "few-shot probe"}
COLS = {"log_nc": "#4C72B0", "caes": "#DD8452", "probe": "#55A868"}
panels = []
for plate in ["BBBC013", "BBBC014"]:
    W = pd.read_csv(R / plate / "wells.csv")
    C = json.load(open(R / plate / "curves.json"))
    for g in sorted(C["groups"]):
        panels.append((plate, g, W, C))

fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.0))
lines = ["| Plate | Group | Readout | Z' (neg vs top dose) | Z' (neg vs pos ctrl) | V-factor | Hill R² | EC50 [95% CI] | Spearman ρ (log dose) |",
         "|---|---|---|---|---|---|---|---|---|"]
for ax, (plate, g, W, C) in zip(axes, panels):
    Wg = W[(W.group == g) & (W.dose > 0)]
    if plate == "BBBC013":
        Wg = Wg[Wg.role == "sample"]
    neg = W[(W.role == "neg") & ((W.group == g) | (plate == "BBBC013"))]
    unit = Wg.unit.iloc[0]
    for r in NAMES:
        st = C["groups"][g][r]
        lo, hi = neg[r].mean(), (st.get("top") if st.get("top") is not None else Wg[r].max())
        norm = lambda v: (np.asarray(v) - lo) / (hi - lo) if hi != lo else v
        x = np.log10(Wg.dose.values)
        ax.scatter(x, norm(Wg[r].values), s=10, color=COLS[r], alpha=0.6)
        if "logec50" in st:
            xx = np.linspace(x.min(), x.max(), 200)
            ax.plot(xx, norm(hill(xx, st["bottom"], st["top"], st["logec50"], st["slope"])), color=COLS[r],
                    label=f"{NAMES[r]}  Z'={st['zprime_neg_vs_topdose']:.2f}")
            ci = st.get("ec50_ci95", [np.nan, np.nan])
            lines.append(f"| {plate} | {g} | {NAMES[r]} | {st['zprime_neg_vs_topdose']:.2f} | {st['zprime_neg_vs_posctrl']:.2f} | "
                         f"{st['vfactor']:.2f} | {st['r2']:.3f} | {st['ec50']:.3g} [{ci[0]:.3g}–{ci[1]:.3g}] {unit} | {st['spearman_logdose']:.2f} |")
    ax.set_title(f"{plate} · {g}", fontsize=11)
    ax.set_xlabel(f"log10 dose ({unit})")
    ax.set_ylabel("response (0 = negative control, 1 = fitted top)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig(R / "fig_dose_response.png", dpi=130)
(R / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
