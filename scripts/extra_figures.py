"""Additional analyses for the report.

  1. single-cell heterogeneity: distribution of per-cell probe probability at each dose (BBBC013 Wortmannin,
     BBBC014 A549) and the fraction of cells with p > 0.5 vs dose  -> results/fig_single_cell.png, results/single_cell.json
  2. plate-layout QC heat maps of the three well readouts       -> results/fig_plate_qc.png
  3. segmentation transfer to BBBC013 / BBBC014 (no ground truth; qualitative) -> results/fig_seg_transfer.png

    python scripts/extra_figures.py
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.data import bbbc013_wells, bbbc014_wells, load_pair
from ooc_readout.seg import UNet3, normalize, segment

R = Path("results")
out = {}

# 1) single-cell heterogeneity -------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for i, (plate, g) in enumerate([("BBBC013", "Wortmannin"), ("BBBC014", "A549")]):
    c = pd.read_parquet(R / plate / "cells.parquet")
    c = c[(c.group == g) & (c.role != "pos")] if plate == "BBBC013" else c[c.group == g]
    doses = sorted(c.dose.unique())
    data = [c[c.dose == d].probe.values for d in doses]
    ax[i].violinplot(data, positions=range(len(doses)), widths=0.85, showmedians=True)
    ax[i].set_xticks(range(len(doses)))
    ax[i].set_xticklabels([f"{d:.3g}" for d in doses], rotation=60, fontsize=8)
    ax[i].set_xlabel(f"dose ({c.unit.iloc[0]})"); ax[i].set_ylabel("per-cell P(responding)")
    ax[i].set_title(f"{plate} · {g}: per-cell probe probability", fontsize=10)
    frac = [float((x > 0.5).mean()) for x in data]
    mid = [float(((x > 0.2) & (x < 0.8)).mean()) for x in data]
    out[f"{plate}/{g}"] = {"doses": [float(d) for d in doses], "frac_p_gt_0.5": frac, "frac_intermediate_0.2_0.8": mid,
                           "n_cells": [len(x) for x in data]}
    xs = np.log10([max(d, min(x for x in doses if x > 0) / 2) for d in doses])
    ax[2].plot(xs, frac, "o-", label=f"{g}: fraction p>0.5")
    ax[2].plot(xs, mid, "s--", alpha=0.7, label=f"{g}: fraction 0.2<p<0.8")
ax[2].set_xlabel("log10 dose (own units; 0 dose plotted at half the lowest dose)")
ax[2].set_title("responding fraction vs intermediate cells", fontsize=10)
ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(R / "fig_single_cell.png", dpi=125); plt.close()
json.dump(out, open(R / "single_cell.json", "w"), indent=1)

# 2) plate-layout QC ------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(15, 6.2))
for i, plate in enumerate(["BBBC013", "BBBC014"]):
    W = pd.read_csv(R / plate / "wells.csv")
    W["row"] = W.well.str[0].map({r: k for k, r in enumerate("ABCDEFGH")})
    W["col"] = W.well.str[1:].astype(int) - 1
    for j, (r, nm) in enumerate([("log_nc", "classic N/C"), ("caes", "CAES"), ("probe", "few-shot probe")]):
        M = np.full((8, 12), np.nan)
        for _, w in W.iterrows():
            M[w.row, w.col] = w[r]
        im = ax[i, j].imshow(M, cmap="viridis")
        ax[i, j].set_xticks(range(12)); ax[i, j].set_xticklabels(range(1, 13), fontsize=7)
        ax[i, j].set_yticks(range(8)); ax[i, j].set_yticklabels(list("ABCDEFGH"), fontsize=7)
        ax[i, j].set_title(f"{plate} · {nm}", fontsize=10)
        plt.colorbar(im, ax=ax[i, j], fraction=0.035)
plt.tight_layout(); plt.savefig(R / "fig_plate_qc.png", dpi=120); plt.close()

# 3) segmentation transfer -----------------------------------------------------------------
from skimage.segmentation import find_boundaries

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = UNet3().to(dev); m.load_state_dict(torch.load("models/unet3_bbbc039.pt", map_location=dev))
picks = [("BBBC013", "A01"), ("BBBC013", "A12"), ("BBBC014", "A12"), ("BBBC014", "E01")]
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
for k, (plate, wn) in enumerate(picks):
    wells = bbbc013_wells() if plate == "BBBC013" else bbbc014_wells()
    w = next(x for x in wells if x.well == wn)
    rep, nuc = load_pair(w)
    scale = json.load(open(R / plate / "curves.json"))["scale"]
    lab = segment(m, nuc, scale=scale, device=dev)
    h, wd = nuc.shape; s = 300
    y0, x0 = (h - s) // 2, (wd - s) // 2
    N = np.clip(normalize(nuc), 0, 1)[y0:y0 + s, x0:x0 + s]
    rgb = np.dstack([N] * 3)
    rgb[find_boundaries(lab[y0:y0 + s, x0:x0 + s])] = [1, 0.35, 0.1]
    ax[k].imshow(rgb); ax[k].axis("off")
    ax[k].set_title(f"{plate} {wn} ({w.group}, {w.dose:.3g} {w.unit})\nnuclear channel, U-Net outlines", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(R / "fig_seg_transfer.png", dpi=120); plt.close()
print(json.dumps(out, indent=1))
