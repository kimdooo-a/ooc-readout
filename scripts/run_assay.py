"""End-to-end readout of one plate: images -> nuclei -> single cells -> embeddings -> scores -> curves.

    python scripts/run_assay.py --plate BBBC013
    python scripts/run_assay.py --plate BBBC014
Outputs (results/<plate>/): cells.parquet, embeddings.npy, wells.csv, curves.json, overlays/*.png
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.cells import crops, measure
from ooc_readout.data import bbbc013_wells, bbbc014_wells, load_pair
from ooc_readout.embed import Embedder
from ooc_readout.scoring import (
    curve_stats,
    score_cells,
    well_table,
    zprime,
)
from ooc_readout.seg import UNet3, estimate_scale, normalize, segment

TRAIN_MEDIAN_NUCLEUS_AREA = 627.0  # px, BBBC039 training masks (computed by scripts/train_segmentation.py data)

ap = argparse.ArgumentParser()
ap.add_argument("--plate", choices=["BBBC013", "BBBC014"], required=True)
ap.add_argument("--model", default="models/unet3_bbbc039.pt")
ap.add_argument("--out", default="results")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(a.out) / a.plate
(out / "overlays").mkdir(parents=True, exist_ok=True)
wells = bbbc013_wells() if a.plate == "BBBC013" else bbbc014_wells()
seg = UNet3().to(dev)
seg.load_state_dict(torch.load(a.model, map_location=dev))
t0 = time.time()

# 1) magnification calibration from negative-control nuclei (no labels needed)
negs = [w for w in wells if w.role == "neg"]
scale = float(np.median([estimate_scale(load_pair(w)[1], TRAIN_MEDIAN_NUCLEUS_AREA) for w in negs]))
print(f"[{a.plate}] rescale factor to training magnification: {scale:.2f}")

# 2) segmentation + single-cell measurement
rows, crop_list = [], []
areas = []
labs = {}
for w in wells:
    rep, nuc = load_pair(w)
    lab = segment(seg, nuc, scale=scale, device=dev)
    labs[w.well] = lab
    cs = measure(rep, lab)
    areas += [c["area"] for c in cs]
    for c in cs:
        c.update(plate=w.plate, well=w.well, group=w.group, dose=w.dose, unit=w.unit, role=w.role)
    rows += cs
cells = pd.DataFrame(rows)
crop = int(np.clip(round(3.0 * 2 * np.sqrt(np.median(areas) / np.pi)), 32, 128))  # ~3 nucleus diameters
print(f"[{a.plate}] {len(cells)} cells in {len(wells)} wells, crop {crop}px, {time.time() - t0:.0f}s")
for w in wells:
    rep, nuc = load_pair(w)
    sub = cells[cells.well == w.well].to_dict("records")
    crop_list.append(crops(rep, nuc, sub, crop))
C = np.concatenate(crop_list)

# 3) frozen foundation-model embeddings (DINOv2 ViT-S/14, no fine-tuning)
emb = Embedder(device=dev)
E = emb(C)
np.save(out / "embeddings.npy", E.astype(np.float16))
print(f"[{a.plate}] embeddings {E.shape}, {time.time() - t0:.0f}s")

# 4) scoring - BBBC013: one plate-wide control set; BBBC014: per cell line (biologically different)
if a.plate == "BBBC013":
    cells = score_cells(cells.reset_index(drop=True), E)
else:
    parts = []
    for g in cells.group.unique():
        idx = (cells.group == g).values
        parts.append(score_cells(cells[idx].reset_index(drop=True), E[idx]))
    cells = pd.concat(parts, ignore_index=True)
cells.to_parquet(out / "cells.parquet")

# 4b) readout models fitted on ALL control wells, saved for scripts/demo.py (inference on new images)
import joblib  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from ooc_readout.scoring import CAES  # noqa: E402
bundle = {"scale": scale, "crop": crop, "groups": {}}
for g in (["plate"] if a.plate == "BBBC013" else sorted(cells.group.unique())):
    sel = np.ones(len(cells), bool) if g == "plate" else (cells.group == g).values
    En, Ep = E[sel & (cells.role == "neg").values], E[sel & (cells.role == "pos").values]
    sc = StandardScaler().fit(np.r_[En, Ep])
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(np.r_[En, Ep]), np.r_[np.zeros(len(En)), np.ones(len(Ep))])
    nc_neg = cells.log_nc.values[sel & (cells.role == "neg").values]
    nc_pos = cells.log_nc.values[sel & (cells.role == "pos").values]
    bundle["groups"][g] = {"caes": CAES(32).fit(En), "probe_scaler": sc, "probe": clf,
                           "log_nc_neg_mean": float(nc_neg.mean()), "log_nc_pos_mean": float(nc_pos.mean())}
joblib.dump(bundle, out / "readout_models.joblib")
W = well_table(cells)
W.to_csv(out / "wells.csv", index=False)

# 5) assay statistics per readout
READ = ["log_nc", "caes", "probe"]
res = {"plate": a.plate, "scale": scale, "crop_px": crop, "n_cells": len(cells), "groups": {}}
for g in sorted(W.group.unique()):
    Wg = W[W.group == g]
    if a.plate == "BBBC013":
        neg = W[W.role == "neg"]
        top = Wg[Wg.dose == Wg[Wg.role == "sample"].dose.max()]
        pos = W[W.role == "pos"]
        fit_set = pd.concat([Wg[Wg.role == "sample"]])
    else:
        neg, top, pos = Wg[Wg.role == "neg"], Wg[Wg.role == "pos"], Wg[Wg.role == "pos"]
        fit_set = Wg
    gr = {}
    for r in READ:
        st = curve_stats(fit_set, r)
        st["zprime_neg_vs_topdose"] = zprime(neg[r], top[r])
        st["zprime_neg_vs_posctrl"] = zprime(neg[r], pos[r])
        gr[r] = st
    res["groups"][g] = gr
res["sec"] = time.time() - t0
json.dump(res, open(out / "curves.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:3000])

# 6) overlays for neg / mid / top dose of each group (figures + demo video)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

for g in sorted(W.group.unique()):
    Wg = W[(W.group == g) & (W.role != "pos")].sort_values("dose")
    doses = sorted(Wg.dose.unique())
    pick = [doses[0], doses[len(doses) // 2], doses[-1]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    for i, d in enumerate(pick):
        wname = Wg[Wg.dose == d].well.iloc[0]
        wo = next(w for w in wells if w.well == wname)
        rep, nuc = load_pair(wo)
        lab = labs[wname]
        R = np.clip(normalize(rep), 0, 1)
        Nn = np.clip(normalize(nuc), 0, 1)
        rgb = np.dstack([Nn * 0.35, R, Nn * 0.9])
        sub = cells[cells.well == wname]
        resp = set(sub[sub.caes_resp > 0].label)
        b = find_boundaries(lab)
        bl = lab * b
        rgb[(bl > 0) & np.isin(bl, list(resp))] = [1, 0.3, 0.1]
        rgb[(bl > 0) & ~np.isin(bl, list(resp))] = [0.9, 0.9, 0.9]
        h, w_ = rgb.shape[:2]
        s = min(h, w_, 512)
        ax[i].imshow(rgb[(h - s) // 2:(h + s) // 2, (w_ - s) // 2:(w_ + s) // 2])
        ax[i].axis("off")
        frac = sub.caes_resp.mean()
        ax[i].set_title(f"{g} {d:g} {wo.unit}  |  CAES responders {frac:.0%}  (n={len(sub)})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out / "overlays" / f"{g}_overlay.png", dpi=120)
    plt.close()
print(f"done {time.time() - t0:.0f}s")
