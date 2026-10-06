"""OoC-Readout demo: one field of view in, a per-cell and per-field drug-response readout out.

    # any reporter + nucleus image pair (e.g. one chip field of view)
    python scripts/demo.py --reporter path/to/reporter.png --nucleus path/to/nucleus.png --models results/BBBC013/readout_models.joblib
    # or pick a well of the public benchmark plates
    python scripts/demo.py --plate BBBC013 --well A08

Writes demo_out/<name>_overlay.png and demo_out/<name>.json and prints a summary.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.cells import crops, measure
from ooc_readout.data import Well, bbbc013_wells, bbbc014_wells, load_pair
from ooc_readout.embed import Embedder
from ooc_readout.seg import UNet3, normalize, segment

ap = argparse.ArgumentParser()
ap.add_argument("--reporter")
ap.add_argument("--nucleus")
ap.add_argument("--plate", choices=["BBBC013", "BBBC014"])
ap.add_argument("--well")
ap.add_argument("--models", help="readout_models.joblib (default: results/<plate>/readout_models.joblib)")
ap.add_argument("--group", help="model group inside the bundle (BBBC014: MCF7 or A549)")
ap.add_argument("--seg", default="models/unet3_bbbc039.pt")
ap.add_argument("--out", default="demo_out")
a = ap.parse_args()
t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"

if a.plate:
    wells = bbbc013_wells() if a.plate == "BBBC013" else bbbc014_wells()
    w = next(x for x in wells if x.well == a.well)
    name = f"{a.plate}_{a.well}"
    a.models = a.models or f"results/{a.plate}/readout_models.joblib"
    a.group = a.group or ("plate" if a.plate == "BBBC013" else w.group)
    print(f"input  : {a.plate} well {w.well} | {w.group} {w.dose:g} {w.unit} ({w.role})")
else:
    w = Well("custom", "-", "-", 0, "-", float("nan"), "-", "sample", a.reporter, a.nucleus)
    name = Path(a.reporter).stem
    a.group = a.group or "plate"
    print(f"input  : {a.reporter} + {a.nucleus}")
rep, nuc = load_pair(w)
B = joblib.load(a.models)
G = B["groups"][a.group]

seg = UNet3().to(dev)
seg.load_state_dict(torch.load(a.seg, map_location=dev))
lab = segment(seg, nuc, scale=B["scale"], device=dev)
cells = measure(rep, lab)
print(f"step 1 : nuclei segmented        -> {len(cells)} cells   ({time.time() - t0:.1f}s)")
C = crops(rep, nuc, cells, B["crop"])
E = Embedder(device=dev)(C)
print(f"step 2 : DINOv2 embeddings        -> {E.shape}   ({time.time() - t0:.1f}s)")
caes = np.log(G["caes"].distance(E))
resp = G["caes"].responder(E)
prob = G["probe"].predict_proba(G["probe_scaler"].transform(E))[:, 1]
nc = np.array([c["log_nc"] for c in cells])
frac_nc = (nc.mean() - G["log_nc_neg_mean"]) / (G["log_nc_pos_mean"] - G["log_nc_neg_mean"])
summary = {
    "input": name, "n_cells": len(cells),
    "classic_log_nc_mean": float(nc.mean()),
    "classic_pct_of_positive_control": float(100 * frac_nc),
    "caes_mean_log_mahalanobis": float(caes.mean()),
    "caes_responder_fraction": float(resp.mean()),
    "probe_mean_probability_responding": float(prob.mean()),
    "seconds": time.time() - t0,
}
print("step 3 : readouts")
for k, v in summary.items():
    if k not in ("input",):
        print(f"         {k:36s} {v:.3f}" if isinstance(v, float) else f"         {k:36s} {v}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

R, N = np.clip(normalize(rep), 0, 1), np.clip(normalize(nuc), 0, 1)
rgb = np.dstack([N * 0.35, R, N * 0.9])
b = find_boundaries(lab) * lab
pr = {c["label"]: p for c, p in zip(cells, prob)}
cmap = plt.get_cmap("coolwarm")
for l, p in pr.items():
    rgb[b == l] = cmap(p)[:3]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
ax[0].imshow(rgb); ax[0].axis("off")
ax[0].set_title(f"{name}: nuclei outlined by P(responding) (blue=0, red=1)", fontsize=10)
ax[1].hist(prob, bins=20, range=(0, 1), color="#55A868")
ax[1].set_xlabel("few-shot probe P(responding) per cell"); ax[1].set_ylabel("cells")
ax[1].set_title(f"n={len(cells)} | probe {prob.mean():.2f} | CAES resp. {resp.mean():.0%} | N/C {100 * frac_nc:.0f}% of pos", fontsize=10)
plt.tight_layout()
Path(a.out).mkdir(exist_ok=True)
plt.savefig(Path(a.out) / f"{name}_overlay.png", dpi=110)
json.dump(summary, open(Path(a.out) / f"{name}.json", "w"), indent=1)
print(f"output : {Path(a.out) / (name + '_overlay.png')}  ({time.time() - t0:.1f}s total)")
