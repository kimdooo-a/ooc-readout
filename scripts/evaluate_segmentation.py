"""Evaluate the U-Net and the Otsu+watershed baseline on the held-out BBBC039 TEST split.

    python scripts/evaluate_segmentation.py --model models/unet3_bbbc039.pt
Writes results/segmentation_test.json and results/fig_seg_examples.png
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.data import bbbc039_split
from ooc_readout.seg import (
    UNet3,
    match_f1,
    normalize,
    otsu_watershed,
    segment,
)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/unet3_bbbc039.pt")
ap.add_argument("--out", default="results")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = UNet3().to(dev)
m.load_state_dict(torch.load(a.model, map_location=dev))
test = bbbc039_split("test")
thrs = [0.5, 0.6, 0.7, 0.8, 0.9]
res = {"unet": {t: [] for t in thrs}, "otsu_watershed": {t: [] for t in thrs}}
counts = {"unet": [0, 0, 0], "otsu_watershed": [0, 0, 0]}
preds = []
for img, gt, name in test:
    pu = segment(m, img, device=dev)
    pb = otsu_watershed(img)
    preds.append((img, gt, pu, pb, name))
    for key, pr in (("unet", pu), ("otsu_watershed", pb)):
        for t in thrs:
            f1, tp, fp, fn = match_f1(gt, pr, t)
            res[key][t].append(f1)
            if t == 0.5:
                counts[key][0] += tp; counts[key][1] += fp; counts[key][2] += fn
summary = {k: {f"F1@{t}": float(np.mean(v[t])) for t in thrs} for k, v in res.items()}
for k in summary:
    tp, fp, fn = counts[k]
    summary[k]["pooled_TP_FP_FN@0.5"] = [tp, fp, fn]
    summary[k]["mean_F1_0.5-0.9"] = float(np.mean([summary[k][f"F1@{t}"] for t in thrs]))
summary["n_test_images"] = len(test)
summary["n_gt_nuclei"] = int(sum(g.max() for _, g, _ in test))
Path(a.out).mkdir(exist_ok=True)
json.dump(summary, open(Path(a.out) / "segmentation_test.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# figure: 3 test images - raw | ground truth | baseline | U-Net
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

pick = sorted(range(len(preds)), key=lambda i: -preds[i][1].max())[:1] + [len(preds) // 2, len(preds) - 1]
fig, ax = plt.subplots(len(pick), 4, figsize=(13, 3.2 * len(pick)))
for r, i in enumerate(pick):
    img, gt, pu, pb, name = preds[i]
    base = np.clip(normalize(img), 0, 1)
    for c, (lab, title) in enumerate([(None, "input (Hoechst)"), (gt, "ground truth"), (pb, "Otsu + watershed"), (pu, "U-Net (ours)")]):
        rgb = np.dstack([base] * 3)
        if lab is not None:
            b = find_boundaries(lab)
            rgb[b] = [1, 0.25, 0.1] if c != 3 else [0.1, 1, 0.4]
        ax[r, c].imshow(rgb); ax[r, c].axis("off")
        if r == 0:
            ax[r, c].set_title(title, fontsize=11)
plt.tight_layout()
plt.savefig(Path(a.out) / "fig_seg_examples.png", dpi=130)
