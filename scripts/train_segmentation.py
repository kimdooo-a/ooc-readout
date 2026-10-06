"""Train the 3-class nucleus U-Net on BBBC039 (official training split; validation split for model selection).

    python scripts/train_segmentation.py --epochs 60 --out models/unet3_bbbc039.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.data import bbbc039_split
from ooc_readout.seg import (
    UNet3,
    match_f1,
    normalize,
    segment,
    targets_from_instances,
)

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--iters", type=int, default=100, help="iterations per epoch")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="models/unet3_bbbc039.pt")
a = ap.parse_args()

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
train = bbbc039_split("training")
val = bbbc039_split("validation")
X = [normalize(i) for i, _, _ in train]
Y = [targets_from_instances(l) for _, l, _ in train]
print(f"train {len(X)} images, val {len(val)} images, device {dev}")


def batch():
    xs, ys = [], []
    for _ in range(a.batch):
        k = rng.integers(len(X))
        x, y = X[k], Y[k]
        i = rng.integers(0, x.shape[0] - a.crop + 1)
        j = rng.integers(0, x.shape[1] - a.crop + 1)
        x, y = x[i:i + a.crop, j:j + a.crop], y[i:i + a.crop, j:j + a.crop]
        r = rng.integers(4)
        x, y = np.rot90(x, r), np.rot90(y, r)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        # intensity / scale augmentation: gamma, gain, noise (helps transfer to other microscopes)
        x = np.clip(x, 0, None) ** rng.uniform(0.7, 1.4) * rng.uniform(0.7, 1.3)
        x = x + rng.normal(0, rng.uniform(0, 0.03), x.shape)
        xs.append(x.copy()); ys.append(y.copy())
    return (torch.from_numpy(np.stack(xs)[:, None].astype(np.float32)).to(dev),
            torch.from_numpy(np.stack(ys)).to(dev))


model = UNet3().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * a.iters)
w = torch.tensor([1.0, 1.0, 10.0], device=dev)  # boundary class up-weighted (Caicedo et al. 2019)
best, log = -1, []
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for ep in range(a.epochs):
    model.train()
    tot = 0
    for _ in range(a.iters):
        x, y = batch()
        loss = F.cross_entropy(model(x), y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    if ep % 5 == 4 or ep == a.epochs - 1:
        f1s = [match_f1(l, segment(model, img, device=dev), 0.5)[0] for img, l, _ in val]
        f1 = float(np.mean(f1s))
        log.append({"epoch": ep + 1, "loss": tot / a.iters, "val_f1@0.5": f1, "sec": time.time() - t0})
        print(log[-1], flush=True)
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), a.out)
json.dump({"best_val_f1@0.5": best, "log": log, "args": vars(a)}, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
print("best val F1@0.5", best)
