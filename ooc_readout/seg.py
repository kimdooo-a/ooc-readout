"""Nucleus instance segmentation.

* `UNet3` - a compact U-Net predicting 3 classes (background / nucleus interior / nucleus boundary),
  the formulation of Caicedo et al. (Cytometry A, 2019) for BBBC039. Instances are recovered by
  labelling interior components and growing them back into the (interior + boundary) foreground.
* `otsu_watershed` - the classical baseline (Otsu threshold + distance-transform watershed).
* `match_f1` - instance-level F1 at an IoU threshold (one-to-one matching).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed
from skimage.transform import rescale
from torch import nn


def normalize(img: np.ndarray, lo: float = 1.0, hi: float = 99.8) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-6), 0, 1.5).astype(np.float32)


def targets_from_instances(lab: np.ndarray) -> np.ndarray:
    """0 = background, 1 = interior, 2 = boundary (2-px inner boundary)."""
    t = (lab > 0).astype(np.int64)
    b = find_boundaries(lab, mode="inner")
    b = ndi.binary_dilation(b, iterations=1) & (lab > 0)
    t[b] = 2
    return t


# ----------------------------------------------------------------- model
def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet3(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        c_in = 1
        for c in ch:
            self.downs.append(_block(c_in, c))
            c_in = c
        self.mid = _block(ch[-1], ch[-1] * 2)
        self.ups, self.dec = nn.ModuleList(), nn.ModuleList()
        c_in = ch[-1] * 2
        for c in reversed(ch):
            self.ups.append(nn.ConvTranspose2d(c_in, c, 2, stride=2))
            self.dec.append(_block(c * 2, c))
            c_in = c
        self.head = nn.Conv2d(ch[0], 3, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, s], 1))
        return self.head(x)


@torch.no_grad()
def predict_probs(model: nn.Module, img01: np.ndarray, device="cuda") -> np.ndarray:
    """Full-image inference with reflect padding to a multiple of 16. Returns (3, H, W) softmax."""
    model.eval()
    h, w = img01.shape
    ph, pw = (-h) % 16, (-w) % 16
    x = np.pad(img01, ((0, ph), (0, pw)), mode="reflect")
    t = torch.from_numpy(x)[None, None].to(device)
    p = torch.softmax(model(t), 1)
    # test-time augmentation: horizontal flip
    p2 = torch.softmax(model(torch.flip(t, [3])), 1)
    p = (p + torch.flip(p2, [3])) / 2
    return p[0, :, :h, :w].cpu().numpy()


def probs_to_instances(p: np.ndarray, min_size: int = 30) -> np.ndarray:
    interior = p[1] > 0.5
    interior = remove_small_objects(interior, min_size // 3)
    seeds = label(interior)
    fg = (p[1] + p[2]) > 0.5
    lab = watershed(-p[1], seeds, mask=fg | interior)
    lab = remove_small_objects(lab, min_size)
    return _relabel(lab)


def _relabel(lab):
    from skimage.segmentation import relabel_sequential
    return relabel_sequential(lab.astype(np.int32))[0]


def segment(model, img: np.ndarray, scale: float = 1.0, device="cuda") -> np.ndarray:
    """Segment a raw nuclear image. `scale` rescales the image to the training magnification."""
    x = normalize(img)
    if scale != 1.0:
        x = rescale(x, scale, order=1, preserve_range=True, anti_aliasing=scale < 1).astype(np.float32)
    lab = probs_to_instances(predict_probs(model, x, device), min_size=30)
    if scale != 1.0:
        from skimage.transform import resize
        lab = resize(lab, img.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
    return lab


# ----------------------------------------------------------------- baseline
def otsu_watershed(img: np.ndarray, min_size: int = 30, min_dist: int = 5) -> np.ndarray:
    x = gaussian(normalize(img), 1)
    fg = x > threshold_otsu(x)
    fg = remove_small_objects(fg, min_size)
    dist = ndi.distance_transform_edt(fg)
    pk = peak_local_max(dist, min_distance=min_dist, labels=label(fg), exclude_border=False)
    markers = np.zeros_like(fg, dtype=np.int32)
    markers[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    return _relabel(watershed(-dist, markers, mask=fg))


def estimate_scale(img: np.ndarray, target_area: float) -> float:
    """Ratio that maps the median Otsu object area of `img` to `target_area` (training median)."""
    lab = otsu_watershed(img)
    areas = [r.area for r in regionprops(lab)]
    if not areas:
        return 1.0
    return float(np.clip(np.sqrt(target_area / np.median(areas)), 0.5, 4.0))


# ----------------------------------------------------------------- metrics
def iou_matrix(gt: np.ndarray, pr: np.ndarray):
    g, p = gt.ravel(), pr.ravel()
    ng, np_ = gt.max(), pr.max()
    if ng == 0 or np_ == 0:
        return np.zeros((ng, np_))
    inter = np.bincount(g * (np_ + 1) + p, minlength=(ng + 1) * (np_ + 1)).reshape(ng + 1, np_ + 1)
    ag, ap = inter.sum(1), inter.sum(0)
    union = ag[:, None] + ap[None, :] - inter
    iou = inter / np.maximum(union, 1)
    return iou[1:, 1:]


def match_f1(gt: np.ndarray, pr: np.ndarray, thr: float = 0.5):
    """One-to-one matching: at IoU > 0.5 matches are unique by construction; above that we keep it greedy."""
    iou = iou_matrix(gt, pr)
    if iou.size == 0:
        return 0.0, 0, int(pr.max()), int(gt.max())
    pairs = np.argwhere(iou > thr)
    order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]])
    ug, up, tp = set(), set(), 0
    for i, j in pairs[order]:
        if i in ug or j in up:
            continue
        ug.add(i); up.add(j); tp += 1
    fp, fn = iou.shape[1] - tp, iou.shape[0] - tp
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, tp, fp, fn
