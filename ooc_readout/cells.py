"""Single-cell measurements and crops.

For every segmented nucleus we measure
  * the classical translocation readout: log( mean reporter in nucleus / mean reporter in a cytoplasmic ring ),
    the quantity CellProfiler pipelines compute for these assays (Carpenter et al., Genome Biol. 2006);
  * a 2-channel crop (reporter, nucleus) centred on the nucleus, later embedded by a frozen foundation model.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import expand_labels


def background(img: np.ndarray) -> float:
    return float(np.percentile(img, 5))


def measure(rep: np.ndarray, nuc_lab: np.ndarray, ring: int = 4, gap: int = 1):
    """Return per-cell dict list with nuclear/cytoplasmic reporter intensities."""
    bg = background(rep)
    r = rep - bg
    inner = expand_labels(nuc_lab, gap)
    outer = expand_labels(nuc_lab, gap + ring)
    ring_lab = np.where(inner > 0, 0, outer)
    idx = np.arange(1, nuc_lab.max() + 1)
    n_mean = ndi.mean(r, nuc_lab, idx)
    c_mean = ndi.mean(r, ring_lab, idx)
    c_cnt = ndi.sum(np.ones_like(r), ring_lab, idx)
    out = []
    for k, p in zip(idx, regionprops(nuc_lab)):
        assert p.label == k
        if c_cnt[k - 1] < 5:
            continue
        nm, cm = max(n_mean[k - 1], 1e-3), max(c_mean[k - 1], 1e-3)
        out.append({"label": int(k), "y": p.centroid[0], "x": p.centroid[1], "area": p.area,
                    "nuc_mean": nm, "cyto_mean": cm, "log_nc": float(np.log(nm / cm))})
    return out


def crops(rep: np.ndarray, nuc: np.ndarray, cells: list, size: int):
    """Square crops (2, size, size) centred on each nucleus; per-image robust normalisation."""
    def norm(a):
        lo, hi = np.percentile(a, [1, 99.8])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    st = np.stack([norm(rep), norm(nuc)])
    h = size // 2
    st = np.pad(st, ((0, 0), (h, h), (h, h)), mode="constant")
    out = np.zeros((len(cells), 2, size, size), np.float32)
    for i, c in enumerate(cells):
        y, x = int(round(c["y"])) + h, int(round(c["x"])) + h
        out[i] = st[:, y - h:y - h + size, x - h:x - h + size]
    return out
