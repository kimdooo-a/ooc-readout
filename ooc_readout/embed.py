"""Frozen self-supervised embeddings of single-cell crops.

We use DINOv2 ViT-S/14 (Oquab et al., 2023; Apache-2.0; weights from the official torch.hub entry point).
The model is never fine-tuned and never sees a label: it only turns each 2-channel crop into a vector.
Channels are mapped to RGB as (reporter, nucleus, reporter) so both stains reach the patch embedding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Embedder:
    def __init__(self, name: str = "dinov2_vits14", side: int = 112, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True).to(self.device).eval()
        self.side = side

    @torch.no_grad()
    def __call__(self, crops: np.ndarray, batch: int = 512) -> np.ndarray:
        outs = []
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i:i + batch]).to(self.device)
            x = torch.stack([x[:, 0], x[:, 1], x[:, 0]], 1)
            x = F.interpolate(x, size=(self.side, self.side), mode="bilinear", align_corners=False)
            x = (x - _MEAN.to(x.device)) / _STD.to(x.device)
            f = self.model.forward_features(x)
            cls = f["x_norm_clstoken"]
            patch = f["x_norm_patchtokens"].mean(1)
            outs.append(torch.cat([cls, patch], 1).float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 768), np.float32)
