"""Dataset access for OoC-Readout.

All datasets are public Broad Bioimage Benchmark Collection (BBBC) sets:
  * BBBC039v1 (CC0)          - nuclei instance masks, used to TRAIN/EVALUATE the segmenter
  * BBBC013v1 (CC BY 3.0)    - FKHR-GFP cytoplasm->nucleus translocation, U2OS, Wortmannin / LY294002 dose curves
  * BBBC014v1 (CC BY 3.0)    - NF-kB cytoplasm->nucleus translocation, MCF7 + A549, TNF-alpha dose curves
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from skimage.measure import label

BBBC = "https://data.broadinstitute.org/bbbc"
FILES = {
    "BBBC039": ["images.zip", "masks.zip", "metadata.zip"],
    "BBBC013": ["BBBC013_v1_images_bmp.zip", "BBBC013_reproduce_logan.zip"],
    "BBBC014": ["BBBC014_v1_images.zip", "BBBC014_v1_platemap_all.txt"],
}


def data_root() -> Path:
    return Path(os.environ.get("OOC_DATA", Path(__file__).resolve().parents[1] / "data"))


# ---------------------------------------------------------------- download
def _download(url: str, out: Path, n_threads: int = 32, chunk: int = 1 << 20) -> None:
    """Parallel HTTP range download (the BBBC server is slow per connection)."""
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    size = int(head.headers["Content-Length"])
    if out.exists() and out.stat().st_size == size:
        return
    parts = [(a, min(a + chunk, size) - 1) for a in range(0, size, chunk)]
    tmp = Path(str(out) + ".parts")
    tmp.mkdir(parents=True, exist_ok=True)
    lock, nxt = threading.Lock(), [0]

    def work():
        while True:
            with lock:
                if nxt[0] >= len(parts):
                    return
                k = nxt[0]
                nxt[0] += 1
            a, b = parts[k]
            p = tmp / f"{k:06d}"
            if p.exists() and p.stat().st_size == b - a + 1:
                continue
            for _ in range(20):
                try:
                    r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                    d = urllib.request.urlopen(r, timeout=120).read()
                    if len(d) == b - a + 1:
                        p.write_bytes(d)
                        break
                except Exception:
                    time.sleep(2)

    ts = [threading.Thread(target=work) for _ in range(n_threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    with open(out, "wb") as f:
        f.writelines((tmp / f"{k:06d}").read_bytes() for k in range(len(parts)))
    if out.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {url}")
    shutil.rmtree(tmp)


def download_all(root: Path | None = None) -> Path:
    root = root or data_root()
    for ds, files in FILES.items():
        d = root / ds
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            out = d / f
            print(f"[download] {ds}/{f}")
            _download(f"{BBBC}/{ds}/{f}", out)
            if f.endswith(".zip"):
                with zipfile.ZipFile(out) as z:
                    z.extractall(d / ("img" if "images" in f and ds != "BBBC039" else ("logan" if "logan" in f else ".")))
    return root


# ---------------------------------------------------------------- BBBC039
def bbbc039_split(split: str, root: Path | None = None):
    """Return list of (image float32 HxW, instance label int32 HxW) for the official split."""
    root = (root or data_root()) / "BBBC039"
    names = [l.strip() for l in open(root / "metadata" / f"{split}.txt") if l.strip()]
    out = []
    for n in names:
        stem = n.replace(".png", "")
        img = iio.imread(root / "images" / f"{stem}.tif").astype(np.float32)
        m = iio.imread(root / "masks" / f"{stem}.png")[..., 0]
        out.append((img, label(m, background=0).astype(np.int32), stem))
    return out


# ---------------------------------------------------------------- plates
@dataclass
class Well:
    plate: str          # "BBBC013" | "BBBC014"
    well: str           # e.g. "A03"
    row: str
    col: int
    group: str          # drug / cell line group used for one dose-response curve
    dose: float         # concentration (units in `unit`)
    unit: str
    role: str           # "neg" | "pos" | "sample"
    reporter_path: str
    nucleus_path: str


def bbbc013_wells(root: Path | None = None) -> list[Well]:
    """BBBC013: rows A-D Wortmannin (nM), rows E-H LY294002 (uM).

    Metadata comes from the authors' CellProfiler files (Logan & Carpenter 2010) shipped with the set:
    Metadata_PosNegCtrls 0 = negative control, 1 = positive control (150 nM Wortmannin), 0.5 = dose sample.
    Control wells (A-D col 1/12, E-H col 1/12) are shared by both drugs; `group` of a control well is just
    its row's drug and analyses pool controls plate-wide.
    """
    root = (root or data_root()) / "BBBC013"
    img = next(Path(p) for p in glob.glob(str(root / "img" / "*")) if os.path.isdir(p))
    meta = pd.concat([
        pd.read_csv(glob.glob(str(root / "logan" / "**" / "*WellMetadata_wortmannin.csv"), recursive=True)[0]),
        pd.read_csv(glob.glob(str(root / "logan" / "**" / "*WellMetadata_LY294002*.csv"), recursive=True)[0]),
    ]).drop_duplicates("Metadata_Well")  # the LY294002 file repeats the shared A-D control wells
    wells = []
    for _, r in meta.iterrows():
        drug = "Wortmannin" if r.Metadata_WellRow in "ABCD" else "LY294002"
        role = {0: "neg", 1: "pos"}.get(float(r.Metadata_PosNegCtrls), "sample")
        wells.append(Well("BBBC013", r.Metadata_Well, r.Metadata_WellRow, int(r.Metadata_WellColumn), drug,
                          float(r.Metadata_Dose), "nM" if drug == "Wortmannin" else "uM", role,
                          str(img / r.Image_FileName_rawGFP), str(img / r.Image_FileName_rawDNA)))
    return wells


def bbbc014_wells(root: Path | None = None) -> list[Well]:
    """BBBC014: 12 TNF-alpha concentrations in columns (1e-7 M ... 1e-13 M), rows A-D MCF7, rows E-H A549.

    Channel 1 = FITC (NF-kB reporter), Channel 2 = DAPI (verified visually: Channel 2 is purely nuclear).
    The lowest concentration (1e-13 M, column 12) serves as the negative reference and the highest
    (1e-7 M, column 1) as the positive reference, as there are no separate control wells on this plate.
    """
    root = (root or data_root()) / "BBBC014"
    img = next(Path(p) for p in glob.glob(str(root / "img" / "*")) if os.path.isdir(p))
    raw = open(root / "BBBC014_v1_platemap_all.txt", "rb").read().decode("latin-1")
    toks = [t for t in re.split(r"[\r\n]+", raw) if t.strip()][1:]
    assert len(toks) == 96, len(toks)
    wells = []
    for i, t in enumerate(toks):
        row, col = "ABCDEFGH"[i // 12], i % 12 + 1
        dose = 10 ** float(t.split("^")[1])  # molar
        line = "MCF7" if row in "ABCD" else "A549"
        role = "pos" if col == 1 else ("neg" if col == 12 else "sample")
        f = lambda ch: str(img / f"Channel {ch}-{i + 1:02d}-{row}-{col:02d}-00.Bmp")
        wells.append(Well("BBBC014", f"{row}{col:02d}", row, col, line, dose * 1e12, "pM", role, f(1), f(2)))
    return wells


def load_pair(w: Well):
    rep = iio.imread(w.reporter_path).astype(np.float32)
    nuc = iio.imread(w.nucleus_path).astype(np.float32)
    if rep.ndim == 3:
        rep = rep[..., 0]
    if nuc.ndim == 3:
        nuc = nuc[..., 0]
    return rep, nuc
