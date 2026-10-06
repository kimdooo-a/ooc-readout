# OoC-Readout

**A label-matched imaging readout ladder for organ-on-a-chip drug-response assays.**
Entry to *AI4S Open Innovation: AI for Life Science (AI + Organ-on-a-Chip)*, 5th Pazhou Algorithm Competition — category **Model & Algorithm**.

Raw two-channel images (reporter + nuclear stain) → nucleus segmentation (U-Net) → single cells → frozen DINOv2 embeddings → three readouts matched to the labels you have → dose-response curve, EC50 with bootstrap CI, Z′, V-factor → how many chips and cells per chip you need.

| Readout | Labels needed | Z′ on 4 public dose-response curves |
|---|---|---|
| **CAES** (Control-Anchored Embedding Score) | only *which chips are untreated* | 0.15 – 0.78 (ranks doses, ρ 0.88–0.91) |
| Classical N/C ratio | knowing what to measure (assay-specific) | 0.23 – 0.62 |
| **Few-shot probe** on frozen DINOv2 | 8 neg + 8 pos control wells (BBBC013) / 4 + 4 (BBBC014) | **0.87 – 0.94** |

* Segmenter on the held-out BBBC039 test split: **F1@0.5 = 0.948** (Otsu + watershed: 0.784).
* OoC-regime simulation: the probe reaches P(Z′ > 0.5) = 0.87 with **2 chips × 20 cells** per condition.
* Full numbers, methods and limitations: [`docs/technical_report.md`](docs/technical_report.md) (PDF: `docs/OoC-Readout_technical_report.pdf`).

![pipeline](results/fig_pipeline.png)

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # or: pip install torch torchvision (CPU)
pip install -r requirements.txt

python scripts/download_data.py                 # ~360 MB of public BBBC data into ./data (or set OOC_DATA)
python scripts/run_assay.py --plate BBBC013     # needs models/unet3_bbbc039.pt (included)
python scripts/demo.py --plate BBBC013 --well A08
```

Full reproduction of every number and figure in the report:

```bash
bash reproduce.sh
```

Runtime on one RTX 4080 SUPER: segmenter training 15 min (skipped if `models/unet3_bbbc039.pt` exists), each plate 1–2.5 min, sample-efficiency simulation ~30 min. CPU works (slower). No paid service, API key, proprietary hardware or non-public data is needed. DINOv2 weights are fetched once from the official `torch.hub` entry point.

## Demo — analyse your own field of view

```bash
python scripts/demo.py --reporter my_chip_GFP.png --nucleus my_chip_DAPI.png \
                       --models results/BBBC013/readout_models.joblib
```

Input: one reporter image and one nuclear image of the same field (any bit depth; grayscale).
Output (`demo_out/`): an overlay PNG with nuclei coloured by the probe's per-cell probability, and a JSON with
the number of cells, the classical N/C response as % of the positive control, the CAES mean score and responder
fraction, and the probe's mean probability of responding. Example console output:

```
input  : BBBC013 well A08 | Wortmannin 31.25 nM (sample)
step 1 : nuclei segmented        -> 300 cells
step 2 : DINOv2 embeddings        -> (300, 768)
step 3 : readouts
         classic_pct_of_positive_control      86.602
         caes_responder_fraction              0.383
         probe_mean_probability_responding    0.974
```

For a new assay, run `scripts/run_assay.py`-style fitting on your own control chips to create a matching
`readout_models.joblib` (the readout models are fitted on control wells only, in seconds).

## Repository layout

| Path | Content |
|---|---|
| `ooc_readout/data.py` | dataset download (parallel range requests), loaders, plate metadata |
| `ooc_readout/seg.py` | 3-class U-Net, instance recovery, Otsu+watershed baseline, magnification calibration, instance F1 |
| `ooc_readout/cells.py` | per-cell nuclear / cytoplasmic-ring intensities, crops |
| `ooc_readout/embed.py` | frozen DINOv2 ViT-S/14 featuriser |
| `ooc_readout/scoring.py` | CAES, few-shot probe, Z′, V-factor, Hill fit, bootstrap EC50 |
| `scripts/train_segmentation.py` / `evaluate_segmentation.py` | segmenter training / BBBC039 test evaluation |
| `scripts/run_assay.py` | end-to-end readout of one plate |
| `scripts/sample_efficiency.py` | organ-on-a-chip regime simulation (2–4 chips, 5–all cells) |
| `scripts/ablation.py`, `make_figures.py`, `extra_figures.py` | ablations, figures, tables |
| `scripts/demo.py` | one field in → per-cell and per-field readout out |
| `models/unet3_bbbc039.pt` | trained segmenter (31 MB) |
| `results/` | metrics (JSON/CSV/MD) and figures produced by the scripts |
| `docs/technical_report.md` | technical report (source of the PDF) |

### Inputs and outputs of `run_assay.py`

* **Input:** a plate of two-channel fields with metadata (well, dose, role = neg / pos / sample).
* **Output** (`results/<plate>/`): `cells.parquet` (one row per cell: position, area, nuclear/cytoplasmic intensity, log N/C, CAES score, probe probability), `embeddings.npy` (float16, 768-d per cell), `wells.csv` (per-well means and cell counts), `curves.json` (Hill parameters, EC50 and 95% CI, Z′, V-factor, Spearman ρ per readout and group), `readout_models.joblib`, overlays.

## Data and licences

| Dataset | Licence | Use |
|---|---|---|
| BBBC039v1 (Caicedo et al.) | CC0 | segmenter training / test |
| BBBC013v1 (Ilya Ravkin) | CC BY 3.0 | FKHR translocation dose-response (U2OS) |
| BBBC014v1 (Ilya Ravkin) | CC BY 3.0 | NF-κB translocation dose-response (MCF7, A549) |

"We used image sets BBBC013v1 and BBBC014v1 provided by Ilya Ravkin, and BBBC039v1 (Caicedo et al. 2018), available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]." Datasets are downloaded from the source, not redistributed. No personal, clinical or restricted data is used.

Code: MIT (see `LICENSE`). DINOv2: Apache-2.0 (Meta AI).

## AI disclosure

This project was built with the help of **Claude (Claude Opus 5.5) via Claude Code, Anthropic**, which wrote code, ran
the experiments locally and drafted the documentation; all reported numbers are produced by the scripts here.
See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) for the full list of AI tools, models and services and their purposes.

## Limitations

No organ-on-a-chip images were used — the chip regime is simulated by sub-sampling plate data. Two translocation
endpoints and three cell lines were evaluated. The probe needs a positive control; the label-free CAES is a ranking /
detection readout, not an assay-grade one on every curve. Details in Section 8 of the report.
