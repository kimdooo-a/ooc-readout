# AI and Third-Party Tool Disclosure

This disclosure is made under Challenge Rules §4 ("Where third-party models, APIs, software, or AI tools constitute
a material component of the project, the participating team must disclose their names, sources, and primary
purposes in the submission materials").

## 1. Generative AI assistant used to build the project

| Name | Source | Primary purpose in this project |
|---|---|---|
| **Claude** (model *Claude Opus 5.5*), used through **Claude Code** | Anthropic, https://www.anthropic.com · https://claude.com/claude-code | Selecting public datasets and checking their licences; writing the Python code in this repository; running the training, evaluation and analysis scripts on the team's local workstation; drafting the README, the technical report, the Kaggle Writeup text and the demo-video narration script; assembling the demo video with the scripts listed in §3. |

How it was used, concretely:
- Claude ran as a coding agent on the team's own Windows workstation (one NVIDIA RTX 4080 SUPER GPU). It
  wrote and executed code locally; **no competition or dataset images were uploaded to any external AI
  service** — all model training and inference happened on the local GPU.
- The team member reviewed the generated code, the numerical results and the written materials. All numbers
  reported in the technical report, the Writeup and the video are produced by the scripts in this repository
  and can be regenerated with `reproduce.sh`; they were not typed by hand or generated as text by the assistant.
- Responsibility for authenticity, accuracy, legality and IP compliance (Rules §4) remains with the team.

## 2. Models that are a material component of the method

| Name | Source / licence | Purpose |
|---|---|---|
| **DINOv2 ViT-S/14** (frozen, not fine-tuned) | Meta AI, Oquab et al. 2023, https://github.com/facebookresearch/dinov2 — Apache-2.0; loaded with `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` | Turns single-cell image crops into 768-d feature vectors (CLS token + mean patch token). |
| **U-Net nucleus segmenter** (trained from scratch by this project) | This repository, MIT; trained only on BBBC039 (CC0) | Nucleus instance segmentation. Weights: `models/unet3_bbbc039.pt`. |

## 3. Other software and services

| Name | Source / licence | Purpose |
|---|---|---|
| PyTorch, torchvision | BSD-3 | Training / inference |
| scikit-learn, scikit-image, SciPy, NumPy, pandas, matplotlib | BSD-3 / BSD / PSF-compatible | Statistics, image processing, figures |
| edge-tts (Python package) → Microsoft Edge online neural text-to-speech, voice `en-US-AndrewNeural` | edge-tts: LGPL-3.0; voice service: Microsoft | Narration audio of the demo video (the narration text was written for this project) |
| FFmpeg | LGPL/GPL | Video assembly |
| Playwright + Chromium | Apache-2.0 / BSD | Rendering slide cards and the PDF report |

## 4. Data (no personal, clinical or restricted data)

| Dataset | Licence | Use |
|---|---|---|
| BBBC039v1 (Caicedo et al. 2019) | CC0 | Segmenter training / test |
| BBBC013v1 (provided by Ilya Ravkin) | CC BY 3.0 | Translocation dose-response validation |
| BBBC014v1 (provided by Ilya Ravkin) | CC BY 3.0 | Translocation dose-response validation (MCF7, A549) |

"We used image sets BBBC013v1 and BBBC014v1 provided by Ilya Ravkin, and BBBC039v1 (Caicedo et al. 2018),
available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012]."
