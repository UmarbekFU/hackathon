# AI in Healthcare Hackathon — Nerds

PyTorch pipeline for the Central Asian University AI in Healthcare Hackathon:

- 12-class biopsy image classification
- binary biopsy region segmentation
- inference-only submission packaging for the organizer checks
- local Gradio demo for the Top 10 presentation round

The current shipped checkpoints are:

- classification: `timm` EfficientNet-B3
- segmentation: U-Net with ResNet34 encoder

Inference always uses checkpoint metadata. Training defaults in [`config.py`](config.py) are aligned to the shipped segmentation checkpoint to avoid drift.

## Setup

Use the project virtualenv, not system Python.

```bash
cd /Users/umar/Desktop/hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Override the dataset location with either:

- `HACKATHON_DATA_ROOT=/your/dataset/path`
- or a direct edit to [`config.py`](config.py)

## Validate Current Checkpoints

```bash
./.venv/bin/python scripts/evaluate_validation.py
./.venv/bin/python scripts/evaluate_validation.py --seg-tta full --seg-threshold 0.6
./.venv/bin/python scripts/evaluate_validation.py --sweep-cls-tta
./.venv/bin/python scripts/evaluate_validation.py --seg-tta full --seg-threshold 0.6 --sweep-seg-postprocess
```

The script reports:

- classification accuracy
- per-class classification accuracy
- segmentation mIoU
- composite proxy score `30 * acc + 40 * mIoU`

## Train

Uses only official training data.

### Default training

```bash
./.venv/bin/python train_classification.py --epochs 40
./.venv/bin/python train_segmentation.py --epochs 50
```

### CPU-friendly classification retrain path

```bash
./.venv/bin/python train_classification.py \
  --model-name resnet34 \
  --img-size 224 \
  --weighted-sampler \
  --label-smoothing 0.05 \
  --randaug-n 1 \
  --randaug-m 5 \
  --vflip-prob 0.2 \
  --freeze-backbone-epochs 3 \
  --epochs 15 \
  --early-stop-patience 4 \
  --checkpoint-out checkpoints/classification_resnet34_cpu.pth
```

## Build Submission

Default export now uses the strongest validated segmentation path:

- segmentation TTA: `full`
- segmentation threshold: `0.6`
- segmentation postprocess: `none`

```bash
./.venv/bin/python scripts/export_submission.py
```

Examples:

```bash
./.venv/bin/python scripts/export_submission.py --cls-tta-mode h
./.venv/bin/python scripts/export_submission.py --cls-tta-mode tencrop --seg-tta-mode full --threshold 0.6
./.venv/bin/python scripts/export_submission.py --seg-tta-mode full --threshold 0.6 --postprocess close3
```

Output folder:

- `submission/Nerds/Nerds test_ground_truth.xlsx`
- `submission/Nerds/Nerds/*.png`
- `submission/Nerds/models/...`

## Verify Before Upload

```bash
./.venv/bin/python scripts/verify_submission.py
```

## Organizer Verification Commands

After unzipping the exported folder:

```bash
pip install -r models/classification/requirements.txt
python models/classification/classify.py --test_dir <path> --checkpoint classification_best.pth --output out.xlsx

pip install -r models/segmentation/requirements.txt
python models/segmentation/segment.py --images_dir <path> --output_dir <path> --checkpoint segmentation_best.pth
```

Optional inference flags:

- classification: `--tta`, `--tta_full`, `--tta_tencrop`, or `--tta_mode`
- segmentation: `--tta`, `--tta_full`, `--tta_mode`, `--threshold`, `--postprocess`

## Demo App

```bash
./.venv/bin/python app.py
```

The demo:

- preloads both models once
- accepts one uploaded PNG or JPG
- shows classification top-3 predictions
- shows the binary mask and overlay
- includes the required disclaimer

## Presentation Prep

- outline: [`docs/presentation_outline.md`](docs/presentation_outline.md)
- judge Q&A: [`docs/judge_qa.md`](docs/judge_qa.md)
- Kaggle GPU runbook: [`docs/kaggle_runbook.md`](docs/kaggle_runbook.md)

---

For research and demonstration purposes only. Not for clinical use.
