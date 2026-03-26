# Kaggle GPU Runbook

This repo is already set up to run on Kaggle without code changes because dataset location can be overridden with `HACKATHON_DATA_ROOT`.

## What You Must Do Yourself

- Create or sign in to your Kaggle account.
- Complete the phone-number verification Kaggle requires for GPU access.
- Create a notebook and enable `GPU` from the notebook settings.
- Upload:
  - this repo as a Kaggle Dataset or notebook files
  - the hackathon dataset as a private Kaggle Dataset

I cannot do the Kaggle account creation or phone verification for you from this environment.

## Recommended Kaggle Layout

Use paths like this inside the notebook:

- repo: `/kaggle/working/hackathon`
- dataset: `/kaggle/input/main-hackathon-dataset/Main hackathon dataset`

If your uploaded dataset name differs, use the actual Kaggle input path in the commands below.

## Notebook Setup Cells

### 1. Move into the repo

```bash
cd /kaggle/working
```

If you uploaded a zip, unzip it first. If you uploaded repo files directly, make sure the repo root is available under `/kaggle/working/hackathon`.

### 2. Install dependencies

```bash
cd /kaggle/working/hackathon
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Point the repo at the Kaggle dataset

```bash
export HACKATHON_DATA_ROOT="/kaggle/input/main-hackathon-dataset/Main hackathon dataset"
```

Smoke-test that the path is correct:

```bash
python - <<'PY'
import config
print(config.DATA_ROOT)
print(config.CLASSIFICATION_TRAIN.exists(), config.SEGMENTATION_TRAIN_IMAGES.exists())
PY
```

## High-Value GPU Jobs

## A. Validate the shipped checkpoints first

```bash
python scripts/evaluate_validation.py --seg-tta full --seg-threshold 0.6
python scripts/evaluate_validation.py --seg-tta full --seg-threshold 0.6 --sweep-cls-tta
```

Do this before retraining so you have a baseline.

## B. CPU-friendly classification retrain, but on Kaggle GPU

This is the current best next move because classification is weaker than segmentation.

```bash
python train_classification.py \
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
  --checkpoint-out checkpoints/classification_resnet34_kaggle.pth
```

Then evaluate it:

```bash
python scripts/evaluate_validation.py \
  --cls-checkpoint checkpoints/classification_resnet34_kaggle.pth \
  --seg-tta full \
  --seg-threshold 0.6
```

Acceptance rule:

- keep the new classifier only if validation accuracy improves by at least `0.015` over `0.6095`

## C. Segmentation retrain only if classification improves first

The current validated segmentation path is already strong:

- TTA: `full`
- threshold: `0.6`
- postprocess: `none`

Only spend Kaggle GPU time on segmentation retraining if classification has already improved and you still have time budget.

## Final Export on Kaggle

If you choose to ship a new classifier checkpoint:

```bash
cp checkpoints/classification_resnet34_kaggle.pth checkpoints/classification_best.pth
python scripts/export_submission.py
python scripts/verify_submission.py
```

If you keep the original classifier:

```bash
python scripts/export_submission.py
python scripts/verify_submission.py
```

## Download Back To Local Machine

Download these from Kaggle notebook output or working directory:

- `submission/Nerds/`
- any improved checkpoint you want to keep
- optional training logs or screenshots for your presentation

## What Is Already Possible Locally

- validation scoring
- submission export
- submission verification
- organizer inference-script verification
- demo app launch
- presentation prep docs

## What Kaggle GPU Is Best Used For

- classification retraining
- classification TTA sweeps
- segmentation retraining only if you still have time
- faster iteration on alternative backbones
