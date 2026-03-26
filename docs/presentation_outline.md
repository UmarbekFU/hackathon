# Nerds Hackathon Presentation Outline

## 1. Problem and Goal
- This hackathon asks for two outputs from biopsy images: 12-class classification and binary region-of-interest segmentation.
- The team goal is to maximize the shortlist score first, then demonstrate a reliable interface and clear technical understanding.
- State the required disclaimer verbally and on-screen: `For research and demonstration purposes only. Not for clinical use.`

## 2. AI Workflow
- Input: biopsy image.
- Branch 1: classification model predicts one of 12 hidden classes.
- Branch 2: segmentation model predicts a binary mask for the relevant region.
- Output: class prediction, confidence ranking, binary mask, and overlay visualization in the demo.

## 3. Data Verification and Preprocessing
- Confirm dataset counts matched the organizer brief before training.
- Keep train/validation/test separation strict; do not use test images for training.
- Classification preprocessing:
  - Resize/crop to the model input size.
  - Normalize with ImageNet mean/std.
  - Training augmentations: random resized crop, RandAugment, horizontal flip, optional vertical flip for targeted retrains, optional Mixup.
- Segmentation preprocessing:
  - Pair image and mask by filename stem.
  - Resize to training size with normalized RGB input.
  - Use spatial augmentations and limited intensity augmentation.
  - Convert masks to binary targets.

## 4. Model Design
- Classification baseline:
  - `timm` pretrained image classifier.
  - Shipped checkpoint: EfficientNet-B3.
  - Class-weighted cross-entropy with label smoothing, EMA checkpointing, and optional Mixup.
- Segmentation baseline:
  - Shipped checkpoint loads as U-Net with ResNet34 encoder.
  - Composite loss mixes Dice, BCE, Focal, and Tversky to balance overlap quality and foreground recall.
- Why pretrained models:
  - The dataset is limited, especially for segmentation and minority classes, so transfer learning is more data-efficient and stable than training from scratch.

## 5. Training and Inference Procedure
- Classification:
  - Build a validation split from the official training set only.
  - Track overall validation accuracy and per-class accuracy.
  - Use TTA sweeps only when validation shows a real gain.
- Segmentation:
  - Train on official training pairs and evaluate on the official validation split.
  - Tune inference separately with TTA, threshold selection, and small binary postprocessing sweeps.
- Submission export:
  - Generate the Excel file for classification.
  - Generate 200 binary PNG masks for segmentation with exact filenames and image sizes.
  - Package the inference-only scripts and checkpoints required by the organizers.

## 6. Challenges and Solutions
- Class imbalance in classification:
  - Use class weighting and an optional weighted sampler.
  - Inspect per-class validation accuracy instead of only overall accuracy.
- Limited segmentation data:
  - Use transfer learning, composite loss, and low-risk inference tuning instead of architecture churn.
- CPU-only constraint:
  - Prioritize inference-time gains first.
  - Keep the live demo on the fast path without TTA.
- Submission reliability:
  - Keep a verifier script and package organizer-facing inference scripts with the model artifacts.

## 7. Results and Observations
- Current validation proxy baseline:
  - Classification accuracy: about `0.6095`.
  - Segmentation mIoU: about `0.7382`.
- Best low-risk segmentation gain found from inference tuning:
  - Full TTA with threshold `0.6` improved validation mIoU to about `0.7648`.
- Explain that the shortlist score is estimated with the same formula structure used in the brief:
  - `30 * classification_accuracy + 40 * segmentation_mIoU`.

## 8. Demo Flow
- Show one image upload.
- Run classification and segmentation from preloaded models.
- Display:
  - original image
  - classification top-3 predictions with confidences
  - binary mask
  - overlay
- Point out clean error handling and the disclaimer.

## 9. Close
- Emphasize three things:
  - strict validation discipline
  - reliable packaging and reproducibility
  - practical demo quality under hackathon time constraints
