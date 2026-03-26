# Judge Q&A Prep

## Why did you use pretrained models instead of training from scratch?
Pretrained backbones are the pragmatic choice for this dataset size. The classification task has class imbalance, and the segmentation set is small enough that transfer learning gives a better accuracy-to-time tradeoff and more stable convergence than random initialization.

## How did you avoid data leakage?
We trained only on the official training folders. Classification validation was split from the official training set only. Segmentation used the official training and validation folders exactly as provided. Test images were never used for training or manual labeling.

## How did you handle class imbalance in classification?
We used class-weighted cross-entropy, tracked per-class validation accuracy, and added an optional weighted sampler for the CPU-friendly retrain path. That avoids hiding weak classes behind a single overall accuracy number.

## Why are the disease names hidden, and how did that affect your approach?
The labels are intentionally anonymized by the organizers, so we treated the task as pure visual pattern recognition. That meant focusing on robust generalization, disciplined validation, and class-balanced training rather than class-specific medical heuristics.

## Why this segmentation loss combination?
Dice improves overlap quality, BCE stabilizes pixel-wise optimization, Focal helps with harder foreground pixels, and Tversky biases the loss toward foreground recall. That combination is a practical fit for small binary medical segmentation datasets.

## Why did you tune threshold and TTA separately from training?
They are low-risk inference-time controls that can improve validation mIoU without retraining. We measured them on the official validation split and only kept settings that produced a real gain.

## Why is the shipped segmentation model described as U-Net with ResNet34 when the repo once mentioned DeepLabV3+?
The repo had drift between training defaults and the checkpoint actually being shipped. We aligned the documentation and config to the validated checkpoint behavior and kept inference tied to checkpoint metadata so the packaged artifact is reproducible.

## Why does the live demo skip TTA?
The judging demo needs to be responsive and reliable on local CPU. We preload both models, use the fast path in the UI, and reserve heavier TTA modes for offline export when latency does not matter.

## How do you know the submission package is valid?
We run the verifier script against the exported folder, confirm the Excel row count and label range, confirm there are exactly 200 binary PNG masks, and run the packaged organizer-facing inference scripts from the copied `models/` directory.

## What are the main weaknesses of the current system?
Classification is the weaker branch, especially on a few minority or visually ambiguous classes. The system is also a competition prototype, not a clinical product, which is why the interface explicitly states that it is for research and demonstration purposes only.

## Why is the system not for clinical use?
The dataset is anonymized and competition-scoped, the models were optimized for hackathon evaluation rather than clinical validation, and there has been no regulatory, safety, or prospective clinical evaluation. The system is a research demonstration only.
