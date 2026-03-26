"""Central paths and hyperparameters for the hackathon project."""
import os
from pathlib import Path

# Dataset root can be overridden for Kaggle or other remote runtimes.
DATA_ROOT = Path(os.environ.get("HACKATHON_DATA_ROOT", "/Users/umar/Downloads/Main hackathon dataset"))

CLASSIFICATION_TRAIN = DATA_ROOT / "classification" / "train"
CLASSIFICATION_TEST = DATA_ROOT / "classification" / "test"

SEGMENTATION_TRAIN_IMAGES = DATA_ROOT / "Segmentation" / "training" / "images"
SEGMENTATION_TRAIN_MASKS = DATA_ROOT / "Segmentation" / "training" / "masks"
SEGMENTATION_VAL_IMAGES = DATA_ROOT / "Segmentation" / "validation" / "images"
SEGMENTATION_VAL_MASKS = DATA_ROOT / "Segmentation" / "validation" / "masks"
SEGMENTATION_TEST_IMAGES = DATA_ROOT / "Segmentation" / "testing" / "images"

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CLS_CHECKPOINT = CHECKPOINT_DIR / "classification_best.pth"
SEG_CHECKPOINT = CHECKPOINT_DIR / "segmentation_best.pth"

# Submission: Excel is "{TEAM_NAME} test_ground_truth.xlsx"; nested folder for masks
TEAM_NAME = "Nerds"

# Classification (timm)
CLS_MODEL_NAME = "efficientnet_b3"
CLS_IMG_SIZE = 300
CLS_BATCH_SIZE = 16
CLS_VAL_FRACTION = 0.12
CLS_NUM_CLASSES = 12
CLS_NUM_WORKERS = 4
CLS_LABEL_SMOOTHING = 0.1
CLS_EARLY_STOP_PATIENCE = 12
# Mixup (train only); Beta(alpha, alpha)
CLS_MIXUP_ALPHA = 0.2
CLS_MIXUP_PROB = 0.5
# EMA weights for validation checkpoint (often +0.5–1% accuracy)
CLS_USE_EMA = True
CLS_EMA_DECAY = 0.999
# RandAugment (replaces heavy hand-tuned color jitter)
CLS_RANDAUG_N = 2
CLS_RANDAUG_M = 7

# Segmentation (smp)
# Training defaults are aligned to the current shipped checkpoint. Inference scripts read the
# architecture from checkpoint metadata and do not depend on these values.
SEG_ARCH = "unet"
SEG_ENCODER = "resnet34"
SEG_IMG_SIZE = 320
SEG_BATCH_SIZE = 4
SEG_NUM_WORKERS = 4
SEG_EARLY_STOP_PATIENCE = 15
SEG_GRAD_CLIP_NORM = 1.0
# Composite loss: dice + bce + w_focal * focal + w_tversky * tversky
SEG_LOSS_FOCAL_WEIGHT = 0.35
SEG_LOSS_TVERSKY_WEIGHT = 0.25
SEG_TVERSKY_ALPHA = 0.3
SEG_TVERSKY_BETA = 0.7

SEED = 42
