#!/usr/bin/env python3
"""Small Gradio demo for classification + segmentation."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

_CACHE_ROOT = Path("/tmp/codex-demo-cache")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

try:
    import gradio as gr
    _GRADIO_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - handled at runtime if gradio is missing
    gr = None
    _GRADIO_IMPORT_ERROR = exc

import config
from models.classification.classify import build_val_transform, load_model as load_cls_model, tta_classification_logits
from models.segmentation.segment import load_model as load_seg_model, prob_to_mask, preprocess, tta_segmentation_probs

APP_TITLE = "Nerds Biopsy AI Demo"
DISCLAIMER = "For research and demonstration purposes only. Not for clinical use."
DEMO_SEG_THRESHOLD = 0.6


@dataclass
class ModelBundle:
    cls_model: torch.nn.Module
    cls_img_size: int
    seg_model: torch.nn.Module
    seg_img_size: int
    device: torch.device


_MODELS: ModelBundle | None = None
_MODEL_ERROR: str | None = None


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models() -> tuple[ModelBundle | None, str | None]:
    global _MODELS, _MODEL_ERROR
    if _MODELS is not None or _MODEL_ERROR is not None:
        return _MODELS, _MODEL_ERROR

    missing = [path for path in (config.CLS_CHECKPOINT, config.SEG_CHECKPOINT) if not path.is_file()]
    if missing:
        _MODEL_ERROR = "Missing checkpoint(s): " + ", ".join(str(path) for path in missing)
        return None, _MODEL_ERROR

    device = choose_device()
    try:
        cls_model, cls_img_size = load_cls_model(config.CLS_CHECKPOINT, device)
        seg_model, seg_img_size = load_seg_model(config.SEG_CHECKPOINT, device)
        _MODELS = ModelBundle(
            cls_model=cls_model,
            cls_img_size=cls_img_size,
            seg_model=seg_model,
            seg_img_size=seg_img_size,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        _MODEL_ERROR = f"Failed to load models: {exc}"
    return _MODELS, _MODEL_ERROR


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    alpha = 0.35
    mask_bool = mask > 0
    overlay[mask_bool] = ((1.0 - alpha) * overlay[mask_bool] + alpha * red[mask_bool]).astype(np.uint8)
    return overlay


def classify_image(bundle: ModelBundle, image: Image.Image) -> pd.DataFrame:
    tensor = build_val_transform(bundle.cls_img_size, tta_mode="none")(image).unsqueeze(0).to(bundle.device)
    with torch.no_grad():
        logits = tta_classification_logits(bundle.cls_model, tensor, "none")
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:3]
    return pd.DataFrame(
        {
            "Label": [int(idx) for idx in top_indices],
            "Confidence": [round(float(probs[idx]), 4) for idx in top_indices],
        }
    )


def segment_image(bundle: ModelBundle, rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    tensor = preprocess(rgb, bundle.seg_img_size).to(bundle.device)
    with torch.no_grad():
        prob = tta_segmentation_probs(bundle.seg_model, tensor, "none")[0, 0].cpu().numpy()
    prob_full = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = prob_to_mask(prob_full, threshold=DEMO_SEG_THRESHOLD, postprocess="none")
    if mask.shape[:2] != (height, width):
        raise ValueError("Segmentation output size mismatch")
    return mask


def run_demo(image_path: str | None) -> tuple[Any, Any, Any, Any, str]:
    if not image_path:
        return None, None, None, None, "Upload a PNG or JPG biopsy image."

    bundle, model_error = load_models()
    if model_error:
        return None, None, None, None, model_error

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return None, None, None, None, "Selected file could not be found."
    except UnidentifiedImageError:
        return None, None, None, None, "The uploaded file is not a readable image."
    except Exception as exc:  # pragma: no cover - runtime guard
        return None, None, None, None, f"Could not read the uploaded image: {exc}"

    rgb = np.array(image)
    try:
        top3 = classify_image(bundle, image)
        mask = segment_image(bundle, rgb)
        overlay = overlay_mask(rgb, mask)
    except Exception as exc:  # pragma: no cover - runtime guard
        return rgb, None, None, None, f"Inference failed cleanly: {exc}"

    return rgb, top3, mask, overlay, "Inference complete."


def build_app():
    if gr is None:
        detail = f" ({_GRADIO_IMPORT_ERROR})" if _GRADIO_IMPORT_ERROR is not None else ""
        raise RuntimeError(f"Gradio is unavailable. Install or fix it with `pip install -r requirements.txt`{detail}.")

    css = """
    .app-shell {max-width: 1200px; margin: 0 auto;}
    .app-shell h1 {letter-spacing: 0.02em;}
    .disclaimer {border-left: 4px solid #b22222; padding-left: 12px; font-weight: 600;}
    """
    with gr.Blocks(title=APP_TITLE, css=css) as demo:
        gr.Markdown(f"<div class='app-shell'><h1>{APP_TITLE}</h1></div>")
        gr.Markdown(f"<div class='app-shell disclaimer'>{DISCLAIMER}</div>")
        with gr.Row(equal_height=True):
            image_input = gr.Image(
                type="filepath",
                label="Biopsy Image",
                sources=["upload"],
                image_mode="RGB",
            )
            with gr.Column():
                run_button = gr.Button("Run Inference", variant="primary")
                status = gr.Markdown("Models load on first use. Upload an image, then run inference.")
                top3 = gr.Dataframe(
                    headers=["Label", "Confidence"],
                    datatype=["number", "number"],
                    interactive=False,
                    label="Classification Top-3",
                )
        with gr.Row():
            original_output = gr.Image(label="Original Image", type="numpy")
            mask_output = gr.Image(label="Binary Segmentation Mask", type="numpy")
            overlay_output = gr.Image(label="Overlay", type="numpy")
        run_button.click(
            run_demo,
            inputs=image_input,
            outputs=[original_output, top3, mask_output, overlay_output, status],
        )
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
