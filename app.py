#!/usr/bin/env python3
"""Minimal local demo server for classification + segmentation inference."""
from __future__ import annotations

import argparse
import base64
import html
import io
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

import config
from models.classification.classify import (
    build_val_transform,
    load_model as load_cls_model,
    tta_classification_logits,
)
from models.segmentation.segment import (
    load_model as load_seg_model,
    preprocess as preprocess_seg,
    prob_to_mask,
    tta_segmentation_probs,
)


HOST = "127.0.0.1"
DEFAULT_PORT = 7865
CLS_TTA_MODE = "tencrop"
SEG_TTA_MODE = "full"
SEG_THRESHOLD = 0.6
SEG_POSTPROCESS = "none"
DISCLAIMER = "For research and demonstration purposes only. Not for clinical use."


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.uint8)
    color = np.zeros_like(base)
    color[..., 0] = 255
    positive = mask > 0
    blended = base.copy().astype(np.float32)
    blended[positive] = (1.0 - alpha) * blended[positive] + alpha * color[positive]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


class DemoRuntime:
    def __init__(self) -> None:
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.cls_model, self.cls_img_size = load_cls_model(config.CLS_CHECKPOINT, self.device)
        self.seg_model, self.seg_img_size = load_seg_model(config.SEG_CHECKPOINT, self.device)
        self.cls_transform = build_val_transform(self.cls_img_size, tta_mode=CLS_TTA_MODE)
        self.class_names = [str(i) for i in range(config.CLS_NUM_CLASSES)]

    @torch.no_grad()
    def classify(self, image: Image.Image) -> list[tuple[str, float]]:
        x = self.cls_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = tta_classification_logits(self.cls_model, x, CLS_TTA_MODE)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=min(3, probs.numel()))
        return [
            (self.class_names[int(index)], float(value))
            for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
        ]

    @torch.no_grad()
    def segment(self, image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        x = preprocess_seg(rgb, self.seg_img_size).to(self.device)
        prob = tta_segmentation_probs(self.seg_model, x, SEG_TTA_MODE)[0, 0].cpu().numpy()
        prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return prob_to_mask(prob_full, threshold=SEG_THRESHOLD, postprocess=SEG_POSTPROCESS)

    def run(self, raw_bytes: bytes) -> dict[str, Any]:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        top3 = self.classify(image)
        mask = self.segment(image)
        overlay = overlay_mask(image, mask)
        return {
            "original": image_to_base64(image),
            "mask": image_to_base64(Image.fromarray(mask)),
            "overlay": image_to_base64(overlay),
            "top3": top3,
        }


def render_page(result: dict[str, Any] | None = None, error: str | None = None) -> bytes:
    result_html = ""
    if error:
        result_html = f'<p class="error">{html.escape(error)}</p>'
    elif result:
        rows = "".join(
            f"<li>Class {html.escape(label)}: {score * 100:.2f}%</li>"
            for label, score in result["top3"]
        )
        result_html = f"""
        <section class="results">
          <div class="panel">
            <h2>Top-3 Predictions</h2>
            <ol>{rows}</ol>
            <p class="note">{html.escape(DISCLAIMER)}</p>
          </div>
          <div class="grid">
            <figure><figcaption>Original</figcaption><img src="data:image/png;base64,{result["original"]}" alt="Original image"></figure>
            <figure><figcaption>Mask</figcaption><img src="data:image/png;base64,{result["mask"]}" alt="Predicted mask"></figure>
            <figure><figcaption>Overlay</figcaption><img src="data:image/png;base64,{result["overlay"]}" alt="Overlay image"></figure>
          </div>
        </section>
        """

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nerds Demo</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --card: #fffdf9;
      --ink: #1c1b19;
      --accent: #8f2d1f;
      --border: #d8cdc0;
    }}
    body {{
      margin: 0;
      font-family: Georgia, serif;
      color: var(--ink);
      background: linear-gradient(135deg, #efe6da, #f8f4ee 50%, #e5ddd0);
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .card, .panel, figure {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
    }}
    .card {{
      padding: 24px;
      margin-bottom: 24px;
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    form {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    input[type=file] {{
      max-width: 100%;
    }}
    button {{
      background: var(--accent);
      color: white;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font-size: 16px;
      cursor: pointer;
    }}
    .results {{
      display: grid;
      gap: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      padding: 12px;
    }}
    figcaption {{
      font-weight: 700;
      margin-bottom: 8px;
    }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 12px;
      display: block;
    }}
    .note {{
      color: #5f564b;
    }}
    .error {{
      color: #9b1c1c;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Nerds Healthcare Demo</h1>
      <p>Upload one biopsy image to run classification and segmentation with the local checkpoints.</p>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept=".png,.jpg,.jpeg,image/png,image/jpeg" required>
        <button type="submit">Run Inference</button>
      </form>
    </section>
    {result_html}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    runtime: DemoRuntime

    def do_GET(self) -> None:
        body = render_page()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            content_type = self.headers.get("Content-Type", "")
            boundary_token = "boundary="
            if "multipart/form-data" not in content_type or boundary_token not in content_type:
                raise ValueError("Expected multipart form upload.")
            boundary = content_type.split(boundary_token, 1)[1].encode("utf-8")
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            uploaded = self._extract_file(raw, boundary)
            result = self.runtime.run(uploaded)
            body = render_page(result=result)
            status = HTTPStatus.OK
        except Exception as exc:
            body = render_page(error=str(exc))
            status = HTTPStatus.BAD_REQUEST

        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    @staticmethod
    def _extract_file(body: bytes, boundary: bytes) -> bytes:
        marker = b"--" + boundary
        for part in body.split(marker):
            if b'name="image"' not in part:
                continue
            head_end = part.find(b"\r\n\r\n")
            if head_end == -1:
                continue
            payload = part[head_end + 4 :]
            tail = b"\r\n--"
            tail_index = payload.find(tail)
            if tail_index != -1:
                payload = payload[:tail_index]
            return payload.rstrip(b"\r\n")
        raise ValueError("No uploaded image found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local demo server.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    DemoHandler.runtime = DemoRuntime()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Serving demo on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
