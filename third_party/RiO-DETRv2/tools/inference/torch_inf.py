"""Run a RiO-DETRv2 checkpoint on an image or a video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.core import YAMLConfig


DOTA_CLASSES = (
    "plane", "baseball-diamond", "bridge", "ground-track-field",
    "small-vehicle", "large-vehicle", "ship", "tennis-court",
    "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
    "harbor", "swimming-pool", "helicopter",
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resize_and_pad(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float]:
    """Resize to fit ``size`` and pad the right and bottom edges."""
    target_h, target_w = size
    width, height = image.size
    scale = min(target_w / width, target_h / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    padded.paste(resized, (0, 0))
    return padded, scale


def obb_to_polygon(box: Sequence[float]) -> np.ndarray:
    """Convert cx, cy, width, height, angle in radians to four corners."""
    cx, cy, width, height, angle = map(float, box)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    half_width = direction * (width / 2.0)
    half_height = normal * (height / 2.0)
    center = np.array([cx, cy], dtype=np.float32)
    return np.stack((
        center - half_width - half_height,
        center + half_width - half_height,
        center + half_width + half_height,
        center - half_width + half_height,
    ))


def load_class_names(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        names = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    else:
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    return names or None


def _label_text(label: int, score: float, class_names: Sequence[str] | None) -> str:
    name = class_names[label] if class_names and 0 <= label < len(class_names) else str(label)
    return f"{name} {score:.2f}"


def draw_detections(
    image: Image.Image,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float = 0.4,
    class_names: Sequence[str] | None = None,
) -> Image.Image:
    """Draw oriented detections on a PIL image and return a copy."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    labels = labels.detach().cpu()
    boxes = boxes.detach().cpu()
    scores = scores.detach().cpu()
    keep = scores >= threshold
    for label, box, score in zip(labels[keep], boxes[keep], scores[keep]):
        polygon = obb_to_polygon(box.tolist())
        points = [tuple(map(float, point)) for point in polygon]
        draw.line(points + [points[0]], fill=(255, 0, 0), width=2, joint="curve")
        draw.text(points[0], _label_text(int(label), float(score), class_names), fill=(0, 0, 255))
    return canvas


def _read_checkpoint(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def build_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, tuple[int, int]]:
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.model.load_state_dict(_read_checkpoint(checkpoint_path))

    class DeployedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images: torch.Tensor, original_sizes: torch.Tensor):
            return self.postprocessor(self.model(images), original_sizes)

    model = DeployedModel().to(device).eval()
    input_shape = getattr(cfg.postprocessor, "input_shape", None)
    if input_shape is None:
        raise ValueError("PostProcessorOBB.input_shape is required")
    return model, (int(input_shape[0]), int(input_shape[1]))


def predict_image(model, image, input_shape, device, threshold, class_names):
    original = image.convert("RGB")
    padded, _ = resize_and_pad(original, input_shape)
    tensor = T.ToTensor()(padded).unsqueeze(0).to(device)
    height, width = original.size[1], original.size[0]
    original_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        labels, boxes, scores = model(tensor, original_size)
    result = draw_detections(
        original, labels[0], boxes[0], scores[0], threshold, class_names
    )
    count = int((scores[0].detach().cpu() >= threshold).sum().item())
    return result, count


def process_image(model, input_path, output_path, input_shape, device, threshold, class_names):
    image = Image.open(input_path).convert("RGB")
    result, count = predict_image(
        model, image, input_shape, device, threshold, class_names
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved {count} detections to {output_path}")


def process_video(model, input_path, output_path, input_shape, device, threshold, class_names):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise OSError(f"Unable to open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and np.isfinite(fps) and fps > 0 else 30.0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Unable to create output video: {output_path}")
    frame_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result, _ = predict_image(
                model, image, input_shape, device, threshold, class_names
            )
            writer.write(cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR))
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames", flush=True)
    finally:
        capture.release()
        writer.release()
    print(f"Saved {frame_count} frames to {output_path}")


def _default_output(input_path: str, is_image: bool) -> str:
    path = Path(input_path)
    return str(path.with_name(path.stem + "_pred" + (path.suffix if is_image else ".mp4")))


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model, input_shape = build_model(args.config, args.resume, device)
    class_names = load_class_names(args.class_names)
    if class_names is None and "dota" in args.config.lower():
        class_names = DOTA_CLASSES
    input_path = Path(args.input)
    is_image = input_path.suffix.lower() in IMAGE_EXTENSIONS
    output_path = args.output or _default_output(args.input, is_image)
    if is_image:
        process_image(model, args.input, output_path, input_shape, device, args.threshold, class_names)
    else:
        process_video(model, args.input, output_path, input_shape, device, args.threshold, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Model YAML config")
    parser.add_argument("-r", "--resume", required=True, help="Checkpoint (.pth)")
    parser.add_argument("-i", "--input", required=True, help="Image or video path")
    parser.add_argument("-o", "--output", help="Output image or video path")
    parser.add_argument("-d", "--device", default="cpu", help="Torch device, for example cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.4, help="Score threshold")
    parser.add_argument("--class-names", help="Comma-separated class names or a text file")
    main(parser.parse_args())
