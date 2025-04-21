"""
Batch inference script: loads a TorchScript model, runs inference
on images in a directory, annotates predictions, and measures timing.
"""

import argparse
import os
import time

import cv2
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.nn.functional import softmax


def load_model(model_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """
    Load a TorchScript model onto the specified device and set to eval mode.

    Args:
        model_path (str): Path to the saved TorchScript model file.
        device (torch.device): Device to map the model to (cpu or cuda).

    Returns:
        torch.jit.ScriptModule: Loaded TorchScript model in eval mode.
    """
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def get_image_transform() -> transforms.Compose:
    """
    Construct the preprocessing pipeline for input images.

    Returns:
        transforms.Compose: Composed torchvision transforms for resizing,
            cropping, tensor conversion, and normalization.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_class_names(training_dir: str) -> list[str]:
    """
    Load class names from an ImageFolder-formatted directory.

    Args:
        training_dir (str): Path to directory containing subfolders per class.

    Returns:
        list[str]: Ordered list of class names inferred from folder names.
    """
    dataset = datasets.ImageFolder(training_dir)
    return dataset.classes


def infer_image(
    model: torch.jit.ScriptModule,
    image_path: str,
    transform: transforms.Compose,
    classes: list[str],
    device: torch.device,
) -> list[tuple[str, float]]:
    """
    Run model inference on a single image and return top-5 predictions.

    Args:
        model (torch.jit.ScriptModule): Loaded TorchScript model.
        image_path (str): Path to the image file to infer.
        transform (transforms.Compose): Preprocessing transforms.
        classes (list[str]): List of class labels.
        device (torch.device): Device for computation.

    Returns:
        list[tuple[str, float]]: Top-5 (label, confidence%) pairs.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, 5)

    top_probs = top_probs.squeeze().tolist()
    top_idxs = top_idxs.squeeze().tolist()

    results: list[tuple[str, float]] = []
    for idx, prob in zip(top_idxs, top_probs):
        label = classes[idx] if idx < len(classes) else f"Class {idx}"
        results.append((label, prob * 100.0))

    return results


def draw_predictions_on_image(
    image_path: str,
    predictions: list[tuple[str, float]],
) -> None | "cv2.Mat":
    """
    Annotate an image with text predictions and return the OpenCV image.

    Args:
        image_path (str): Path to the input image file.
        predictions (list[tuple[str, float]]): List of (label, confidence%) pairs.

    Returns:
        cv2.Mat or None: Annotated image array, or None if image failed to load.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    color = (255, 255, 255)
    thickness = 4
    x_offset = 20
    y_start = 2000
    line_height = 80

    for i, (label, conf) in enumerate(predictions):
        text = f"{i + 1}. {label} ({conf:.2f}%)"
        y = y_start + i * line_height
        cv2.putText(img, text, (x_offset, y), font, font_scale, color, thickness)

    return img


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for batch inference.

    Returns:
        argparse.Namespace: Parsed arguments including model path,
            image directory, training directory, and output directory.
    """
    parser = argparse.ArgumentParser(
        description="Batch image inference and annotation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the TorchScript model file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Directory containing training dataset (ImageFolder format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directory to save annotated images",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point: load model, run inference on all images, and save annotations.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_path, device)
    transform = get_image_transform()
    classes = load_class_names(args.training_dir)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith(exts)
    ]

    total_time = 0.0
    for fname in image_files:
        img_path = os.path.join(args.image_dir, fname)

        start = time.time()
        preds = infer_image(model, img_path, transform, classes, device)
        duration = time.time() - start
        total_time += duration

        print(f"\n{fname}:")
        for idx, (lbl, conf) in enumerate(preds, start=1):
            print(f"  {idx}. {lbl} ({conf:.2f}%)")
        print(f"Inference time: {duration:.2f}s")

        annotated = draw_predictions_on_image(img_path, preds)
        if annotated is not None:
            base, _ = os.path.splitext(fname)
            out_path = os.path.join(
                args.output_dir, f"{base}_annotated.png"
            )
            if cv2.imwrite(out_path, annotated):
                print(f"Saved: {out_path}")
            else:
                print(f"Failed to save: {out_path}")
        else:
            print(f"Skipped invalid image: {fname}")

    if image_files:
        avg = total_time / len(image_files)
        print(f"\nAverage inference time: {avg:.2f}s per image")


if __name__ == "__main__":
    main()
