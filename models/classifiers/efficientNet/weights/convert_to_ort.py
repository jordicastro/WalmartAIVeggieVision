#!/usr/bin/env python3
"""
convert_to_ort.py

Convert a TorchScript model to ONNX and then to an ORT-optimized format.

Usage:
    python convert_to_ort.py \
        --torchscript full-produce-dataset-04_18-efficientnet_b0.zip \
        --onnx produce_model_04_18.onnx \
        --ort produce_model_04_18.ort \
        --opset 13 \
        --batch 1 \
        --height 224 \
        --width 224
"""

import argparse
import torch
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel


def parse_args():
    """
    Parse command-line arguments for model conversion.

    Returns:
        argparse.Namespace: Parsed arguments for paths, opset, and input shape.
    """
    parser = argparse.ArgumentParser(
        description="Convert TorchScript → ONNX → ORT"
    )
    parser.add_argument(
        "--torchscript", required=True,
        help="Path to the TorchScript .zip (traced) file"
    )
    parser.add_argument(
        "--onnx", default="model.onnx",
        help="Path to save the intermediate ONNX model"
    )
    parser.add_argument(
        "--ort", default="model.ort",
        help="Path to save the final ORT format model"
    )
    parser.add_argument(
        "--opset", type=int, default=13,
        help="ONNX opset version for export"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for dummy input"
    )
    parser.add_argument(
        "--height", type=int, default=224,
        help="Height of input images"
    )
    parser.add_argument(
        "--width", type=int, default=224,
        help="Width of input images"
    )
    return parser.parse_args()


def convert_to_onnx(js_path, onnx_path, input_shape, opset):
    """
    Export a TorchScript model to ONNX format.

    Args:
        js_path (str): Path to the TorchScript model file.
        onnx_path (str): Destination path for the ONNX model.
        input_shape (tuple): Shape of dummy input (batch, C, H, W).
        opset (int): ONNX opset version to use.
    """
    model = torch.jit.load(js_path, map_location="cpu")
    model.eval()

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[✓] ONNX model saved to {onnx_path}")


def optimize_onnx_to_ort(onnx_path, ort_path):
    """
    Optimize an ONNX model and save in ORT format.

    Args:
        onnx_path (str): Path to the ONNX model file.
        ort_path (str): Destination path for the ORT model.
    """
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    session_options.optimized_model_filepath = ort_path

    try:
        session_options.optimized_model_format = ort.ModelFormat.ORT
    except AttributeError:
        # Older ORT versions default to ORT format
        pass

    _ = ort.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    print(f"[✓] ORT-optimized model saved to {ort_path}")


def main():
    """
    Main entry point for conversion script.

    Parses arguments, converts to ONNX, then optimizes to ORT.
    """
    args = parse_args()
    shape = (args.batch, 3, args.height, args.width)

    convert_to_onnx(
        args.torchscript,
        args.onnx,
        shape,
        args.opset,
    )
    optimize_onnx_to_ort(
        args.onnx,
        args.ort,
    )
    print("[✓] Conversion complete!")


if __name__ == "__main__":
    main()
