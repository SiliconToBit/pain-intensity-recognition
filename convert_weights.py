"""
Convert original VGGFace Lua Torch (.t7) weights to torchvision VGG16 format.

Usage:
    1. Clone prlz77/vgg-face.pytorch and download weights:
       git clone https://github.com/prlz77/vgg-face.pytorch /tmp/vgg-face.pytorch
       cd /tmp/vgg-face.pytorch
       bash pretrained/vgg_face.sh          # downloads → pretrained/VGG_FACE.t7

    2. Run this conversion:
       python convert_weights.py \
           --input /tmp/vgg-face.pytorch/pretrained/VGG_FACE.t7 \
           --output /home/featurize/work/dataset/mintpain/weights/vgg_face_dag.pth
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models


def load_t7_weights(t7_path):
    """Load weights from a Lua Torch .t7 file using torchfile."""
    try:
        import torchfile
    except ImportError:
        print("Error: 'torchfile' package is required. Install with:")
        print("  pip install torchfile")
        sys.exit(1)

    if not os.path.exists(t7_path):
        print(f"Error: .t7 file not found at: {t7_path}")
        print("Download it first:")
        print("  wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz")
        print("  tar -xvf vgg_face_torch.tar.gz  # → VGG_FACE.t7")
        sys.exit(1)

    print(f"Loading Lua Torch weights from: {t7_path}")
    t7_model = torchfile.load(t7_path)
    print(f"Loaded T7 model with {len(t7_model.modules)} modules.")
    return t7_model


def convert_t7_to_torchvision(t7_model):
    """
    Map VGG-Face T7 weights to torchvision VGG16 state_dict keys.

    T7 layer order (VGG-16 conv blocks):
        conv_1_1, conv_1_2, maxpool,
        conv_2_1, conv_2_2, maxpool,
        conv_3_1, conv_3_2, conv_3_3, maxpool,
        conv_4_1, conv_4_2, conv_4_3, maxpool,
        conv_5_1, conv_5_2, conv_5_3, maxpool,
        fc6 (4096), fc7 (4096), fc8 (2622)

    Torchvision VGG16 feature indices (conv layers only):
        features.0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28
    """
    # Conv layer index mapping: T7 conv order → torchvision features index
    conv_mapping = [
        ("features.0", 0),   # conv_1_1
        ("features.2", 1),   # conv_1_2
        ("features.5", 2),   # conv_2_1
        ("features.7", 3),   # conv_2_2
        ("features.10", 4),  # conv_3_1
        ("features.12", 5),  # conv_3_2
        ("features.14", 6),  # conv_3_3
        ("features.17", 7),  # conv_4_1
        ("features.19", 8),  # conv_4_2
        ("features.21", 9),  # conv_4_3
        ("features.24", 10), # conv_5_1
        ("features.26", 11), # conv_5_2
        ("features.28", 12), # conv_5_3
    ]

    # FC layer index mapping (comes after the 13 conv layers and 5 maxpool layers)
    # In T7 model.modules, the conv layers with weights are interspersed with
    # ReLU and MaxPool layers. We need to iterate and match by order.
    #
    # T7 model.modules contains (in order): conv, relu, conv, relu, maxpool, ...
    # We only care about modules that have .weight attribute.

    state_dict = {}

    # Collect layers that have weights from T7
    t7_layers = []
    for i, module in enumerate(t7_model.modules):
        if hasattr(module, 'weight') and module.weight is not None:
            t7_layers.append(module)
        elif hasattr(module, 'weight'):
            # Some wrappers, skip
            pass

    print(f"Found {len(t7_layers)} layers with weights in T7 model.")

    if len(t7_layers) < 16:
        print(f"Error: expected at least 16 layers (13 conv + 3 fc), got {len(t7_layers)}.")
        sys.exit(1)

    # Map 13 conv layers
    for i in range(13):
        tv_name, t7_idx = conv_mapping[i]
        layer = t7_layers[t7_idx]
        w = torch.tensor(layer.weight)
        b = torch.tensor(layer.bias) if layer.bias is not None else torch.zeros(w.size(0))
        state_dict[tv_name + ".weight"] = w
        state_dict[tv_name + ".bias"] = b
        print(f"  {tv_name} ← T7 layer {t7_idx}  shape={list(w.shape)}")

    # Map 3 FC layers (they come after the 13 conv layers)
    fc_names = ["classifier.0", "classifier.3", "classifier.6"]
    for i, fc_name in enumerate(fc_names):
        layer = t7_layers[13 + i]
        w = torch.tensor(layer.weight)
        b = torch.tensor(layer.bias) if layer.bias is not None else torch.zeros(w.size(0))
        state_dict[fc_name + ".weight"] = w
        state_dict[fc_name + ".bias"] = b
        print(f"  {fc_name} ← T7 layer {13 + i}  shape={list(w.shape)}")

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert VGGFace .t7 to torchvision VGG16 .pth")
    parser.add_argument("--input", "-i", required=True, help="Path to VGG_FACE.t7")
    parser.add_argument("--output", "-o", required=True, help="Output path for vgg_face_dag.pth")
    args = parser.parse_args()

    t7_model = load_t7_weights(args.input)
    state_dict = convert_t7_to_torchvision(t7_model)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"\nSaved torchvision-compatible weights to: {args.output}")
    print("Done! You can now use vgg_face_dag.pth with model.py.")


if __name__ == "__main__":
    main()
