"""Download and convert ArcFace R50 pretrained weights.

Downloads the insightface buffalo_s model pack (ResNet-50 trained on MS1MV2
with ArcFace loss, 5.8M images / 85K identities), extracts the recognition
model weights, and converts them to PyTorch-compatible format.

Usage:
    python download_arcface.py

Output:
    pretrained/arcface_r50_backbone.pth

Requirements:
    pip install onnx numpy
"""

import os
import sys
import zipfile
import shutil
import subprocess

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
OUTPUT_PATH = os.path.join(PRETRAINED_DIR, "arcface_r50_backbone.pth")

# insightface buffalo_s model pack URL
BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
RECOGNITION_MODEL = "w600k_r50.onnx"


def ensure_packages():
    """Install required packages if missing."""
    for pkg in ["onnx", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def download_file(url, dest_path):
    """Download a file with progress display."""
    import urllib.request

    print(f"Downloading: {url}")
    print(f"Saving to:   {dest_path}")

    def progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {pct}% ({mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        print(f"\n  Download complete!")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def find_onnx_in_zip(zip_path, target_name):
    """Extract the target ONNX file from a zip archive."""
    extract_dir = zip_path.replace(".zip", "_extracted")
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the target file in the archive
        for name in zf.namelist():
            if name.endswith(target_name):
                zf.extract(name, extract_dir)
                extracted = os.path.join(extract_dir, name)
                print(f"  Found: {name}")
                return extracted
    return None


def convert_onnx_to_pytorch(onnx_path, output_path):
    """Convert ONNX recognition model weights to PyTorch state_dict."""
    import onnx
    import numpy as np
    import torch

    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)

    # Extract all weight tensors from ONNX initializers
    onnx_weights = {}
    for init in model.graph.initializer:
        name = init.name
        data = np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
        onnx_weights[name] = data

    print(f"  Found {len(onnx_weights)} weight tensors in ONNX model")

    # Identify the weight name prefix (some models use 'body.', 'backbone.', etc.)
    sample_keys = list(onnx_weights.keys())
    prefix = ""
    for key in sample_keys:
        if "layer1" in key:
            # Extract prefix before "layer1"
            idx = key.index("layer1")
            prefix = key[:idx]
            break

    if prefix:
        print(f"  Detected prefix: '{prefix}' (will be stripped)")

    # Map ONNX weights to torchvision ResNet-50 state_dict
    # Skip conv1/bn1 (insightface uses 3x3, torchvision uses 7x7)
    SKIP_LAYERS = {"conv1.", "bn1.", "prelu", "fc.", "features.", "dropout", "bn2."}

    state_dict = {}
    skipped = {"conv1/bn1 (arch diff)": 0, "non-backbone": 0, "unknown": 0}

    for name, weight in onnx_weights.items():
        # Strip prefix
        clean_name = name[len(prefix):] if prefix else name

        # Skip non-backbone layers
        should_skip = False
        for skip in SKIP_LAYERS:
            if clean_name.startswith(skip):
                skipped["non-backbone" if skip in ("fc.", "features.", "dropout", "bn2.", "prelu")
                        else "conv1/bn1 (arch diff)"] += 1
                should_skip = True
                break
        if should_skip:
            continue

        # Only keep layer1-4 (Bottleneck blocks + downsample)
        if not clean_name.startswith("layer"):
            skipped["unknown"] += 1
            continue

        state_dict[clean_name] = torch.from_numpy(weight.copy())

    print(f"  Converted: {len(state_dict)} layers for PyTorch ResNet-50")
    for reason, count in skipped.items():
        if count > 0:
            print(f"  Skipped:   {count} ({reason})")

    # Save
    torch.save(state_dict, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")


def main():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"ArcFace R50 weights already exist: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        resp = input("Re-download and convert? (y/N): ").strip().lower()
        if resp != "y":
            print("Skipped.")
            return

    ensure_packages()

    # Step 1: Download buffalo_s.zip
    zip_path = os.path.join(PRETRAINED_DIR, "buffalo_s.zip")
    if not os.path.exists(zip_path):
        if not download_file(BUFFALO_S_URL, zip_path):
            print("\nFailed to download. Manual instructions:")
            print(f"  1. Download: {BUFFALO_S_URL}")
            print(f"  2. Place in: {PRETRAINED_DIR}/")
            print(f"  3. Re-run this script")
            return
    else:
        print(f"Using existing zip: {zip_path}")

    # Step 2: Extract the recognition ONNX model
    onnx_path = find_onnx_in_zip(zip_path, RECOGNITION_MODEL)
    if not onnx_path:
        print(f"ERROR: {RECOGNITION_MODEL} not found in {zip_path}")
        return

    # Step 3: Convert to PyTorch
    try:
        convert_onnx_to_pytorch(onnx_path, OUTPUT_PATH)
    except Exception as e:
        print(f"Conversion failed: {e}")
        print("Make sure 'onnx' package is installed: pip install onnx")
        return

    # Cleanup
    extract_dir = zip_path.replace(".zip", "_extracted")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"  Cleaned up temporary files")

    print(f"\nDone! To use ArcFace pretrained weights, run:")
    print(f"  python main.py --arcface --binary --num_folds 1")
    print(f"\nNote: ArcFace uses ResNet-50 backbone (2048-dim features)")
    print(f"      vs ImageNet ResNet-18 (512-dim). LSTM input adjusts automatically.")


if __name__ == "__main__":
    main()
