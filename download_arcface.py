"""Download ArcFace R50 pretrained model (ONNX).

Downloads the insightface buffalo_l model pack (ResNet-50 trained on MS1MV2
with ArcFace loss, 5.8M images / 85K identities), extracts the recognition
ONNX model, and cleans up the zip archive.

The ONNX model is loaded at runtime via onnx2torch in model.py.

Usage:
    python download_arcface.py

Output:
    pretrained/w600k_r50.onnx  (166 MB)

Requirements:
    pip install onnx2torch onnx
"""

import os
import sys
import zipfile
import subprocess

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
OUTPUT_PATH = os.path.join(PRETRAINED_DIR, "w600k_r50.onnx")

# insightface buffalo_l model pack URL (contains w600k_r50.onnx)
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ZIP_NAME = "buffalo_l.zip"
TARGET_ONNX = "w600k_r50.onnx"


def ensure_packages():
    """Install required packages if missing."""
    for pkg in ["onnx2torch", "onnx"]:
        try:
            __import__(pkg.replace("-", "_"))
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


def extract_onnx_from_zip(zip_path, target_name, output_path):
    """Extract the target ONNX file from a zip archive to output_path."""
    print(f"Extracting from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(target_name):
                with zf.open(name) as src, open(output_path, "wb") as dst:
                    dst.write(src.read())
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  Extracted: {name} → {output_path} ({size_mb:.1f} MB)")
                return True
    return False


def main():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"ArcFace R50 ONNX model already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        resp = input("Re-download? (y/N): ").strip().lower()
        if resp != "y":
            print("Skipped.")
            return
        os.remove(OUTPUT_PATH)

    ensure_packages()

    # Step 1: Download buffalo_l.zip
    zip_path = os.path.join(PRETRAINED_DIR, ZIP_NAME)
    if not os.path.exists(zip_path):
        if not download_file(BUFFALO_URL, zip_path):
            print("\nFailed to download. Manual instructions:")
            print(f"  1. Download: {BUFFALO_URL}")
            print(f"  2. Place in: {PRETRAINED_DIR}/")
            print(f"  3. Re-run this script")
            return
    else:
        print(f"Using existing zip: {zip_path}")

    # Step 2: Extract the ONNX model
    if not extract_onnx_from_zip(zip_path, TARGET_ONNX, OUTPUT_PATH):
        print(f"ERROR: {TARGET_ONNX} not found in {zip_path}")
        return

    # Step 3: Cleanup zip
    os.remove(zip_path)
    print(f"  Cleaned up {zip_path}")

    print(f"\nDone! ArcFace R50 ONNX model ready: {OUTPUT_PATH}")
    print(f"\nTo use ArcFace pretrained weights, run:")
    print(f"  python main.py --arcface --binary --num_folds 1")
    print(f"\nNote: InsightFace R50 outputs 512-dim face embeddings")
    print(f"      Input: 112x112 BGR (handled automatically by the model)")


if __name__ == "__main__":
    main()
