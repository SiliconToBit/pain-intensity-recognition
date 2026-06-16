"""Download pretrained models for pain intensity recognition.

Supported models:
    arcface   — InsightFace ArcFace R50 (MS1MV2), ONNX format, 512-dim embedding
    vggface2  — VGGFace2 ResNet-18 weights (legacy; current project uses facenet-pytorch instead)

Usage:
    python scripts/download_models.py arcface
    python scripts/download_models.py vggface2
    python scripts/download_models.py all
"""

import argparse
import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"


# ─── Shared Utilities ────────────────────────────────────────────────────────

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_file(url, dest_path):
    """Download a file with progress display.

    Returns True on success, False on failure.
    """
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


def confirm_overwrite(path):
    """Ask user to confirm overwrite if file exists. Returns True if OK to proceed."""
    if not os.path.exists(path):
        return True
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"File already exists: {path} ({size_mb:.1f} MB)")
    resp = input("Re-download? (y/N): ").strip().lower()
    if resp != "y":
        print("Skipped.")
        return False
    os.remove(path)
    return True


# ─── ArcFace R50 ─────────────────────────────────────────────────────────────

ARCFACE_BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE_ZIP_NAME = "buffalo_l.zip"
ARCFACE_ONNX_NAME = "w600k_r50.onnx"


def ensure_packages():
    """Install required packages for ArcFace if missing."""
    for pkg in ["onnx2torch", "onnx"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


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


def download_arcface():
    """Download ArcFace R50 ONNX model from insightface buffalo_l pack."""
    output_path = PRETRAINED_DIR / ARCFACE_ONNX_NAME
    ensure_dir(PRETRAINED_DIR)

    if not confirm_overwrite(output_path):
        return

    ensure_packages()

    # Step 1: Download buffalo_l.zip
    zip_path = PRETRAINED_DIR / ARCFACE_ZIP_NAME
    if not zip_path.exists():
        if not download_file(ARCFACE_BUFFALO_URL, zip_path):
            print("\nFailed to download. Manual instructions:")
            print(f"  1. Download: {ARCFACE_BUFFALO_URL}")
            print(f"  2. Place in: {PRETRAINED_DIR}/")
            print(f"  3. Re-run this script")
            return
    else:
        print(f"Using existing zip: {zip_path}")

    # Step 2: Extract the ONNX model
    if not extract_onnx_from_zip(zip_path, ARCFACE_ONNX_NAME, output_path):
        print(f"ERROR: {ARCFACE_ONNX_NAME} not found in {zip_path}")
        return

    # Step 3: Cleanup zip
    os.remove(zip_path)
    print(f"  Cleaned up {zip_path}")

    print(f"\nDone! ArcFace R50 ONNX model ready: {output_path}")
    print(f"\nTo use ArcFace pretrained weights, run:")
    print(f"  python main.py --arcface --binary --num_folds 1")
    print(f"\nNote: InsightFace R50 outputs 512-dim face embeddings")
    print(f"      Input: 112x112 BGR (handled automatically by the model)")


# ─── VGGFace2 ResNet-18 ──────────────────────────────────────────────────────

VGGFACE2_WEIGHTS_URL = "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/resnet18_ft_weight.pth"
VGGFACE2_WEIGHTS_NAME = "resnet18_vggface2.pth"


def download_vggface2():
    """Download VGGFace2 ResNet-18 pretrained weights (legacy).

    NOTE: The current project uses facenet_pytorch.InceptionResnetV1 for VGGFace2
    (via --vggface2 CLI flag), which auto-downloads its own weights internally.
    This ResNet-18 VGGFace2 model is kept for reference / future use.
    """
    output_path = PRETRAINED_DIR / VGGFACE2_WEIGHTS_NAME
    ensure_dir(PRETRAINED_DIR)

    if not confirm_overwrite(output_path):
        return

    success = download_file(VGGFACE2_WEIGHTS_URL, output_path)

    if success:
        print(f"\nDone! VGGFace2 ResNet-18 weights ready: {output_path}")
        print(f"\nNOTE: This is a legacy model. The current project uses facenet-pytorch")
        print(f"      InceptionResnetV1 for VGGFace2 (via --vggface2 CLI flag).")
        print(f"      To use that instead:")
        print(f"        pip install facenet-pytorch")
        print(f"        python main.py --vggface2")
    else:
        print(f"\nFailed to download weights. You can manually download from:")
        print(f"  {VGGFACE2_WEIGHTS_URL}")
        print(f"  and save to: {output_path}")


# ─── AffectNet ResNet-50 ─────────────────────────────────────────────────────

AFFECTNET_URL = "https://huggingface.co/ElenaRyumina/face_emotion_recognition/resolve/main/FER_static_ResNet50_AffectNet.pt"
AFFECTNET_WEIGHTS_NAME = "FER_static_ResNet50_AffectNet.pt"


def download_affectnet():
    """Download AffectNet pretrained ResNet-50 from ElenaRyumina/face_emotion_recognition.

    Source: https://huggingface.co/ElenaRyumina/face_emotion_recognition
    Model: ResNet-50 trained on AffectNet (7-class facial expression recognition).
    Format: PyTorch state_dict (non-standard key names, mapped at load time).
    """
    output_path = PRETRAINED_DIR / AFFECTNET_WEIGHTS_NAME
    ensure_dir(PRETRAINED_DIR)

    if not confirm_overwrite(output_path):
        return

    success = download_file(AFFECTNET_URL, output_path)

    if success:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nDone! AffectNet ResNet-50 weights ready: {output_path} ({size_mb:.1f} MB)")
        print(f"\nTo use AffectNet pretrained weights, run:")
        print(f"  python main.py --affectnet --binary --num_folds 1")
        print(f"\nNote: Outputs 2048-dim features (ResNet-50)")
    else:
        print(f"\nFailed to download. Manual download:")
        print(f"  {AFFECTNET_URL}")
        print(f"  Save to: {output_path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "arcface": download_arcface,
    "vggface2": download_vggface2,
    "affectnet": download_affectnet,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained models for pain intensity recognition"
    )
    parser.add_argument(
        "model",
        choices=list(DOWNLOADERS.keys()) + ["all"],
        help="Which model to download: arcface, vggface2, or all",
    )
    args = parser.parse_args()

    if args.model == "all":
        for name, downloader in DOWNLOADERS.items():
            print(f"\n{'='*60}")
            print(f"Downloading: {name}")
            print(f"{'='*60}")
            downloader()
    else:
        DOWNLOADERS[args.model]()


if __name__ == "__main__":
    main()
