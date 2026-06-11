"""Download VGGFace2 pretrained ResNet-18 weights.

NOTE: This script downloads a VGGFace2-fine-tuned **ResNet-18** model.
The current project uses `facenet_pytorch.InceptionResnetV1` for VGGFace2
(via `--vggface2` CLI flag), which auto-downloads its own weights internally.

This script is kept for reference / future use as an alternative backbone option.
To use the InceptionResnetV1 VGGFace2 model (recommended):
    pip install facenet-pytorch
    python main.py --vggface2

To use this ResNet-18 VGGFace2 model instead, you would need to add a new
backbone class in model.py that loads this .pth file.

Usage:
    python download_vggface2.py

This will download the pretrained weights to ./pretrained/resnet18_vggface2.pth

Sources:
    - VGGFace2 pretrained models from the community
    - ResNet-18 architecture, 8631 identity classes
"""

import os
import sys
import urllib.request
from pathlib import Path

# Direct download URL for VGGFace2 pretrained ResNet-18
# Source: https://github.com/ox-vgg/vgg_face2
WEIGHTS_URL = "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/models/resnet18_ft_weight.pth"

PRETRAINED_DIR = Path(__file__).parent / "pretrained"
WEIGHTS_PATH = PRETRAINED_DIR / "resnet18_vggface2.pth"


def download_file(url, dest_path):
    """Download a file with progress indication."""
    print(f"Downloading VGGFace2 ResNet-18 weights...")
    print(f"  Source: {url}")
    print(f"  Destination: {dest_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        downloaded_mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print(f"\n  Download complete: {dest_path}")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def main():
    # Create pretrained directory
    PRETRAINED_DIR.mkdir(exist_ok=True)

    if WEIGHTS_PATH.exists():
        size_mb = WEIGHTS_PATH.stat().st_size / (1024 * 1024)
        print(f"VGGFace2 weights already exist: {WEIGHTS_PATH} ({size_mb:.1f} MB)")
        response = input("Re-download? (y/N): ").strip().lower()
        if response != "y":
            print("Skipped.")
            return

    success = download_file(WEIGHTS_URL, WEIGHTS_PATH)

    if success:
        print(f"\nTo use VGGFace2 pretrained weights, run:")
        print(f"  python main.py --vggface2 --binary --num_folds 1")
        print(f"\nOr set in config YAML:")
        print(f"  pretrained_source: vggface2")
    else:
        print(f"\nFailed to download weights. You can manually download from:")
        print(f"  {WEIGHTS_URL}")
        print(f"  and save to: {WEIGHTS_PATH}")
        print(f"\nAlternative sources:")
        print(f"  1. https://github.com/ox-vgg/vgg_face2")
        print(f"  2. https://huggingface.co/models?search=vggface2+resnet18")


if __name__ == "__main__":
    main()
