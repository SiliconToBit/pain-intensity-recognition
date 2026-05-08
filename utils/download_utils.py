import os
import requests
from tqdm import tqdm


def download_file(url, destination, chunk_size=8192):
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded to: {destination}")


def download_pretrained_weights(config):
    if os.path.exists(config.vggface_weights_path):
        print(f"VGGFace2 weights found at: {config.vggface_weights_path}")
        return

    print("=" * 60)
    print("VGGFace2 weights not found.")
    print("To use VGGFace2 pretrained weights (recommended):")
    print("  1. Download from: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/")
    print("  2. Convert to PyTorch format, or use a converted version:")
    print("     - https://github.com/ox-vgg/vgg_face2")
    print("     - https://github.com/cydonia999/VGGFace2-Pytorch")
    print("  3. Place the .pth file at:")
    print(f"     {config.vggface_weights_path}")
    print("=" * 60)
    print("Falling back to ImageNet pretrained VGG16-bn weights.")
