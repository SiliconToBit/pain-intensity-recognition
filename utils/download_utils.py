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
    """Guide the user to obtain and convert VGGFace pretrained weights.

    Uses original VGGFace (VGG16, Oxford 2015, Lua Torch format) per the EDLM paper.

    Steps:
      1. Download vgg_face_torch.tar.gz (Lua Torch format)
      2. Extract VGG_FACE.t7
      3. Convert to torchvision VGG16 .pth via convert_weights.py
    """
    if os.path.exists(config.vggface_weights_path):
        print(f"VGGFace weights found at: {config.vggface_weights_path}")
        return

    print("=" * 60)
    print("VGGFace weights not found.")
    print("To obtain original VGGFace pretrained weights:")
    print("")
    print("  1. Download Lua Torch weights from:")
    print("     http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz")
    print("")
    print("  2. Extract the archive:")
    print("     tar -xvf vgg_face_torch.tar.gz")
    print("     This produces: VGG_FACE.t7")
    print("")
    print("  3. Convert to PyTorch format:")
    print("     pip install torchfile")
    print(f"     python convert_weights.py -i VGG_FACE.t7 -o {config.vggface_weights_path}")
    print("=" * 60)
    print("Falling back to ImageNet pretrained VGG16 weights.")
