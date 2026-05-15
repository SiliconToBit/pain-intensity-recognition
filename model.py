import os
import torch
import torch.nn as nn
import torchvision.models as models


def load_vggface_weights(model, weights_path):
    """Load VGGFace weights (converted from .t7 via convert_weights.py) into VGG16.

    The converted .pth contains torchvision VGG16-compatible keys:
    - features.N.weight/bias (13 conv layers)
    - classifier.0/3/6.weight/bias (3 FC layers)

    We copy only the matching keys into the model, leaving non-matching keys
    (e.g., classifier.6 which is 2622-dim vs our 1000-dim ImageNet head)
    to use ImageNet pretrained values as fallback.
    """
    if not weights_path or not os.path.exists(weights_path):
        print(f"[WARNING] VGGFace weights not found at: {weights_path}")
        print("[WARNING] Falling back to ImageNet pretrained weights.")
        print("[INFO] Steps to obtain VGGFace weights:")
        print("  1. Download vgg_face_torch.tar.gz from:")
        print("     http://www.robots.ox.ac.uk/~vgg/software/vgg_face/")
        print("  2. Extract: tar -xvf vgg_face_torch.tar.gz  # → VGG_FACE.t7")
        print("  3. Convert: python convert_weights.py -i VGG_FACE.t7")
        print(f"     -o {weights_path}")
        return False

    print(f"Loading VGGFace weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    model_state = model.state_dict()
    matched = 0
    skipped = 0

    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v
            matched += 1
        else:
            skipped += 1

    if matched == 0:
        print("[WARNING] No matching keys — weights may be in wrong format.")
        print("[WARNING] Run convert_weights.py to convert .t7 to torchvision format.")
        return False

    model.load_state_dict(model_state, strict=False)
    print(f"Loaded {matched}/{matched+skipped} layers from VGGFace weights"
          f" ({skipped} skipped/shape-mismatch).")
    return True


class FeatureExtractor(nn.Module):
    """Standard VGG16 backbone with VGGFace pretrained weights, frozen conv layers.

    Architecture follows the original paper:
    - VGG16 (5 conv blocks) pretrained on VGGFace (Oxford 2015)
    - All convolutional layers completely frozen
    - Bottleneck: 25088 -> 4096 -> ReLU -> Dropout -> 4
    - Classifier: 4 -> num_classes
    """

    def __init__(self, num_classes=5, bottleneck_dim=4, vggface_weights_path=None, backbone="vgg16"):
        super().__init__()
        self.backbone_name = backbone

        # Standard VGG16 (no batch-norm, matching original VGGFace)
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.features = vgg.features
        self.avgpool = vgg.avgpool

        # Strictly freeze ALL convolutional layers (per original paper)
        for param in self.features.parameters():
            param.requires_grad = False

        # Load VGGFace pretrained weights if available
        if vggface_weights_path:
            load_vggface_weights(self, vggface_weights_path)

        # Bottleneck: 25088 -> 4096 -> 4 (matching original paper Figure 3)
        self.bottleneck = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, bottleneck_dim),
        )

        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.bottleneck(x)
        out = self.classifier(features)
        if return_features:
            return features
        return out


class StreamDNN1(nn.Module):
    def __init__(self, input_dim=3, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(256, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, 4096)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])
        return self.dropout(out)


class StreamDNN2(nn.Module):
    """DNN2 stream: 2×Conv1d + BiLSTM(32 hidden units).

    Per original paper: 32 hidden units in BiLSTM (64 output dim).
    """

    def __init__(self, input_dim=3, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(128, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, 4096)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])
        return self.dropout(out)


class StreamDNN3(nn.Module):
    """DNN3 stream: 1×Conv1d + unidirectional LSTM(128 hidden units).

    Per original paper: unidirectional LSTM (not bidirectional) for causal
    temporal modeling — suitable for real-time systems.
    """

    def __init__(self, input_dim=3, hidden=128):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(256, hidden, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden, 4096)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.dropout(out)


class EnsembleEDLM(nn.Module):
    """Three-stream ensemble: DNN1 + DNN2 + DNN3 → concat → direct classification.

    Per original paper: three 4096-dim outputs are concatenated (4096×3) and
    fed directly into a final classification layer — no intermediate FC layers.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.dnn1 = StreamDNN1()
        self.dnn2 = StreamDNN2()
        self.dnn3 = StreamDNN3()
        self.fc = nn.Linear(4096 * 3, num_classes)

    def forward(self, x):
        o1 = self.dnn1(x)
        o2 = self.dnn2(x)
        o3 = self.dnn3(x)
        merged = torch.cat([o1, o2, o3], dim=1)
        return self.fc(merged)
