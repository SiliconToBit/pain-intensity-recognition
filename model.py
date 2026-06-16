import os
import torch
import torch.nn as nn
import torchvision.models as models


# ─── Feature Extractors ─────────────────────────────────────────────────────

class ResNet18FeatureExtractor(nn.Module):
    """ResNet-18 backbone pretrained on ImageNet (default)."""

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

    def forward(self, x):
        return self.backbone(x).flatten(1)


class FaceNetFeatureExtractor(nn.Module):
    """InceptionResnetV1 pretrained on VGGFace2 (via facenet-pytorch).

    512-dim face embedding, trained on 3.3M face images at 160x160.
    Input renormalized from ImageNet stats → resize to 160x160 → [-1, 1].
    Weights are auto-downloaded by facenet-pytorch on first use (~100 MB).
    """

    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is required for VGGFace2 pretrained features.\n"
                "Install: pip install facenet-pytorch"
            )
        self.backbone = InceptionResnetV1(
            pretrained='vggface2' if pretrained else None,
            classify=False,
            num_classes=None,
        )
        self.feature_dim = 512
        # Register normalization constants as buffers (move with .to(device))
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Renormalize: ImageNet → [0,1]
        x = x * self.img_std + self.img_mean
        # Resize to 160x160 (native InceptionResnetV1 input size)
        if x.shape[2] != 160 or x.shape[3] != 160:
            x = torch.nn.functional.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
        # Normalize to [-1, 1] for VGGFace2
        x = x * 2.0 - 1.0
        return self.backbone(x)


class ArcFaceR50FeatureExtractor(nn.Module):
    """InsightFace ArcFace R50 (MS1MV2) via onnx2torch.

    Loads the full insightface w600k_r50.onnx model as a PyTorch Module
    using onnx2torch, preserving the complete computation graph for
    fine-tuning. Input: (B, 3, 112, 112) in [-1, 1] (insightface convention).

    Handles the preprocessing conversion from ImageNet normalization
    (applied by the dataset pipeline) to insightface format internally:
        1. Denormalize from ImageNet stats → [0, 1]
        2. Reverse channel order (RGB → BGR)
        3. Normalize to [-1, 1]

    Output: 512-dim face embedding.
    """

    def __init__(self, pretrained=True, weights_path=None, **kwargs):
        super().__init__()
        try:
            from onnx2torch import convert
        except ImportError:
            raise ImportError(
                "onnx2torch is required for ArcFace pretrained features.\n"
                "Install: pip install onnx2torch"
            )

        if pretrained:
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"ArcFace ONNX model not found at: {weights_path}\n"
                    f"Run: python scripts/download_models.py arcface"
                )
            self.backbone = convert(weights_path)
            print(f"  Loaded ArcFace R50 via onnx2torch: {weights_path}")
        else:
            raise ValueError(
                "ArcFace R50 requires pretrained weights (no random init available)."
            )

        self.feature_dim = 512

        # Enable gradient checkpointing to reduce VRAM usage during fine-tuning
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()

        # Register normalization constants as buffers (move with .to(device))
        # Conversion: ImageNet-normalized → [0,1] → BGR → insightface [-1,1]
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # x is ImageNet-normalized: (B, 3, H, W)
        # 1. Denormalize to [0, 1]
        x = x * self.img_std + self.img_mean
        # 2. Resize to 112x112 (insightface input size)
        if x.shape[2] != 112 or x.shape[3] != 112:
            x = torch.nn.functional.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        # 3. RGB → BGR
        x = x[:, [2, 1, 0], :, :]
        # 4. Normalize to [-1, 1]
        x = x * 2.0 - 1.0
        # 5. Forward through insightface model → (B, 512)
        return self.backbone(x)


class AffectNetFeatureExtractor(nn.Module):
    """AffectNet pretrained ResNet-50 via ElenaRyumina/face_emotion_recognition.

    Loads a ResNet-50 state_dict trained on AffectNet (7-class facial expression
    recognition, 1M+ images) with non-standard key names, maps them to
    torchvision ResNet-50 format, and extracts the backbone (output: 2048-dim).

    Weights source: https://huggingface.co/ElenaRyumina/face_emotion_recognition
    File: FER_static_ResNet50_AffectNet.pt

    Key mapping:
        conv_layer_s2_same → conv1
        batch_norm1 → bn1
        batch_norm → bn
        i_downsample → downsample
        fc1 → fc.0  (custom 2-layer head, removed for feature extraction)
        fc2 → fc.2
    """

    def __init__(self, pretrained=True, weights_path=None, **kwargs):
        super().__init__()
        from torchvision.models import resnet50

        if not pretrained:
            raise ValueError(
                "AffectNet R50 requires pretrained weights (no random init available)."
            )

        if not weights_path or not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"AffectNet R50 weights not found at: {weights_path}\n"
                f"Run: python scripts/download_models.py affectnet"
            )

        raw_sd = torch.load(weights_path, map_location="cpu", weights_only=True)

        # Build torchvision-compatible ResNet-50
        resnet = resnet50(weights=None)

        # Key mapping: original naming → torchvision naming
        mapping = {
            "conv_layer_s2_same": "conv1",
            "batch_norm1": "bn1",       # must come before batch_norm→bn
            "batch_norm": "bn",          # layerX.Y.batch_normZ → layerX.Y.bnZ
            "i_downsample": "downsample",
            "fc1.": "fc.0.",
            "fc2.": "fc.2.",
        }

        # Rebuild the custom FC head so all keys can be loaded with strict=True
        resnet.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 7),
        )

        clean_sd = {}
        for k, v in raw_sd.items():
            new_key = k
            for old, new in mapping.items():
                new_key = new_key.replace(old, new)
            clean_sd[new_key] = v

        resnet.load_state_dict(clean_sd, strict=True)

        # Extract backbone (remove FC head)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        # Freeze — AffectNet weights used as frozen features
        for param in self.backbone.parameters():
            param.requires_grad = False

        print(f"  Loaded AffectNet ResNet-50: {weights_path}")

    def forward(self, x):
        return self.backbone(x).flatten(1)  # (B, 2048)


# ─── Backbone Registry ──────────────────────────────────────────────────────

BACKBONE_BUILDERS = {
    "imagenet": lambda **kw: ResNet18FeatureExtractor(pretrained=kw.get("pretrained", True)),
    "vggface2": lambda **kw: FaceNetFeatureExtractor(pretrained=kw.get("pretrained", True)),
    "arcface": lambda **kw: ArcFaceR50FeatureExtractor(
        pretrained=kw.get("pretrained", True),
        weights_path=kw.get("weights_path"),
    ),
    "affectnet": lambda **kw: AffectNetFeatureExtractor(
        pretrained=kw.get("pretrained", True),
        weights_path=kw.get("weights_path"),
    ),
}


# ─── Main Model ─────────────────────────────────────────────────────────────

class TemporalAttentionPooling(nn.Module):
    """Learnable temporal attention pooling over LSTM output sequence.

    Computes softmax-weighted sum of all timestep outputs, allowing the model
    to focus on the most discriminative frames (e.g., peak pain expressions)
    rather than treating all frames equally.

    Input:  (B, T, D) — LSTM output sequence
    Output: (B, D)   — attention-pooled feature vector
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out):
        # lstm_out: (B, T, D)
        weights = self.attention(lstm_out).squeeze(-1)  # (B, T)
        weights = torch.softmax(weights, dim=1)          # (B, T)
        return (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)


class PainRecognitionModel(nn.Module):
    """Pain intensity recognition: Backbone + LSTM + Classifier.

    Supports three backbone options:
        - "imagenet": ResNet-18 (ImageNet) → 512-dim
        - "vggface2": InceptionResnetV1 (VGGFace2) → 512-dim
        - "arcface":  InsightFace R50 (ArcFace/MS1MV2) → 512-dim

    Input:  (B, T, C, H, W) — batch of T-frame image sequences
    Output: (B, num_classes) — logits
    """

    def __init__(
        self,
        num_classes=5,
        pretrained=True,
        pretrained_source="imagenet",
        weights_path=None,
        lstm_hidden_dim=256,
        lstm_num_layers=1,
        dropout=0.5,
        corn_mode=False,
        use_attention_pooling=False,
        single_frame=False,
    ):
        super().__init__()

        builder = BACKBONE_BUILDERS.get(pretrained_source)
        if builder is None:
            raise ValueError(
                f"Unknown pretrained_source: '{pretrained_source}'. "
                f"Choose from: {list(BACKBONE_BUILDERS.keys())}"
            )
        self.feature_extractor = builder(pretrained=pretrained, weights_path=weights_path)
        feature_dim = self.feature_extractor.feature_dim

        # Single-frame mode: skip LSTM, classify each frame independently
        self.single_frame = single_frame
        if not single_frame:
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=dropout if lstm_num_layers > 1 else 0.0,
            )

            # Temporal pooling: attention-weighted or simple mean
            self.use_attention_pooling = use_attention_pooling
            if use_attention_pooling:
                self.attention_pool = TemporalAttentionPooling(lstm_hidden_dim)

        # Corn ordinal regression: output K-1 logits for K classes
        self.corn_mode = corn_mode
        output_dim = num_classes - 1 if corn_mode else num_classes
        classifier_input = feature_dim if single_frame else lstm_hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, output_dim),
        )

    def freeze_backbone(self):
        """Freeze ResNet-18 backbone parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ResNet-18 backbone parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr=1e-4, classifier_lr=1e-3):
        """Get parameter groups with different learning rates.

        Returns:
            list of param groups for optimizer
        """
        backbone_params = list(self.feature_extractor.parameters())
        classifier_params = list(self.classifier.parameters())
        if not self.single_frame:
            classifier_params += list(self.lstm.parameters())
            if self.use_attention_pooling:
                classifier_params += list(self.attention_pool.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr, "is_backbone": True},
            {"params": classifier_params, "lr": classifier_lr, "is_backbone": False},
        ]

    def forward(self, x):
        """Forward pass.

        Args:
            x: single_frame → (B, C, H, W) image tensor
               sequence    → (B, T, C, H, W) image sequence tensor

        Returns:
            logits: (B, num_classes)
        """
        if self.single_frame:
            # Single-frame: (B, C, H, W) → backbone → classifier
            features = self.feature_extractor(x)  # (B, 512)
            return self.classifier(features)

        B, T, C, H, W = x.shape

        # Reshape to (B*T, C, H, W) for ResNet
        x = x.view(B * T, C, H, W)

        # Extract features: (B*T, 512)
        features = self.feature_extractor(x)

        # Reshape to (B, T, 512) for LSTM
        features = features.view(B, T, -1)

        # LSTM: (B, T, hidden_dim)
        lstm_out, _ = self.lstm(features)

        # Temporal pooling: attention-weighted or last-timestep
        if self.use_attention_pooling:
            last_out = self.attention_pool(lstm_out)
        else:
            last_out = lstm_out[:, -1, :]  # use final LSTM timestep

        # Classify: (B, num_classes)
        logits = self.classifier(last_out)
        return logits
