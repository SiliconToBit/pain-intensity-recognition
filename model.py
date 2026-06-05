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

    512-dim face embedding, trained on 3.3M face images.
    Input renormalized from ImageNet stats to [-1, 1] in forward pass.
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
        # Renormalize: ImageNet → [0,1] → VGGFace2 [-1,1]
        x = x * self.img_std + self.img_mean
        x = x * 2.0 - 1.0
        return self.backbone(x)


class ArcFaceR50FeatureExtractor(nn.Module):
    """ResNet-50 backbone with ArcFace pretrained weights.

    Uses insightface's ResNet-50 trained on MS1MV2 (5.8M images, 85K identities)
    with Additive Angular Margin loss. Only layer1-4 weights are loaded (conv1/bn1
    remain ImageNet-initialized due to architecture difference: 3x3 vs 7x7 first conv).

    Output: 2048-dim features.
    """

    def __init__(self, pretrained=True, weights_path=None, **kwargs):
        super().__init__()
        resnet = models.resnet50(weights=None)

        if pretrained:
            if not weights_path or not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"ArcFace weights not found at: {weights_path}\n"
                    f"Run: python download_arcface.py"
                )
            state_dict = torch.load(weights_path, map_location="cpu")
            # Only load compatible layers (layer1-4), skip conv1/bn1 (different arch)
            compatible = {k: v for k, v in state_dict.items()
                         if k.startswith("layer") or k.startswith("downsample")}
            missing, unexpected = resnet.load_state_dict(compatible, strict=False)
            loaded = len(compatible)
            skipped = [k for k in missing if not k.startswith("fc.")]
            print(f"  Loaded ArcFace R50: {loaded} layers from {weights_path}")
            if skipped:
                print(f"  Kept ImageNet init: conv1, bn1 ({len(skipped)} params)")

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

    def forward(self, x):
        return self.backbone(x).flatten(1)


# ─── Backbone Registry ──────────────────────────────────────────────────────

BACKBONE_BUILDERS = {
    "imagenet": lambda **kw: ResNet18FeatureExtractor(pretrained=kw.get("pretrained", True)),
    "vggface2": lambda **kw: FaceNetFeatureExtractor(pretrained=kw.get("pretrained", True)),
    "arcface": lambda **kw: ArcFaceR50FeatureExtractor(
        pretrained=kw.get("pretrained", True),
        weights_path=kw.get("weights_path"),
    ),
}


# ─── Main Model ─────────────────────────────────────────────────────────────

class PainRecognitionModel(nn.Module):
    """Pain intensity recognition: Backbone + LSTM + Classifier.

    Supports three backbone options:
        - "imagenet": ResNet-18 (ImageNet) → 512-dim
        - "vggface2": InceptionResnetV1 (VGGFace2) → 512-dim
        - "arcface":  ResNet-50 (ArcFace/MS1MV2) → 2048-dim

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

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_classes),
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
        classifier_params = list(self.lstm.parameters()) + list(self.classifier.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr, "is_backbone": True},
            {"params": classifier_params, "lr": classifier_lr, "is_backbone": False},
        ]

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, T, C, H, W) image sequence tensor

        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Reshape to (B*T, C, H, W) for ResNet
        x = x.view(B * T, C, H, W)

        # Extract features: (B*T, 512)
        features = self.feature_extractor(x)

        # Reshape to (B, T, 512) for LSTM
        features = features.view(B, T, -1)

        # LSTM: (B, T, hidden_dim)
        lstm_out, _ = self.lstm(features)

        # Use last timestep output: (B, hidden_dim)
        last_out = lstm_out[:, -1, :]

        # Classify: (B, num_classes)
        logits = self.classifier(last_out)
        return logits
