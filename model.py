import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18FeatureExtractor(nn.Module):
    """ResNet-18 backbone for per-frame feature extraction.

    Removes the final FC layer, outputs 512-dim features.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

    def forward(self, x):
        """Extract features from a batch of images.

        Args:
            x: (B, C, H, W) image tensor

        Returns:
            features: (B, 512) feature tensor
        """
        features = self.backbone(x)
        return features.flatten(1)


class PainRecognitionModel(nn.Module):
    """ResNet-18 + LSTM for pain intensity recognition.

    Architecture:
        1. ResNet-18 extracts 512-dim features per frame
        2. LSTM processes temporal sequence of frame features
        3. FC classifier outputs pain intensity logits

    Input: (B, T, C, H, W) — batch of T-frame image sequences
    Output: (B, num_classes) — logits
    """

    def __init__(
        self,
        num_classes=5,
        pretrained=True,
        lstm_hidden_dim=256,
        lstm_num_layers=1,
        dropout=0.5,
    ):
        super().__init__()
        self.feature_extractor = ResNet18FeatureExtractor(pretrained=pretrained)
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
