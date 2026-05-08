import os
import torch
import torch.nn as nn
import torchvision.models as models

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError:
    InceptionResnetV1 = None


def load_vggface2_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print(f"[WARNING] VGGFace2 weights not found at: {weights_path}")
        print("[WARNING] Falling back to ImageNet pretrained weights.")
        print("[INFO] Download VGGFace2 weights from: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/")
        print("[INFO] Or use converted PyTorch weights (e.g., from https://github.com/ox-vgg/vgg_face2)")
        return False

    print(f"Loading VGGFace2 weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_state = model.state_dict()
    filtered_dict = {}
    matched = 0
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if k in model_state and v.shape == model_state[k].shape:
            filtered_dict[k] = v
            matched += 1

    if matched == 0:
        print("[WARNING] No matching keys found in VGGFace2 weights. Falling back to ImageNet.")
        return False

    model_state.update(filtered_dict)
    model.load_state_dict(model_state, strict=False)
    print(f"Loaded {matched}/{len(model_state)} layers from VGGFace2 weights.")
    return True


class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=5, bottleneck_dim=4, vggface_weights_path=None, backbone="vgg16_bn"):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "inceptionresnet_vggface2":
            if InceptionResnetV1 is None:
                raise ImportError(
                    "facenet-pytorch is required for VGGFace2 backbone. "
                    "Install with: pip install facenet-pytorch"
                )
            print("Using InceptionResnetV1 backbone pretrained on VGGFace2.")
            self.backbone = InceptionResnetV1(pretrained="vggface2", classify=False)
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Fine-tune the top of the network instead of freezing everything.
            for name, param in self.backbone.named_parameters():
                if name.startswith(("repeat_3", "block8", "last_linear", "last_bn")):
                    param.requires_grad = True
            self.bottleneck = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, bottleneck_dim),
            )
        else:
            vgg = models.vgg16_bn(weights='IMAGENET1K_V1')
            self.features = vgg.features
            for param in self.features.parameters():
                param.requires_grad = False
            self.avgpool = vgg.avgpool

            if vggface_weights_path:
                load_vggface2_weights(self, vggface_weights_path)

            self.bottleneck = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, bottleneck_dim),
            )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x, return_features=False):
        if self.backbone_name == "inceptionresnet_vggface2":
            x = self.backbone(x)
            features = self.bottleneck(x)
        else:
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
    def __init__(self, input_dim=3, hidden=128):
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
    def __init__(self, input_dim=3, hidden=128):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(256, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, 4096)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])
        return self.dropout(out)


class EnsembleEDLM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.dnn1 = StreamDNN1()
        self.dnn2 = StreamDNN2()
        self.dnn3 = StreamDNN3()
        self.fc1 = nn.Linear(4096 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        o1 = self.dnn1(x)
        o2 = self.dnn2(x)
        o3 = self.dnn3(x)
        merged = torch.cat([o1, o2, o3], dim=1)
        out = torch.relu(self.fc1(merged))
        return self.fc2(out)
