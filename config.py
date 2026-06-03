import os


class Config:
    """Configuration for ResNet-18 pain intensity recognition.

    Path priority:
        1. config YAML file
        2. Environment variable MINTPAIN_ROOT
        3. Default: <project_root>/../dataset/mintpain
    """

    def __init__(self, config_path=None):
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Dataset root: env var > default
        # 默认路径: 项目目录向上两级/dataset/mintpain
        # 适用于: Workspace/ 下同时有 dataset/ 和 ai_projects/ 的结构
        self.mintpain_root = os.environ.get(
            "MINTPAIN_ROOT",
            "/home/featurize/data",
        )
        self.preprocessed_dir = os.path.join(self.mintpain_root, "rgb_preprocessed")
        self.output_dir = os.path.join(self.mintpain_root, "results")

        # Model
        self.num_classes = 5
        self.sequence_length = 5
        self.backbone = "resnet18"
        self.pretrained = True
        self.backbone_frozen = True  # Phase 1: freeze backbone

        # Phase 1: train classifier only (backbone frozen)
        self.phase1_epochs = 10
        self.phase1_lr = 1e-3

        # Phase 2: unfreeze backbone with lower LR
        self.phase2_epochs = 20
        self.phase2_backbone_lr = 5e-5
        self.phase2_classifier_lr = 5e-4
        self.warmup_epochs = 3  # backbone warmup epochs in Phase 2

        # Training (optimized for RTX 3080 10.5GB)
        self.batch_size = 96
        self.patience = 5
        self.lstm_hidden_dim = 256
        self.lstm_num_layers = 1
        self.dropout = 0.5

        # Data & Class Imbalance
        self.num_workers = 8
        self.undersample = True        # 欠采样多数类
        self.class_weight = "inverse"  # "none" | "inverse" | "sqrt_inverse"
        self.num_folds = 0  # 0 = all folds

        # Device
        self.device = "cuda"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _load_config(self, config_path):
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
