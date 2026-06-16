import os
import multiprocessing


def detect_gpu_vram():
    """Detect GPU VRAM in GB. Returns 0 if no CUDA GPU found."""
    try:
        import torch
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            return vram_bytes / (1024 ** 3)
    except Exception:
        pass
    return 0


class Config:
    """Configuration for pain intensity recognition.

    Path priority:
        1. config YAML file
        2. Environment variable MINTPAIN_ROOT
        3. Default: /home/featurize/data (cloud GPU server)

    GPU auto-tuning: batch_size and num_workers are automatically scaled
    based on detected GPU VRAM and CPU core count at init time.
    """

    def __init__(self, config_path=None, batch_size=None, num_workers=None):
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Dataset root: env var > default
        self.mintpain_root = os.environ.get(
            "MINTPAIN_ROOT",
            "/home/featurize/data",
        )
        self.preprocessed_dir = os.path.join(self.mintpain_root, "rgb_preprocessed")
        self.output_dir = os.path.join(self.mintpain_root, "results")

        # Model
        self.num_classes = 5
        self.sequence_length = 5
        self.slide_step = 2
        self.backbone = "resnet18"
        self.pretrained = True
        self.pretrained_source = "imagenet"  # "imagenet" | "vggface2" | "arcface"
        self.pretrained_weights_path = os.path.join(project_root, "pretrained")
        self.backbone_frozen = True

        # Phase 1: train classifier only (backbone frozen)
        self.phase1_epochs = 10
        self.phase1_lr = 1e-3

        # Phase 2: unfreeze backbone with lower LR
        self.phase2_epochs = 20
        self.phase2_backbone_lr = 5e-5
        self.phase2_classifier_lr = 5e-4
        self.warmup_epochs = 3

        # Training — auto-scaled by GPU VRAM, override via CLI or YAML
        self.batch_size = batch_size
        self.patience = 7
        self.lstm_hidden_dim = 256
        self.lstm_num_layers = 1
        self.dropout = 0.5
        self.use_attention_pooling = False

        # Data & Class Imbalance
        self.num_workers = num_workers
        self.undersample = True
        self.class_weight = "none"  # 欠采样已平衡，不需要额外加权
        self.num_folds = 0

        # Loss function
        self.loss_type = "ce"
        self.focal_gamma = 2.0
        self.focal_alpha = None

        # Task mode
        self.binary_mode = False
        self.single_frame = False

        # Device
        self.device = "cuda"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Auto-tune for detected GPU (only if not explicitly set)
        self._auto_tune()

    def _auto_tune(self):
        """Auto-scale batch_size and num_workers to detected GPU VRAM."""
        vram_gb = detect_gpu_vram()

        # Scale batch_size to VRAM (base: 96 for 10.5 GB, linear scaling)
        if self.batch_size is None:
            if vram_gb > 0:
                # Reserve ~2GB for CUDA context, scale remainder
                usable_gb = max(0, vram_gb - 2)
                self.batch_size = max(16, min(256, int(96 * usable_gb / 8.5)))
            else:
                self.batch_size = 32  # safe CPU fallback

        # Scale num_workers to CPU cores (cap at 16, minimum 2)
        if self.num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.num_workers = max(2, min(16, cpu_count // 2))

    def _load_config(self, config_path):
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def gpu_summary(self):
        """Return a human-readable GPU / training config summary."""
        vram_gb = detect_gpu_vram()
        lines = []
        if vram_gb > 0:
            try:
                import torch
                name = torch.cuda.get_device_name(0)
                lines.append(f"GPU: {name} ({vram_gb:.1f} GB)")
            except Exception:
                lines.append(f"GPU VRAM: {vram_gb:.1f} GB")
        else:
            lines.append("GPU: none (CPU mode)")
        lines.append(f"Batch size: {self.batch_size}  |  Workers: {self.num_workers}")
        lines.append(f"Sequence: {self.sequence_length} frames × {self.slide_step} stride")
        lines.append(f"Backbone: {self.pretrained_source}  |  Loss: {self.loss_type}")
        lines.append(f"Attention pooling: {self.use_attention_pooling}")
        return "\n".join(lines)
