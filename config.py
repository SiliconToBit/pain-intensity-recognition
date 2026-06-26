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

    def __init__(self, config_path=None, batch_size=None, num_workers=None,
                 pretrained_source=None, single_frame=False):
        project_root = os.path.dirname(os.path.abspath(__file__))

        # Dataset root: env var > default
        self.mintpain_root = os.environ.get(
            "MINTPAIN_ROOT",
            "/home/featurize/data",
        )
        self.preprocessed_dir = os.path.join(self.mintpain_root, "rgb_preprocessed")
        self.output_dir = os.path.join(self.mintpain_root, "results")

        # Reproducibility
        # True = fully reproducible (cudnn.deterministic, slightly slower);
        # False = cudnn.benchmark acceleration (faster but non-deterministic).
        self.seed = 42
        self.deterministic = True

        # Model
        self.num_classes = 5
        self.sequence_length = 5
        self.num_windows_per_sweep = 3  # K: uniform windows per sweep (replaces slide_step)
        self.slide_step = 2             # legacy: only used when num_windows_per_sweep=0
        self.pretrained = True
        self.pretrained_source = pretrained_source or "imagenet"  # "imagenet" | "vggface2" | "arcface" | "affectnet"
        self.pretrained_weights_path = os.path.join(project_root, "pretrained")

        # Pretrained weight filenames (override via YAML if needed)
        self.arcface_weights_file = "w600k_r50.onnx"
        self.affectnet_weights_file = "FER_static_ResNet50_AffectNet.pt"

        # Phase 1: train classifier only (backbone frozen)
        self.phase1_epochs = 10
        self.phase1_lr = 1e-3

        # Phase 2: unfreeze backbone with lower LR
        self.phase2_epochs = 20
        self.phase2_backbone_lr = 5e-5
        self.phase2_classifier_lr = 5e-4
        self.warmup_epochs = 3

        # Learning-rate scheduler (ReduceLROnPlateau, applied in both phases)
        self.lr_scheduler_factor = 0.5
        self.lr_scheduler_patience = 2

        # Training — auto-scaled by GPU VRAM, override via CLI or YAML
        self.batch_size = batch_size
        self.gradient_accumulation_steps = 1  # simulate larger batches (1=no accumulation)
        self.patience = 7
        self.lstm_hidden_dim = 256
        self.lstm_num_layers = 1
        self.dropout = 0.5
        self.use_attention_pooling = False
        self.classifier_hidden_dim = 0   # 0=single Linear, >0=MLP hidden dim

        # Data & Class Imbalance
        self.num_workers = num_workers
        self.undersample = True
        self.class_weight = "none"  # 欠采样已平衡，不需要额外加权
        self.num_folds = 0

        # Data augmentation (training transforms)
        self.aug_scale = (0.8, 1.0)            # RandomResizedCrop scale range
        self.aug_color_jitter = (0.2, 0.2, 0.2)  # (brightness, contrast, saturation)
        self.aug_rotation_degrees = 5          # RandomAffine rotation ±degrees
        self.aug_translate = 0.03              # RandomAffine translate fraction

        # Loss function
        self.loss_type = "ce"
        self.focal_gamma = 2.0
        self.focal_alpha = None
        self.label_smoothing = 0.0       # 0=off, 0.1=moderate smoothing
        self.coral_consistency_weight = 0.05  # CORAL rank-consistency penalty λ
        self.ordinal_lambda = 0.1            # WeightedOrdinalCE ordinal penalty λ

        # Task mode
        self.binary_mode = False
        self.single_frame = single_frame

        # Device
        self.device = "cuda"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Auto-tune for detected GPU (only if not explicitly set)
        self._auto_tune()

        # Validate mutual exclusivity of undersample and class_weight
        self._validate_imbalance_strategy()

    def _auto_tune(self):
        """Auto-scale batch_size and num_workers to detected GPU VRAM."""
        vram_gb = detect_gpu_vram()

        # Scale batch_size to VRAM
        # Heavier backbones (ResNet-50: arcface, affectnet) need more VRAM per sample
        # so we use a smaller base batch size for them.
        # Single-frame mode uses less VRAM per sample (1 image, no LSTM).
        if self.batch_size is None:
            if vram_gb > 0:
                # Empirical per-sample VRAM usage (full forward+backward+optimizer):
                #   LSTM mode:
                #     ResNet-18 (imagenet/vggface2): ~0.06 GB/sample
                #     ResNet-50 (arcface/affectnet + LSTM): ~0.17 GB/sample
                #   Single-frame mode (no LSTM, 1 image per sample):
                #     ResNet-18: ~0.015 GB/sample
                #     ResNet-50 (arcface/affectnet): ~0.04 GB/sample
                usable_gb = max(0, vram_gb - 1.0)  # 1 GB for CUDA context
                if self.pretrained_source in ("arcface", "affectnet"):
                    per_sample_gb = 0.04 if self.single_frame else 0.17
                    self.batch_size = max(8, min(128, int(usable_gb / per_sample_gb)))
                else:
                    per_sample_gb = 0.015 if self.single_frame else 0.06
                    self.batch_size = max(16, min(256, int(usable_gb / per_sample_gb)))
            else:
                self.batch_size = 32  # safe CPU fallback

        # Scale num_workers to CPU cores (cap at all cores, minimum 2)
        if self.num_workers is None:
            cpu_count = multiprocessing.cpu_count()
            self.num_workers = max(2, cpu_count)

    def _validate_imbalance_strategy(self):
        """Ensure undersample and class_weight are not both active."""
        if self.undersample and self.class_weight != "none":
            import warnings
            warnings.warn(
                f"Both undersample=True and class_weight='{self.class_weight}' are set. "
                f"These strategies are mutually exclusive — class_weight will be disabled. "
                f"Use one: undersample balances data, class_weight re-weights the loss.",
                UserWarning,
                stacklevel=2,
            )
            self.class_weight = "none"

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
        if self.gradient_accumulation_steps > 1:
            eff = self.batch_size * self.gradient_accumulation_steps
            lines.append(f"Gradient accumulation: {self.gradient_accumulation_steps} steps  |  Effective batch: {eff}")
        lines.append(f"Sequence: {self.sequence_length} frames × slide_step={self.slide_step}")
        lines.append(f"Backbone: {self.pretrained_source}  |  Loss: {self.loss_type}")
        lines.append(f"Attention pooling: {self.use_attention_pooling}")
        return "\n".join(lines)

    def to_dict(self):
        """Serialize config to a plain dict (for checkpoint / logging)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
