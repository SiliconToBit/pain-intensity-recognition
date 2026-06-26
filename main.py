import argparse
import os

from config import Config
from train import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Pain intensity recognition with ResNet-18 + LSTM"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to mintpain dataset root (overrides MINTPAIN_ROOT env var)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--num_folds", type=int, default=None, help="Number of LOSO folds (0=all)")
    parser.add_argument("--binary", action="store_true", help="Binary mode: pain (1) vs no-pain (0)")
    parser.add_argument("--single_frame", action="store_true",
                        help="Single-frame mode: no LSTM, classify each frame independently")
    parser.add_argument("--vggface2", action="store_true", help="Use VGGFace2 pretrained weights (facenet-pytorch)")
    parser.add_argument("--arcface", action="store_true", help="Use ArcFace pretrained weights (insightface R50)")
    parser.add_argument("--affectnet", action="store_true",
                        help="Use AffectNet pretrained weights (ResNet-50, facial expression)")
    parser.add_argument("--loss", type=str, default=None, choices=["ce", "corn", "coral", "focal", "weighted_ordinal"],
                        help="Loss function: ce (default), corn (ordinal), coral (rank-consistent ordinal), focal, weighted_ordinal (CE+ordinal penalty)")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="Focal loss gamma (higher = more focus on hard examples)")
    parser.add_argument("--coral_consistency_weight", type=float, default=None,
                        help="CORAL rank-consistency penalty weight (default: 0.05)")
    parser.add_argument("--ordinal_lambda", type=float, default=None,
                        help="WeightedOrdinalCE ordinal distance penalty λ (default: 0.1)")
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help="Label smoothing factor (0=off, 0.1=moderate)")
    parser.add_argument("--classifier_hidden", type=int, default=None,
                        help="Classifier MLP hidden dim (0=single Linear, >0=MLP with BN)")
    parser.add_argument("--attention", action="store_true",
                        help="Use temporal attention pooling over LSTM outputs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto-scaled to GPU VRAM if not set)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (auto-scaled to CPU cores if not set)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    # Set env var before creating Config
    if args.data_root:
        os.environ["MINTPAIN_ROOT"] = args.data_root

    config = Config(
        args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrained_source="affectnet" if args.affectnet else
                          "arcface" if args.arcface else
                          "vggface2" if args.vggface2 else None,
        single_frame=args.single_frame,
    )

    if args.vggface2:
        config.pretrained_source = "vggface2"
    if args.arcface:
        config.pretrained_source = "arcface"
    if args.affectnet:
        config.pretrained_source = "affectnet"

    if args.num_folds is not None:
        config.num_folds = args.num_folds

    if args.binary:
        config.binary_mode = True
        config.num_classes = 2

    if args.single_frame:
        config.single_frame = True

    if args.loss:
        config.loss_type = args.loss
    if args.focal_gamma is not None:
        config.focal_gamma = args.focal_gamma
    if args.coral_consistency_weight is not None:
        config.coral_consistency_weight = args.coral_consistency_weight
    if args.ordinal_lambda is not None:
        config.ordinal_lambda = args.ordinal_lambda
    if args.label_smoothing is not None:
        config.label_smoothing = args.label_smoothing
    if args.classifier_hidden is not None:
        config.classifier_hidden_dim = args.classifier_hidden
    if args.attention:
        config.use_attention_pooling = True

    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum

    if args.seed is not None:
        config.seed = args.seed

    if args.resume:
        print("Resume mode enabled")

    mode_str = "Binary (pain vs no-pain)" if config.binary_mode else f"{config.num_classes}-class"
    frame_str = "Single-frame" if config.single_frame else f"LSTM (T={config.sequence_length})"
    pretrained_str = config.pretrained_source.upper() if config.pretrained else "None"
    print(f"Dataset: {config.preprocessed_dir}")
    print(f"Output:  {config.output_dir}")
    print(f"Model:   {pretrained_str} + {frame_str}  |  Seed: {config.seed}")
    print(f"Task:    {mode_str}  |  LOSO folds: {config.num_folds or 'all'}")
    print(config.gpu_summary())

    train_and_evaluate(config, resume=args.resume)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
