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
    parser.add_argument("--vggface2", action="store_true", help="Use VGGFace2 pretrained weights (facenet-pytorch)")
    parser.add_argument("--arcface", action="store_true", help="Use ArcFace pretrained weights (insightface R50)")
    parser.add_argument("--loss", type=str, default=None, choices=["ce", "corn", "focal"],
                        help="Loss function: ce (default), corn (ordinal regression), focal")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="Focal loss gamma (higher = more focus on hard examples)")
    parser.add_argument("--attention", action="store_true",
                        help="Use temporal attention pooling over LSTM outputs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto-scaled to GPU VRAM if not set)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (auto-scaled to CPU cores if not set)")
    args = parser.parse_args()

    # Set env var before creating Config
    if args.data_root:
        os.environ["MINTPAIN_ROOT"] = args.data_root

    config = Config(
        args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.vggface2:
        config.pretrained_source = "vggface2"
    if args.arcface:
        config.pretrained_source = "arcface"

    if args.num_folds is not None:
        config.num_folds = args.num_folds

    if args.binary:
        config.binary_mode = True
        config.num_classes = 2

    if args.loss:
        config.loss_type = args.loss
    if args.focal_gamma is not None:
        config.focal_gamma = args.focal_gamma
    if args.attention:
        config.use_attention_pooling = True

    if args.resume:
        print("Resume mode enabled")

    mode_str = "Binary (pain vs no-pain)" if config.binary_mode else f"{config.num_classes}-class"
    pretrained_str = config.pretrained_source.upper() if config.pretrained else "None"
    print(f"Dataset: {config.preprocessed_dir}")
    print(f"Output:  {config.output_dir}")
    print(f"Model:   {config.backbone} + LSTM  |  Pretrained: {pretrained_str}")
    print(f"Task:    {mode_str}  |  LOSO folds: {config.num_folds or 'all'}")
    print(config.gpu_summary())

    train_and_evaluate(config, resume=args.resume)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
