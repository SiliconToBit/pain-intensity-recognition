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
    args = parser.parse_args()

    # Set env var before creating Config
    if args.data_root:
        os.environ["MINTPAIN_ROOT"] = args.data_root

    config = Config(args.config)

    if args.num_folds is not None:
        config.num_folds = args.num_folds

    if args.resume:
        print("Resume mode enabled")

    print(f"Dataset: {config.preprocessed_dir}")
    print(f"Output:  {config.output_dir}")
    print(f"Model:   {config.backbone} + LSTM")
    print(f"Classes: {config.num_classes}")
    print(f"LOSO folds: {config.num_folds or 'all'}")

    train_and_evaluate(config, resume=args.resume)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
