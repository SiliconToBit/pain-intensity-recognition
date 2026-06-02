import argparse

from config import Config
from train import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Pain intensity recognition with ResNet-18 + LSTM"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--num_folds", type=int, default=None, help="Number of LOSO folds (0=all)")
    args = parser.parse_args()

    config = Config(args.config)

    if args.num_folds is not None:
        config.num_folds = args.num_folds

    if args.resume:
        print("Resume mode enabled")

    print(f"Model: {config.backbone} + LSTM")
    print(f"Classes: {config.num_classes}")
    print(f"Sequence length: {config.sequence_length}")
    print(f"LOSO folds: {config.num_folds or 'all'}")

    train_and_evaluate(config, resume=args.resume)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
