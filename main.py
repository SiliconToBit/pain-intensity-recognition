import argparse
import os
import sys

from config import Config
from feature_extraction import extract_features
from train import train_and_evaluate
from utils.download_utils import download_pretrained_weights


def main():
    parser = argparse.ArgumentParser(description="EDLM MIntPAIN reproduce pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()

    config = Config(args.config)

    if args.resume:
        print("🔄 Resume mode enabled - will skip completed folds and load checkpoints")

    if not args.skip_extraction:
        print("Starting feature extraction (fine-tuning + PCA)...")
        download_pretrained_weights(config)
        extract_features(config, resume=args.resume)

    if not args.skip_train:
        print("Starting ensemble training and evaluation...")
        train_and_evaluate(config, resume=args.resume)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
