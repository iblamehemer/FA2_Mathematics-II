"""
Evaluate a YOLOv8 PPE detection model on a dataset.

This script uses the Ultralytics YOLO API to evaluate a trained model on the
validation or test split of a dataset defined in a YAML file.  It prints
precision, recall and mAP metrics to the console and saves predictions in
the run directory.

Example:

    python evaluate.py --weights models/best.pt --data data/ppe.yaml --split val
"""

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model for PPE detection")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to the trained model weights (.pt file)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ppe.yaml",
        help="Path to the dataset YAML configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load model
    model = YOLO(args.weights)
    # Evaluate model
    results = model.val(data=args.data, split=args.split)
    # Print summary metrics
    print(f"\nEvaluation results for {args.weights} on {args.split} split:\n")
    for metric, value in results.metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()