"""
Train a YOLOv8 model for PPE detection.

This script wraps the Ultralytics YOLO API to make it easy to train a model
on a custom dataset.  Use it to fine‑tune a pre‑trained YOLO model on
construction safety data.  The default configuration points at
`data/ppe.yaml` within this repository, but you should update the YAML
file's paths to match your local dataset layout.

Example:

    python train.py --data data/ppe.yaml --model yolov8n.pt --epochs 50 --batch 8 --imgsz 640

After training completes, the best model weights will be saved under
`runs/detect/<experiment_name>/weights/best.pt`.  You can then copy this
file into the `models/` directory and use it with the Streamlit app.
"""

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for PPE detection")
    parser.add_argument(
        "--data",
        type=str,
        default="data/ppe.yaml",
        help="Path to the dataset YAML configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to a pre‑trained YOLO model to fine‑tune (e.g. yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (pixels) for training",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ppe",
        help="Name of the training run (used in output directory names)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load a YOLO model (pre‑trained weights)
    model = YOLO(args.model)
    # Train the model
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, name=args.name)


if __name__ == "__main__":
    main()