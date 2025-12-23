import argparse
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model on Linemod dataset.")
    parser.add_argument("--dataset_yaml", type=str, default="data\linemod_yolo\data.yml",
                        help="Path to the YOLO dataset config file.")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Pretrained YOLO model name.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for dataloader.")
    parser.add_argument("--freeze", type=int, default=0, help="Number of layers to be frozen during fine-tuning.")
   
    return parser.parse_args()

def main():
    args = parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)

    results = model.train(
        data=args.dataset_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        freeze=args.freeze,
        device=device,

        # Dataset augmentation
        degrees=10.0,      # Rotate +/- 10 degrees
        translate=0.1,     # Translate +/- 10%
        scale=0.5,         # Scale gain +/- 50%
        fliplr=0.0,        # LINEMOD objects are not symmetric in 2D, so be careful with flip!
        hsv_h=0.015,       # Color jitter (Hue)
        hsv_s=0.7,         # Color jitter (Saturation)
        hsv_v=0.4,         # Color jitter (Value)
        mosaic=1.0,        # Probability of mosaic (VERY IMPORTANT)

        # Optional: keep things simple / stable
        pretrained= True,
        optimizer="AdamW",
        lr0=1e-3,
    )

if __name__ == "__main__":
    main()