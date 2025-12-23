import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Verify that training and evaluation sets do not overlap.")
    parser.add_argument("--dataset_path", type=str, default="data/linemod_yolo", help="Path to the YOLO dataset.")
   
    return parser.parse_args()

def main(path):
    root = Path(path)
    train_imgs = {p.name for p in (root/"images/train").glob("*") if p.suffix.lower() in [".jpg",".png",".jpeg"]}
    val_imgs   = {p.name for p in (root/"images/val").glob("*")   if p.suffix.lower() in [".jpg",".png",".jpeg"]}

    overlap = len(train_imgs & val_imgs)

    print("Training set size:", len(train_imgs))
    print("Valuation set size:", len(val_imgs))
    print("Overlap:", overlap)
    print("\tOverlap examples:", list(train_imgs & val_imgs)[:5])

    if overlap > 0:
        print("\nDATA LEAKAGE. Check your dataset: training and evaluation sets overlap!\n")

if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_path)