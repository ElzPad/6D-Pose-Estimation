#!/usr/bin/env python3

import argparse
import os
import gdown

YOLO_MODEL_URL = "https://drive.google.com/file/d/1Y8UAsjREYTxjUEieqLIEaduggoX94smU/view?usp=sharing"
ROTATION_MODEL_URL = "https://drive.google.com/file/d/1DjPKBkXjz2AFfNr2D4Ho-H6YdWfipmWs/view?usp=sharing"
TRANSLATION_MODEL_URL = "https://drive.google.com/file/d/1Lq786mR-GDeyrmV2BaTTImfXFH3Uh9wA/view?usp=sharing"
ROTATION_RGBD_MODEL_URL = "https://drive.google.com/file/d/1S9-Gp2yV5MYRdu2rGpR7w52pl0lNMjgq/view?usp=sharing"
TRANSLATION_RGBD_MODEL_URL = "https://drive.google.com/file/d/1SWteLV5pxVx4c5Bd-BrHVpmmHAqYI9Jr/view?usp=sharing"

def download_model(url: str, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"[INFO] File already exists, skipping: {output_path}")
        return

    print(f"[INFO] Downloading {filename} to {output_dir}")
    gdown.download(url, output_path, quiet=False, fuzzy=True)

def main():
    parser = argparse.ArgumentParser(description="Download rotation and translation .pth models from Google Drive")
    parser.add_argument(
        "--yolo_dir",
        type=str,
        default="checkpoints/yolo/weights/",
        help="Destination directory for rotation model",
    )
    parser.add_argument(
        "--rotation_dir",
        type=str,
        default="checkpoints/rgb/resnetRotation",
        help="Destination directory for rotation model",
    )
    parser.add_argument(
        "--translation_dir",
        type=str,
        default="checkpoints/rgb/resnetTranslation",
        help="Destination directory for translation model",
    )
    parser.add_argument(
        "--rotation_rgbd_dir",
        type=str,
        default="checkpoints/rgbd/resnetRotationRGBD",
        help="Destination directory for rotation RGBD model",
    )
    parser.add_argument(
        "--translation_rgbd_dir",
        type=str,
        default="checkpoints/rgbd/resnetTranslationRGBD",
        help="Destination directory for translation RGBD model",
    )

    args = parser.parse_args()

    download_model(YOLO_MODEL_URL, args.yolo_dir, "yolo_model.pt")
    download_model(ROTATION_MODEL_URL, args.rotation_dir, "rotation_model.pth")
    download_model(TRANSLATION_MODEL_URL, args.translation_dir, "translation_model.pth")
    download_model(ROTATION_RGBD_MODEL_URL, args.rotation_rgbd_dir, "rotation_rgbd_model.pth")
    download_model(TRANSLATION_RGBD_MODEL_URL, args.translation_rgbd_dir, "translation_rgbd_model.pth")

if __name__ == "__main__":
    main()
