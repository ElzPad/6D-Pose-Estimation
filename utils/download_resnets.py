#!/usr/bin/env python3

import argparse
import os
import gdown

ROTATION_MODEL_URL = "https://drive.google.com/file/d/1DjPKBkXjz2AFfNr2D4Ho-H6YdWfipmWs/view?usp=sharing"

TRANSLATION_MODEL_URL = "https://drive.google.com/file/d/1Lq786mR-GDeyrmV2BaTTImfXFH3Uh9wA/view?usp=sharing"


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
        "--rotation_dir",
        type=str,
        default="models/rgb/resnetRotation/",
        help="Destination directory for rotation model",
    )
    parser.add_argument(
        "--translation_dir",
        type=str,
        default="models/rgb/resnetTranslation/",
        help="Destination directory for translation model",
    )

    args = parser.parse_args()

    download_model(ROTATION_MODEL_URL, args.rotation_dir, "rotation_model.pth")

    download_model(TRANSLATION_MODEL_URL, args.translation_dir, "translation_model.pth")


if __name__ == "__main__":
    main()
