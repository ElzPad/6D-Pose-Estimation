#!/usr/bin/env python3
"""
Download and unzip the DenseFusion preprocessed LINEMOD dataset zip from Google Drive.

Usage:
  python download_linemod.py
"""

from pathlib import Path
import zipfile
import sys

URL = "https://drive.google.com/file/d/1qQ8ZjUI6QauzFsiF8EpaaI2nKFWna_kQ/view?usp=sharing"
OUT_DIR = Path("datasets/linemod")
ZIP_NAME = "Linemod_preprocessed.zip"


def main() -> int:
    # Create output directory (like: !mkdir -p datasets/linemod/)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = OUT_DIR / ZIP_NAME

    # Download (like: !gdown --fuzzy <url> -O Linemod_preprocessed.zip)
    try:
        import gdown
    except ImportError:
        print("Missing dependency: gdown\nInstall it with: pip install gdown", file=sys.stderr)
        return 1

    print(f"Downloading to: {zip_path}")
    downloaded = gdown.download(URL, str(zip_path), quiet=False, fuzzy=True)
    if not downloaded or not zip_path.exists():
        print("Download failed.", file=sys.stderr)
        return 2

    # Unzip (like: !unzip Linemod_preprocessed.zip)
    print(f"Extracting into: {OUT_DIR}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(OUT_DIR)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
