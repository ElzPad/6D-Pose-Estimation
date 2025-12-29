import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np

class LinemodDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_ext='.jpg'):
        """
        Args:
            root_dir (str): Path to the 'linemod_yolo' folder.
            split (str): 'train' or 'val'.
            transform (callable, optional): Optional transform to be applied on a sample.
            img_ext (str): Extension of your images (e.g., .jpg or .png).
        """
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_ext = img_ext
        
        # Define paths based on your provided structure
        self.images_dir = self.root / 'images' / split
        self.labels_dir = self.root / 'labels' / split
        self.intrinsics_dir = self.root / 'camera_intrinsics' / split
        
        # Pre-load all filenames (stems) to ensure corresponding files exist
        # We filter for files that actually have images
        self.file_ids = [
            f.stem for f in self.images_dir.glob(f'*{img_ext}')
        ]
        
        # Sort them to ensure deterministic ordering across runs
        self.file_ids.sort()

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # --- 1. Load Image ---
        img_path = self.images_dir / (file_id + self.img_ext)
        # Load as RGB (OpenCV loads BGR by default)
        image = cv2.imread(str(img_path))
        if image is None:
             raise FileNotFoundError(f"Image not found or unable to read at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- 2. Load Labels (YOLO format: class_id xc yc w h) ---
        label_path = self.labels_dir / (file_id + '.txt')
        
        boxes = []
        labels = []
        
        if label_path.exists():
            # Load raw data with ndmin=2. 
            # This ensures shape is always (N, 5), even if there is only 1 object.
            try:
                data = np.loadtxt(str(label_path), ndmin=2)
                if data.size > 0:
                    labels = data[:, 0]      # First column is Class ID
                    boxes = data[:, 1:]      # Remaining 4 columns are Coordinates (xc, yc, w, h)
            except Exception:
                # Handle empty files gracefully
                pass
            
        # --- 3. Load Camera Intrinsics ---
        # Format: One line, space separated, 9 values
        intrin_path = self.intrinsics_dir / (file_id + '.txt')
        
        if not intrin_path.exists():
             raise FileNotFoundError(f"Intrinsic file not found at {intrin_path}")

        # Read the single line
        with open(intrin_path, 'r') as f:
            line = f.readline().strip()
            # Parse space-separated values
            values = list(map(float, line.split()))
            
        # Convert to 3x3 Tensor
        intrinsics = torch.tensor(values, dtype=torch.float32).view(3, 3)

        # --- 4. Prepare Output ---
        # Convert image to tensor (C, H, W) and normalize to 0-1
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),  # (N, 4)
            "labels": torch.tensor(labels, dtype=torch.long),   # (N,) INTEGER type for classes
            "intrinsics": intrinsics,                           # (3, 3)
            "image_id": file_id
        }

        # --- 5. Optional Transforms ---
        # Place your augmentation logic here if needed
        if self.transform:
             # Note: Ensure your transform function handles 'target' structure correctly
             image, target = self.transform(image, target)

        return image, target