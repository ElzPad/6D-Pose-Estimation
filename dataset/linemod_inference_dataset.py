import torch
import numpy as np
import os
import cv2
from pathlib import Path
from torch.utils.data import Dataset

class LinemodInferenceDataset(Dataset):
    def __init__(self, yolo_root, split="val", img_ext=".png"):
        """
        Inference dataset that loads all data from linemod_yolo folder structure.
        
        Args:
            yolo_root (str): Path to 'linemod_yolo' folder containing images/, labels/, 
                             pose_labels/, and camera_intrinsics/.
            split (str): 'train' or 'val'.
            img_ext (str): Image file extension.
        """
        self.yolo_root = Path(yolo_root)
        
        # Paths to linemod_yolo structure
        self.images_dir = self.yolo_root / "images" / split
        self.intrinsics_dir = self.yolo_root / "camera_intrinsics" / split
        self.pose_labels_dir = self.yolo_root / "pose_labels" / split
        
        # Get list of file stems (e.g., "01_0000")
        self.file_ids = sorted([f.stem for f in self.images_dir.glob(f"*{img_ext}")])

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        filename = self.file_ids[idx]  # e.g., "01_0000"
        
        # --- 1. Parse Filename ---
        # "01_0000" -> Obj 1, Frame 0
        parts = filename.split('_')
        obj_id = int(parts[0])

        # --- 2. Load Image ---
        img_path = self.images_dir / (filename + ".png")
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image missing: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- 3. Load Intrinsics ---
        # Format: "572.4114 0.0 325.2611 ..." (9 values flat, row-major 3x3)
        intrin_path = self.intrinsics_dir / (filename + ".txt")
        with open(intrin_path, "r") as f:
            vals = list(map(float, f.readline().strip().split()))
        K = torch.tensor(vals, dtype=torch.float32).view(3, 3)

        # --- 4. Load Ground Truth Pose from pose_labels ---
        # Format: class_id r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz
        # Note: Each file contains only the target object's pose (one line)
        pose_path = self.pose_labels_dir / (filename + ".txt")
        with open(pose_path, "r") as f:
            pose_vals = f.readline().strip().split()
        
        # Extract rotation matrix (indices 1-9) and translation (indices 10-12)
        rot_vals = [float(pose_vals[i]) for i in range(1, 10)]
        trans_vals = [float(pose_vals[i]) for i in range(10, 13)]
        
        gt_R = torch.tensor(rot_vals, dtype=torch.float32).view(3, 3)
        gt_t = torch.tensor(trans_vals, dtype=torch.float32).view(3, 1)

        # --- 5. Prepare Output ---
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            "image_id": filename,       # "01_0000"
            "intrinsics": K,            # (3, 3)
            "gt_class_id": obj_id,      # Integer ID (1, 2, etc.)
            "gt_R": gt_R,               # (3, 3)
            "gt_t": gt_t                # (3, 1)
        }

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))