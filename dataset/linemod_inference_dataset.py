import torch
import yaml
import numpy as np
import os
import cv2
from pathlib import Path
from torch.utils.data import Dataset

class LinemodInferenceDataset(Dataset):
    def __init__(self, yolo_root, linemod_orig_root, split="val", img_ext=".png"):
        """
        Args:
            yolo_root (str): Path to your processed 'linemod_yolo' folder.
            linemod_orig_root (str): Path to the original LINEMOD folder containing '01', '02', etc.
            split (str): 'train' or 'val'.
        """
        self.yolo_root = Path(yolo_root)
        self.orig_root = Path(linemod_orig_root)
        
        # Paths to your specific yolo structure
        self.images_dir = self.yolo_root / "images" / split
        self.intrinsics_dir = self.yolo_root / "camera_intrinsics" / split
        
        # Get list of file stems (e.g., "01_0000")
        self.file_ids = sorted([f.stem for f in self.images_dir.glob(f"*{img_ext}")])
        
        # Cache for gt.yml files to avoid re-reading them thousands of times
        # Format: { obj_id (int): { frame_id (int): annotation_dict } }
        self.gt_cache = {}

    def _get_gt_pose(self, obj_id, frame_id):
        """
        Lazy-loads the gt.yml for a specific object and retrieves the pose.
        """
        if obj_id not in self.gt_cache:
            # Construct path: original_root/01/gt.yml
            gt_path = self.orig_root / f"{obj_id:02d}" / "gt.yml"
            if not gt_path.exists():
                # Fallback for unpadded names if necessary
                gt_path = self.orig_root / str(obj_id) / "gt.yml"
                
            if gt_path.exists():
                with open(gt_path, 'r') as f:
                    self.gt_cache[obj_id] = yaml.safe_load(f)
            else:
                self.gt_cache[obj_id] = None # Mark as missing
        
        # Retrieve data
        gt_data = self.gt_cache[obj_id]
        if gt_data and frame_id in gt_data:
            anns = gt_data[frame_id]
            # Standard LINEMOD usually lists the object itself. 
            # We filter just in case multiple objects exist in one frame.
            for ann in anns:
                if ann['obj_id'] == obj_id:
                    R = torch.tensor(ann['cam_R_m2c'], dtype=torch.float32).view(3, 3)
                    t = torch.tensor(ann['cam_t_m2c'], dtype=torch.float32).view(3, 1)
                    return R, t
                    
        return torch.eye(3), torch.zeros(3, 1) # Return identity/zeros if missing

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        filename = self.file_ids[idx]  # e.g., "01_0000"
        
        # --- 1. Parse Filename ---
        # "01_0000" -> Obj 1, Frame 0
        parts = filename.split('_')
        obj_id = int(parts[0])
        frame_id = int(parts[1])

        # --- 2. Load Image (from linemod_yolo) ---
        img_path = self.images_dir / (filename + ".png") # or .jpg
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image missing: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- 3. Load Intrinsics (from linemod_yolo) ---
        # Your format: "572.4114 0.0 325.2611 ..." (9 values flat)
        intrin_path = self.intrinsics_dir / (filename + ".txt")
        with open(intrin_path, "r") as f:
            vals = list(map(float, f.readline().strip().split()))
        K = torch.tensor(vals, dtype=torch.float32).view(3, 3)

        # --- 4. Fetch Ground Truth (from Original LINEMOD) ---
        gt_R, gt_t = self._get_gt_pose(obj_id, frame_id)

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