import os
import yaml
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from augmentations.transforms import get_train_transforms, get_val_transforms
from tqdm import tqdm


class LineModRotationDataset(Dataset):
    """
    A PyTorch Dataset for object rotation prediction on the LINEMOD dataset.

    Improvements:
    1. Lazy Loading: Images are loaded in __getitem__ to prevent RAM explosion.
    2. Dynamic Augmentation: Transforms are applied per-epoch, ensuring variation.
    3. Robustness: Added checks for file existence and valid annotations.
    """

    def __init__(self, root_dir, object_id, split="train", split_percentage=0.8):
        """
        Args:
            root_dir (str): Path to LINEMOD root.
            object_id (str or int, optional): Object ID to train on (e.g., '5').
                                              If None, loads ALL detected objects.
            split (str): 'train' or 'test'.
            split_percentage (float): Fraction of images used for training.
        """
        self.root_dir = root_dir
        self.split = split
        self.split_percentage = split_percentage

        # Select transforms based on split
        # Note: We store the transform function to call it later in __getitem__
        if split == "train":
            self.transform = get_train_transforms(image_size=224)
        else:
            self.transform = get_val_transforms(image_size=224)

        # Metadata storage (Lightweight: just paths and numbers)
        self.metadata = []
        self.target_object_ids = []

        # 1. Identify Target Objects
        if object_id is None:
            if os.path.exists(root_dir):
                for d in os.listdir(root_dir):
                    dir_path = os.path.join(root_dir, d)
                    # Check if valid object folder with gt.yml
                    if os.path.isdir(dir_path) and os.path.exists(
                        os.path.join(dir_path, "gt.yml")
                    ):
                        try:
                            self.target_object_ids.append(int(d))
                        except ValueError:
                            continue
            self.target_object_ids.sort()
            print(
                f"Found {len(self.target_object_ids)} classes: {self.target_object_ids}"
            )
        else:
            self.target_object_ids = [int(object_id)]

        # 2. Scan Files & Build Metadata Index
        for obj_id in self.target_object_ids:
            self._scan_object_files(obj_id)

        print(f"[{split.upper()}] Total samples indexed: {len(self.metadata)}")

    def _scan_object_files(self, obj_id):
        """
        Scans directory for a specific object and adds valid frame metadata to self.metadata.
        Does NOT load images into memory.
        """
        # Handle folder naming '05' vs '5'
        str_id_padded = f"{obj_id:02d}"
        obj_folder = os.path.join(self.root_dir, str_id_padded)
        if not os.path.exists(obj_folder):
            obj_folder = os.path.join(self.root_dir, str(obj_id))
            if not os.path.exists(obj_folder):
                print(f"Warning: Folder for object {obj_id} not found.")
                return

        rgb_folder = os.path.join(obj_folder, "rgb")
        gt_path = os.path.join(obj_folder, "gt.yml")

        if not os.path.exists(gt_path):
            print(f"Warning: GT file not found for object {obj_id}, skipping.")
            return

        # Load GT YAML (Small enough to fit in memory)
        with open(gt_path, "r") as f:
            gt_data = yaml.safe_load(f)

        # Split logic
        all_indices = sorted([int(k) for k in gt_data.keys()])
        split_cutoff = int(len(all_indices) * self.split_percentage)

        if self.split == "train":
            target_indices = all_indices[:split_cutoff]
        else:
            target_indices = all_indices[split_cutoff:]

        # Iterate and store metadata
        # We assume every frame in GT has a corresponding image.
        for frame_id in target_indices:
            img_name = f"{frame_id:04d}.png"
            img_path = os.path.join(rgb_folder, img_name)

            # Quick check if file exists (avoids crashes later)
            if not os.path.exists(img_path):
                continue

            # Find annotation for the specific object ID in this frame
            anns = gt_data[frame_id]
            target_ann = None
            for ann in anns:
                if ann["obj_id"] == obj_id:
                    target_ann = ann
                    break

            if target_ann is None:
                continue

            # Extract necessary data for __getitem__
            # LINEMOD GT format: [x, y, w, h] -> Convert to xyxy for transforms
            x, y, w, h = target_ann["obj_bb"]
            bbox_xyxy = [x, y, x + w, y + h]

            rot_list = target_ann["cam_R_m2c"]

            # Store minimal info needed to process this sample later
            meta_item = {
                "img_path": img_path,
                "bbox": bbox_xyxy,
                "rot_matrix_list": rot_list,  # Store as list, convert to numpy later
                "obj_id": obj_id,
            }
            self.metadata.append(meta_item)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        Lazy loading happens here.
        """
        meta = self.metadata[idx]

        # 1. Load Image
        # cv2 loads as BGR, convert to RGB
        image = cv2.imread(meta["img_path"])

        # Robustness check: if image is corrupted, just return the next valid index
        if image is None:
            new_idx = (idx + 1) % len(self.metadata)
            return self.__getitem__(new_idx)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transform & Crop
        # The transform handles the cropping using the bbox and augments the image
        img_tensor = self.transform(image, meta["bbox"])

        # 3. Process Rotation (Matrix -> Quaternion)
        # Convert list back to numpy 3x3
        rot_matrix = np.array(meta["rot_matrix_list"], dtype=np.float32).reshape(3, 3)

        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # Scipy returns (x, y, z, w)

        # Reorder to (w, x, y, z)
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

        # 4. Enforce Hemisphere Constraint (w >= 0)
        # This handles the double cover problem by mapping all q to the upper hemisphere
        if quat_wxyz[0] < 0:
            quat_wxyz *= -1

        quat_tensor = torch.from_numpy(quat_wxyz)

        return img_tensor, quat_tensor
