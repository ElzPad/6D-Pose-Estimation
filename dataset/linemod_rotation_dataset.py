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
    High-RAM Version: Pre-loads all raw images into memory to avoid Disk I/O.
    Transforms (augmentation) are applied dynamically in __getitem__.
    """

    def __init__(self, root_dir, object_id, split="train", split_percentage=0.8):
        self.root_dir = root_dir
        self.split = split
        self.split_percentage = split_percentage

        # Define transforms (will be applied in __getitem__)
        if split == "train":
            self.transform = get_train_transforms(image_size=224)
        else:
            self.transform = get_val_transforms(image_size=224)

        # The buffer will store dictionaries containing raw numpy images and metadata
        self.memory_buffer = []
        self.target_object_ids = []

        # 1. Identify Target Objects
        if object_id is None:
            if os.path.exists(root_dir):
                for d in os.listdir(root_dir):
                    dir_path = os.path.join(root_dir, d)
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

        # 2. Load Everything into RAM
        print("Pre-loading images into RAM... (This may take a while)")
        for obj_id in self.target_object_ids:
            self._preload_object(obj_id)

        print(
            f"[{split.upper()}] Total samples loaded in RAM: {len(self.memory_buffer)}"
        )

    def _preload_object(self, obj_id):
        """
        Loads images and annotations for a specific object into self.memory_buffer.
        """
        str_id_padded = f"{obj_id:02d}"
        obj_folder = os.path.join(self.root_dir, str_id_padded)
        if not os.path.exists(obj_folder):
            obj_folder = os.path.join(self.root_dir, str(obj_id))

        rgb_folder = os.path.join(obj_folder, "rgb")
        gt_path = os.path.join(obj_folder, "gt.yml")

        if not os.path.exists(gt_path):
            return

        with open(gt_path, "r") as f:
            gt_data = yaml.safe_load(f)

        all_indices = sorted([int(k) for k in gt_data.keys()])
        split_cutoff = int(len(all_indices) * self.split_percentage)

        if self.split == "train":
            target_indices = all_indices[:split_cutoff]
        else:
            target_indices = all_indices[split_cutoff:]

        # Iterate and load
        for frame_id in tqdm(target_indices, desc=f"Loading Obj {obj_id}", leave=False):
            img_name = f"{frame_id:04d}.png"
            img_path = os.path.join(rgb_folder, img_name)

            if not os.path.exists(img_path):
                continue

            # --- RAM INTENSIVE PART ---
            # Load the raw image immediately
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # --------------------------

            # Find annotation
            if frame_id not in gt_data:
                continue

            anns = gt_data[frame_id]
            target_ann = None
            for ann in anns:
                if ann["obj_id"] == obj_id:
                    target_ann = ann
                    break

            if target_ann is None:
                continue

            # Prepare metadata
            x, y, w, h = target_ann["obj_bb"]
            bbox_xyxy = [x, y, x + w, y + h]
            rot_list = target_ann["cam_R_m2c"]

            # Store the RAW image and metadata in memory
            self.memory_buffer.append(
                {
                    "image": image,  # Huge numpy array
                    "bbox": bbox_xyxy,  # List
                    "rot_matrix": rot_list,  # List
                }
            )

    def __len__(self):
        return len(self.memory_buffer)

    def __getitem__(self, idx):
        """
        Retrieves data from RAM and applies dynamic transformations.
        """
        sample = self.memory_buffer[idx]

        # 1. Get raw image from memory
        image_raw = sample["image"]
        bbox = sample["bbox"]

        # 2. Apply Dynamic Transform (Augmentation/Crop/Resize happens here)
        # Because we stored the raw image, this is calculated fresh every epoch.
        img_tensor = self.transform(image_raw, bbox)

        # 3. Process Rotation
        rot_matrix = np.array(sample["rot_matrix"], dtype=np.float32).reshape(3, 3)
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # (x, y, z, w)

        # Reorder to (w, x, y, z)
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

        # Hemisphere constraint
        if quat_wxyz[0] < 0:
            quat_wxyz *= -1

        quat_tensor = torch.from_numpy(quat_wxyz)

        return img_tensor, quat_tensor
