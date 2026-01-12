import os
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
    
    Loads data from linemod_yolo format:
    - images/{split}/ contains RGB images as {obj_id:02d}_{frame_id:04d}.png
    - labels/{split}/ contains YOLO bbox labels (class_id x_center y_center w h) normalized
    - pose_labels/{split}/ contains pose labels (class_id r00..r22 tx ty tz)
    """

    def __init__(self, root_dir, object_id, split="train", split_percentage=0.8):
        self.root_dir = root_dir
        self.split = split
        self.split_percentage = split_percentage  # Not used with YOLO format (split is predefined)

        # Define transforms (will be applied in __getitem__)
        if split == "train":
            self.transform = get_train_transforms(image_size=224)
        else:
            self.transform = get_val_transforms(image_size=224)

        # The buffer will store dictionaries containing raw numpy images and metadata
        self.memory_buffer = []
        self.target_object_ids = []

        # 1. Identify Target Objects from available files
        images_folder = os.path.join(root_dir, "images", split)
        if object_id is None:
            # Scan available object IDs from image filenames
            if os.path.exists(images_folder):
                obj_ids_found = set()
                for fname in os.listdir(images_folder):
                    if fname.endswith(".png"):
                        # Parse object ID from filename: {obj_id:02d}_{frame_id:04d}.png
                        try:
                            obj_id_str = fname.split("_")[0]
                            obj_ids_found.add(int(obj_id_str))
                        except (ValueError, IndexError):
                            continue
                self.target_object_ids = sorted(list(obj_ids_found))
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
        Uses linemod_yolo folder structure.
        """
        str_id_padded = f"{obj_id:02d}"
        
        images_folder = os.path.join(self.root_dir, "images", self.split)
        labels_folder = os.path.join(self.root_dir, "labels", self.split)
        pose_labels_folder = os.path.join(self.root_dir, "pose_labels", self.split)

        if not os.path.exists(images_folder):
            return

        # Find all files for this object
        all_files = []
        for fname in os.listdir(images_folder):
            if fname.endswith(".png") and fname.startswith(f"{str_id_padded}_"):
                all_files.append(fname)
        
        all_files.sort()
        
        # Iterate and load
        for img_name in tqdm(all_files, desc=f"Loading Obj {obj_id}", leave=False):
            img_path = os.path.join(images_folder, img_name)
            
            # Derive label filenames
            base_name = img_name.replace(".png", "")
            label_path = os.path.join(labels_folder, f"{base_name}.txt")
            pose_label_path = os.path.join(pose_labels_folder, f"{base_name}.txt")

            if not os.path.exists(label_path) or not os.path.exists(pose_label_path):
                continue

            # --- RAM INTENSIVE PART ---
            # Load the raw image immediately
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]
            # --------------------------

            # Read YOLO bbox label: class_id x_center y_center w h (normalized)
            with open(label_path, "r") as f:
                label_line = f.readline().strip()
            label_parts = label_line.split()
            if len(label_parts) < 5:
                continue
            
            # Convert normalized YOLO bbox to pixel coordinates
            x_center_norm = float(label_parts[1])
            y_center_norm = float(label_parts[2])
            w_norm = float(label_parts[3])
            h_norm = float(label_parts[4])
            
            x_center = x_center_norm * img_w
            y_center = y_center_norm * img_h
            w = w_norm * img_w
            h = h_norm * img_h
            bbox_center = [x_center, y_center, w, h]

            # Read pose label: class_id r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz
            with open(pose_label_path, "r") as f:
                pose_line = f.readline().strip()
            pose_parts = pose_line.split()
            if len(pose_parts) < 10:  # At least class_id + 9 rotation values
                continue
            
            # Extract rotation matrix values (indices 1-9)
            rot_list = [float(pose_parts[i]) for i in range(1, 10)]

            # Store the RAW image and metadata in memory
            self.memory_buffer.append(
                {
                    "image": image,  # Huge numpy array
                    "bbox": bbox_center,  # List [x_center, y_center, w, h] in pixels
                    "rot_matrix": rot_list,  # List of 9 rotation matrix values
                    "object_id": obj_id,  # Object identity for conditioning
                }
            )

    def __len__(self):
        return len(self.memory_buffer)

    def __getitem__(self, idx):
        """
        Retrieves data from RAM and applies dynamic transformations.
        Returns: (img_tensor, quat_tensor, object_id)
        """
        sample = self.memory_buffer[idx]

        # 1. Get raw image from memory
        image_raw = sample["image"]
        bbox = sample["bbox"]
        object_id = sample["object_id"]

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

        return img_tensor, quat_tensor, object_id
