import os
import cv2
import math
import torch
import numpy as np
import yaml
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
    - depths/{split}/ contains depth images with the same filename as RGB
    - labels/{split}/ contains YOLO bbox labels (class_id x_center y_center w h) normalized
    - pose_labels/{split}/ contains pose labels (class_id r00..r22 tx ty tz)

    If return_depth=True, __getitem__ returns:
        (rgb_tensor, depth_mask_tensor, quat_tensor, object_id)
    else (default, backwards compatible):
        (rgb_tensor, quat_tensor, object_id)
    """

    def __init__(
        self,
        root_dir,
        object_id,
        split="train",
        split_percentage=0.8,
        intrinsics_path=None,
        models_info_path=None,
        return_depth: bool = False,
        image_size: int = 224,
        bbox_pad: float = 1.2,
        depth_unit_scale: float = 1000.0,  # LINEMOD depth is typically in mm -> meters
        depth_clip_m: float = 2.0,         # clip depth to [0, 2m] then normalize to [0, 1]
    ):
        self.root_dir = root_dir
        self.split = split
        self.split_percentage = split_percentage

        self.return_depth = return_depth
        self.image_size = int(image_size)
        self.bbox_pad = float(bbox_pad)
        self.depth_unit_scale = float(depth_unit_scale)
        self.depth_clip_m = float(depth_clip_m)

        # Load camera intrinsics (assume all images use same intrinsics)
        if intrinsics_path is None:
            intrinsics_path = os.path.join(root_dir, "camera_intrinsics", split, "0000.yml")
        if os.path.exists(intrinsics_path):
            with open(intrinsics_path, "r") as f:
                K = yaml.safe_load(f)
            self.f_x = K["fx"]
            self.f_y = K["fy"]
            self.c_x = K["cx"]
            self.c_y = K["cy"]
        else:
            self.f_x = self.f_y = 572.4114
            self.c_x = 325.2611
            self.c_y = 242.0489

        # Load object diameters
        if models_info_path is None:
            models_info_path = os.path.join(root_dir, "models/models_info.yml")
        if os.path.exists(models_info_path):
            with open(models_info_path, "r") as f:
                data = yaml.safe_load(f)
            self.diameters = {int(k): v['diameter']/1000 for k, v in data.items()}
        else:
            raise FileNotFoundError(f"Cannot read models info from {models_info_path}")

        # Original RGB-only transforms (kept for backwards compatibility)
        if split == "train":
            self.transform_rgb_only = get_train_transforms(image_size=self.image_size)
        else:
            self.transform_rgb_only = get_val_transforms(image_size=self.image_size)

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
                        try:
                            obj_id_str = fname.split("_")[0]
                            obj_ids_found.add(int(obj_id_str))
                        except (ValueError, IndexError):
                            continue
                self.target_object_ids = sorted(list(obj_ids_found))
            print(f"Found {len(self.target_object_ids)} classes: {self.target_object_ids}")
        else:
            self.target_object_ids = [int(object_id)]

        # 2. Load Everything into RAM
        print("Pre-loading images into RAM... (This may take a while)")
        for obj_id in self.target_object_ids:
            self._preload_object(obj_id)

        print(f"[{split.upper()}] Total samples loaded in RAM: {len(self.memory_buffer)}")

    def _preload_object(self, obj_id):
        """
        Loads images and annotations for a specific object into self.memory_buffer.
        Uses linemod_yolo folder structure.
        """
        str_id_padded = f"{obj_id:02d}"

        images_folder = os.path.join(self.root_dir, "images", self.split)
        depths_folder = os.path.join(self.root_dir, "depths", self.split)
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

            base_name = img_name.replace(".png", "")
            label_path = os.path.join(labels_folder, f"{base_name}.txt")
            pose_label_path = os.path.join(pose_labels_folder, f"{base_name}.txt")

            if not os.path.exists(label_path) or not os.path.exists(pose_label_path):
                continue

            # --- RAM INTENSIVE PART ---
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]

            depth = None
            if self.return_depth:
                depth_path = os.path.join(depths_folder, img_name)
                if os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # keep uint16 if present
                    if depth is not None and depth.ndim == 3:
                        depth = depth[:, :, 0]
                if depth is None:
                    # keep shapes consistent to avoid crashes later
                    depth = np.zeros((img_h, img_w), dtype=np.uint16)
            # --------------------------

            # Read YOLO bbox label: class_id x_center y_center w h (normalized)
            with open(label_path, "r") as f:
                label_line = f.readline().strip()
            label_parts = label_line.split()
            if len(label_parts) < 5:
                continue

            x_center_norm = float(label_parts[1])
            y_center_norm = float(label_parts[2])
            w_norm = float(label_parts[3])
            h_norm = float(label_parts[4])

            x_center = x_center_norm * img_w
            y_center = y_center_norm * img_h
            w = w_norm * img_w
            h = h_norm * img_h
            bbox_center = [x_center, y_center, w, h]

            # Read pose label: class_id r00..r22 tx ty tz
            with open(pose_label_path, "r") as f:
                pose_line = f.readline().strip()
            pose_parts = pose_line.split()
            if len(pose_parts) < 10:
                continue

            rot_list = [float(pose_parts[i]) for i in range(1, 10)]

            item = {
                "image": image,
                "bbox": bbox_center,
                "rot_matrix": rot_list,
                "object_id": obj_id,
            }
            if self.return_depth:
                item["depth"] = depth

            self.memory_buffer.append(item)

    def __len__(self):
        return len(self.memory_buffer)

    @staticmethod
    def _imagenet_normalize(rgb: np.ndarray) -> np.ndarray:
        """rgb: float32 in [0,1], HxWx3"""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (rgb - mean) / std

    def _bbox_to_xyxy(self, bbox, img_w, img_h):
        x_c, y_c, w, h = bbox
        w = w * self.bbox_pad
        h = h * self.bbox_pad
        x1 = int(round(x_c - w / 2.0))
        y1 = int(round(y_c - h / 2.0))
        x2 = int(round(x_c + w / 2.0))
        y2 = int(round(y_c + h / 2.0))

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(1, min(x2, img_w))
        y2 = max(1, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            return 0, 0, img_w, img_h
        return x1, y1, x2, y2

    def _crop_resize_rgbd(self, image_raw: np.ndarray, depth_raw: np.ndarray, bbox):
        """Deterministic crop+resize for aligned RGB-D."""
        img_h, img_w = image_raw.shape[:2]
        x1, y1, x2, y2 = self._bbox_to_xyxy(bbox, img_w, img_h)

        rgb = image_raw[y1:y2, x1:x2]
        depth = depth_raw[y1:y2, x1:x2]

        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # RGB tensor (ImageNet normalized)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = self._imagenet_normalize(rgb)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # (3,H,W)

        # Depth mask tensor (1,H,W) in [0,1], zeros are invalid/background
        depth = depth.astype(np.float32)
        if self.depth_unit_scale > 0:
            depth = depth / self.depth_unit_scale  # mm -> m

        valid = (depth > 0).astype(np.float32)
        depth = np.clip(depth, 0.0, self.depth_clip_m)
        depth_norm = (depth / self.depth_clip_m) * valid  # keep only masked values

        depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).contiguous()  # (1,H,W)
        return rgb_tensor, depth_tensor

    def __getitem__(self, idx):
        sample = self.memory_buffer[idx]

        image_raw = sample["image"]
        bbox = sample["bbox"]
        object_id = sample["object_id"]

        if self.return_depth:
            depth_raw = sample["depth"]
            rgb_tensor, depth_mask_tensor = self._crop_resize_rgbd(image_raw, depth_raw, bbox)
            img_tensor = rgb_tensor
        else:
            # Backwards compatible path (uses your existing augmentation pipeline)
            img_tensor = self.transform_rgb_only(image_raw, bbox)

        rot_matrix = np.array(sample["rot_matrix"], dtype=np.float32).reshape(3, 3)
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # (x, y, z, w)

        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)
        if quat_wxyz[0] < 0:
            quat_wxyz *= -1

        quat_tensor = torch.from_numpy(quat_wxyz)

        if self.return_depth:
            return img_tensor, depth_mask_tensor, quat_tensor, object_id
        return img_tensor, quat_tensor, object_id
