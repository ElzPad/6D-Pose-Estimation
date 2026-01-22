import os
import cv2
import torch
import numpy as np
import yaml
from torch.utils.data import Dataset
from augmentations.transforms import get_train_translation_transforms, get_val_translation_transforms
from tqdm import tqdm

class LineModTranslationDataset(Dataset):
    """
    Loads data from linemod_yolo format for translation regression:
    - images/{split}/ contains RGB images as {obj_id:02d}_{frame_id:04d}.png
    - labels/{split}/ contains YOLO bbox labels (class_id x_center y_center w h) normalized
    - pose_labels/{split}/ contains pose labels (class_id r00..r22 tx ty tz)
    Returns: (img_tensor, object_id, bbox_tensor, diameter, translation_tensor)
    """
    def __init__(self, root_dir, object_id=None, split="train", split_percentage=0.8,
                 intrinsics_path=None, models_info_path=None):
        self.root_dir = root_dir
        self.split = split
        self.split_percentage = split_percentage

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
            self.diameters = {int(k): v['diameter'] for k, v in data.items()}
        else:
            raise FileNotFoundError(f"Cannot read models info from {models_info_path}")

        # Define transforms
        if split == "train":
            self.transform = get_train_translation_transforms(image_size=224)
        else:
            self.transform = get_val_translation_transforms(image_size=224)

        self.memory_buffer = []
        self.target_object_ids = []

        images_folder = os.path.join(root_dir, "images", split)
        if object_id is None:
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

        print("Pre-loading images into RAM... (This may take a while)")
        for obj_id in self.target_object_ids:
            self._preload_object(obj_id)
        print(f"[{split.upper()}] Total samples loaded in RAM: {len(self.memory_buffer)}")

    def _preload_object(self, obj_id):
        str_id_padded = f"{obj_id:02d}"
        images_folder = os.path.join(self.root_dir, "images", self.split)
        labels_folder = os.path.join(self.root_dir, "labels", self.split)
        pose_labels_folder = os.path.join(self.root_dir, "pose_labels", self.split)
        if not os.path.exists(images_folder):
            return
        all_files = []
        for fname in os.listdir(images_folder):
            if fname.endswith(".png") and fname.startswith(f"{str_id_padded}_"):
                all_files.append(fname)
        all_files.sort()
        for img_name in tqdm(all_files, desc=f"Loading Obj {obj_id}", leave=False):
            img_path = os.path.join(images_folder, img_name)
            base_name = img_name.replace(".png", "")
            label_path = os.path.join(labels_folder, f"{base_name}.txt")
            pose_label_path = os.path.join(pose_labels_folder, f"{base_name}.txt")
            if not os.path.exists(label_path) or not os.path.exists(pose_label_path):
                continue
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]
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
            bbox_center_norm = [x_center_norm, y_center_norm, w_norm, h_norm]

            x_center = x_center_norm * img_w
            y_center = y_center_norm * img_h
            w = w_norm * img_w
            h = h_norm * img_h
            bbox_center = [x_center, y_center, w, h]

            # Read pose label: class_id r00 r01 r02 r10 r11 r12 r20 r21 r22 tx ty tz
            with open(pose_label_path, "r") as f:
                pose_line = f.readline().strip()
            pose_parts = pose_line.split()
            if len(pose_parts) < 13:
                continue
            tx = float(pose_parts[10])
            ty = float(pose_parts[11])
            tz = float(pose_parts[12])
            translation = [tx, ty, tz]
            diameter = self.diameters.get(obj_id, 0.2)
            self.memory_buffer.append({
                "image": image,
                "bbox": bbox_center_norm,
                "object_id": obj_id,
                "diameter": diameter,
                "translation": translation
            })

    def __len__(self):
        return len(self.memory_buffer)

    def __getitem__(self, idx):
        sample = self.memory_buffer[idx]
        image_raw = sample["image"]
        bbox = sample["bbox"]
        object_id = sample["object_id"]
        diameter = sample["diameter"]
        translation = sample["translation"]
        img_tensor = self.transform(image_raw)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        translation_tensor = torch.tensor(translation, dtype=torch.float32) / 1000.0  # Convert mm to meters
        return img_tensor, object_id, bbox_tensor, torch.tensor(diameter, dtype=torch.float32), translation_tensor
