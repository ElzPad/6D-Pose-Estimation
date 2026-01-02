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
    A PyTorch Dataset for object's rotation prediction task on LINEMODE dataset.

    This dataset loads RGB images, crops the object using Ground Truth bounding boxes,
    and returns the cropped image tensor along with its 3D rotation represented as a quaternion.

    Data Loading Strategy:
        - Parses 'gt.yml' to find all annotations for the specific object ID (if provided).
        - Pre-loads all processed tensors into memory during initialization.
        - Applies specific transformations (augmentation/normalization) based on the split.
    """

    def __init__(self, root_dir, object_id, split="train", split_percentage=0.8):
        """
        Args:
            root_dir (str): Path to LINEMOD root.
            object_id (str or int, optional): Object ID to train on (e.g., '5'). If None, loads ALL detected objects in root_dir.
            split (str): 'train' or 'test'.
            split_percentage (float): Fraction of images used for training.
        """
        self.root_dir = root_dir
        self.split = split
        self.split_percentage = split_percentage

        # Select transforms based on split
        if split == "train":
            self.transform = get_train_transforms(image_size=224)
        else:
            self.transform = get_val_transforms(image_size=224)

        self.memory_buffer = []

        self.target_object_ids = []

        if object_id is None:
            if os.path.exists(root_dir):
                for d in os.listdir(root_dir):
                    dir_path = os.path.join(root_dir, d)
                    # Check if it's a directory and has gt.yml (valid object folder)
                    if os.path.isdir(dir_path) and os.path.exists(
                        os.path.join(dir_path, "gt.yml")
                    ):
                        try:
                            self.target_object_ids.append(int(d))
                        except ValueError:
                            continue  # Skip non-integer folders
            self.target_object_ids.sort()
            print(
                f"Found {len(self.target_object_ids)} classes: {self.target_object_ids}"
            )
        else:
            # Single object mode
            self.target_object_ids = [int(object_id)]

        # Load Data for each Object
        for obj_id in self.target_object_ids:
            self._load_single_object(obj_id)

        print(f"Total samples loaded in RAM: {len(self.memory_buffer)}")

    def _load_single_object(self, obj_id):
        """
        Helper to load data for a specific object ID and append to memory buffer.
        """
        # (try '05' then '5')
        str_id_padded = f"{obj_id:02d}"
        obj_folder = os.path.join(self.root_dir, str_id_padded)
        if not os.path.exists(obj_folder):
            obj_folder = os.path.join(self.root_dir, str(obj_id))

        rgb_folder = os.path.join(obj_folder, "rgb")
        gt_path = os.path.join(obj_folder, "gt.yml")

        if not os.path.exists(gt_path):
            print(f"Warning: GT file not found for object {obj_id}, skipping.")
            return

        # ground Truth
        with open(gt_path, "r") as f:
            gt_data = yaml.safe_load(f)

        all_indices = sorted([int(k) for k in gt_data.keys()])
        split_cutoff = int(len(all_indices) * self.split_percentage)

        if self.split == "train":
            target_indices = all_indices[:split_cutoff]
        else:
            target_indices = all_indices[split_cutoff:]

        print(f"Loading {len(target_indices)} samples for object {obj_id}...")
        for frame_id in tqdm(target_indices, desc=f"Obj {obj_id}", leave=False):
            # pass specific folder and data for this object to the processor
            data_item = self._process_frame(frame_id, rgb_folder, gt_data, obj_id)
            if data_item is not None:
                self.memory_buffer.append(data_item)

    def _process_frame(self, frame_id, rgb_folder, gt_data, target_obj_id):
        """
        Internal helper to process a single frame.
        Args:
            frame_id (int): Image index.
            rgb_folder (str): Path to this object's RGB folder.
            gt_data (dict): Loaded YAML data for this object.
            target_obj_id (int): The specific ID we are looking for in the GT.
        """
        img_name = f"{frame_id:04d}.png"
        img_path = os.path.join(rgb_folder, img_name)

        if not os.path.exists(img_path):
            return None

        image = cv2.imread(img_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get Annotation
        if frame_id not in gt_data:
            return None

        anns = gt_data[frame_id]
        target_ann = None

        # find the specific object annotation in this frame
        for ann in anns:
            if ann["obj_id"] == target_obj_id:
                target_ann = ann
                break

        if target_ann is None:
            return None

        # Crop & Preprocess using GT Bounding Box
        x, y, w, h = target_ann["obj_bb"]
        bbox_xyxy = [x, y, x + w, y + h]

        img_tensor = self.transform(image, bbox_xyxy)

        # process Rotation (Matrix -> Quaternion)
        rot_list = target_ann["cam_R_m2c"]
        rot_matrix = np.array(rot_list).reshape(3, 3)

        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # (x, y, z, w)

        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

        # q == -q
        if quat_wxyz[0] < 0:
            quat_wxyz *= -1

        quat_tensor = torch.from_numpy(quat_wxyz)

        return (img_tensor, quat_tensor)

    def __len__(self):
        return len(self.memory_buffer)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (img_tensor, quat_tensor)

            - **img_tensor** (torch.Tensor):
                The preprocessed RGB image crop.
                Shape: `(3, 224, 224)`
                Normalization: ImageNet standards (Mean: [0.485...], Std: [0.229...])

            - **quat_tensor** (torch.Tensor):
                The ground truth orientation as a unit quaternion.
                Shape: `(4,)`
                Format: `[w, x, y, z]` (Scalar-first convention)
                Constraint: `w >= 0` (Hemisphere constraint enforced)
        """
        return self.memory_buffer[idx]
