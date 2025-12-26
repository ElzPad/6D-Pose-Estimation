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
    """LineMod dataset for object's rotation prediction task"""

    """
    A PyTorch Dataset for object's rotation prediction task on LINEMODE dataset.
    
    This dataset loads RGB images, crops the object of interest using Ground Truth bounding boxes,
    and returns the cropped image tensor along with its 3D rotation represented as a quaternion.
    
    Data Loading Strategy:
        - Parses 'gt.yml' to find all annotations for the specific object ID.
        - Pre-loads all processed tensors into memory during initialization.
        - Applies specific transformations (augmentation/normalization) based on the split.
    """

    def __init__(self, root_dir, object_id, split="train", split_percentage=0.15):
        """
        Args:
            root_dir (str): Path to LINEMOD root.
            object_id (str or int): Object ID to train on (e.g., '5' or '05').
            split (str): 'train' or 'test'.
            split_percentage (float): Fraction of images used for training.
        """
        self.root_dir = root_dir
        # Store object_id as integer to match the YAML format (e.g., obj_id: 5)
        self.obj_id = int(object_id)

        # Select transforms based on split
        if split == "train":
            self.transform = get_train_transforms(image_size=224)
        else:
            self.transform = get_val_transforms(image_size=224)

        # Setup Paths
        # Try both '05' and '5' folder naming conventions just in case
        str_id_padded = f"{self.obj_id:02d}"  # e.g. "05"
        self.obj_folder = os.path.join(root_dir, str_id_padded)

        if not os.path.exists(self.obj_folder):
            # Fallback to unpadded "5" if "05" doesn't exist
            self.obj_folder = os.path.join(root_dir, str(self.obj_id))

        self.rgb_folder = os.path.join(self.obj_folder, "rgb")
        gt_path = os.path.join(self.obj_folder, "gt.yml")

        # Parse Ground Truth
        print(f"Loading metadata from {gt_path}...")
        with open(gt_path, "r") as f:
            self.gt_data = yaml.safe_load(f)

        # Determine Split Indices
        # The YAML keys are integers (1, 2, 3...) as seen in your snippet
        all_indices = sorted([int(k) for k in self.gt_data.keys()])
        split_cutoff = int(len(all_indices) * split_percentage)

        if split == "train":
            target_indices = all_indices[:split_cutoff]
        else:
            target_indices = all_indices[split_cutoff:]

        # Pre-load Data
        self.memory_buffer = []
        print(
            f"Pre-loading {len(target_indices)} images for object {self.obj_id} ({split} set)..."
        )

        for frame_id in tqdm(target_indices):
            data_item = self._process_frame(frame_id)
            if data_item is not None:
                self.memory_buffer.append(data_item)

        print(f"Successfully loaded {len(self.memory_buffer)} samples into RAM.")

    def _process_frame(self, frame_id):
        """
        Internal helper to process a single frame from disk.
        Returns None if file is missing or annotation is invalid.
        """
        # Load Image
        # Images usually padded: 0001.png, 0002.png...
        img_name = f"{frame_id:04d}.png"
        img_path = os.path.join(self.rgb_folder, img_name)

        if not os.path.exists(img_path):
            return None

        image = cv2.imread(img_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get Annotation
        if frame_id not in self.gt_data:
            return None

        anns = self.gt_data[frame_id]

        target_ann = None
        # Handle list of annotations (some frames have multiple objects)
        for ann in anns:
            # Match the integer ID from YAML (e.g., obj_id: 5)
            if ann["obj_id"] == self.obj_id:
                target_ann = ann
                break

        if target_ann is None:
            return None

        # Crop & Preprocess
        # YAML: obj_bb: [x, y, w, h]
        x, y, w, h = target_ann["obj_bb"]
        bbox_xyxy = [x, y, x + w, y + h]

        # Apply transforms
        img_tensor = self.transform(image, bbox_xyxy)

        # Process Rotation
        rot_list = target_ann["cam_R_m2c"]  # List of 9 floats
        rot_matrix = np.array(rot_list).reshape(3, 3)

        # Matrix -> Quaternion
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # (x, y, z, w)

        # Reorder to (w, x, y, z)
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float32)

        # Hemisphere check (q == -q)
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
