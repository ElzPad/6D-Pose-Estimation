import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class LineMODDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_dir,
        transforms=None,
    ):
        """
        images_dir: data/linemod_yolo/images/{train|val}
        labels_dir: data/linemod_yolo/labels/{train|val}
        transforms: detection augmentations (image, boxes) -> (image, boxes)
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        self.image_files = sorted(
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if not self.image_files:
            raise RuntimeError(f"No images found in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def _load_boxes(self, label_path):
        boxes = []
        if not os.path.exists(label_path):
            return boxes

        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                boxes.append([int(cls), x, y, w, h])

        return boxes

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(
            self.labels_dir, os.path.splitext(img_name)[0] + ".txt"
        )

        image = Image.open(img_path).convert("RGB")
        boxes = self._load_boxes(label_path)

        if self.transforms is not None:
            image, boxes = self.transforms(image, boxes)

        targets = torch.tensor(boxes, dtype=torch.float32)

        return image, targets
