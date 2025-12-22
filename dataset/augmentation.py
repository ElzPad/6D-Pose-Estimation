import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
import math


class YOLOBaseTransform(object):
    """Base class with shared transformation methods"""

    def _letterbox_resize(self, image, boxes, target_size):
        """Resize with letterboxing to maintain aspect ratio"""
        orig_w, orig_h = image.size

        # Calculate scaling factor
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize image
        image = TF.resize(image, (new_h, new_w))

        # Create new image with padding
        new_image = Image.new("RGB", (target_size, target_size), (114, 114, 114))

        # Calculate padding
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2

        # Paste resized image
        new_image.paste(image, (pad_w, pad_h))

        # Adjust box coordinates for letterboxing
        adjusted_boxes = []
        for box in boxes:
            class_id, x_center, y_center, width, height = box

            # Scale and shift coordinates
            new_x = (x_center * new_w + pad_w) / target_size
            new_y = (y_center * new_h + pad_h) / target_size
            new_w_norm = width * new_w / target_size
            new_h_norm = height * new_h / target_size

            adjusted_boxes.append([class_id, new_x, new_y, new_w_norm, new_h_norm])

        return new_image, adjusted_boxes

    def _normalize_image(self, image):
        """Convert to tensor and normalize"""
        image = TF.to_tensor(image)
        image = TF.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return image


class YOLOAugmentation(YOLOBaseTransform):
    def __init__(
        self,
        image_size=640,
        scale_range=(0.8, 1.2),
        rotation_range=(-15, 15),
        brightness_range=(0.8, 1.2),
        flip_prob=0.5,
    ):
        """
        Data augmentation for YOLO object detection

        Args:
            image_size: Target image size (YOLO typically uses 640)
            scale_range: Range for random scaling (min, max)
            rotation_range: Range for random rotation in degrees (min, max)
            brightness_range: Range for brightness adjustment (min, max)
            flip_prob: Probability of horizontal flip
        """
        self.image_size = image_size
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.flip_prob = flip_prob

        # Color transformations (don't affect bounding boxes)
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness_range,
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
        )

    def __call__(self, image, boxes):
        """
        Apply augmentation to image and bounding boxes

        Args:
            image: PIL Image
            boxes: List of [class_id, x_center, y_center, width, height] in YOLO format
                   All coordinates are normalized (0-1)

        Returns:
            augmented_image: torch.Tensor
            augmented_boxes: List of augmented boxes
        """
        # Convert boxes to list if numpy array
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()

        # Track original image size
        orig_w, orig_h = image.size

        # 1. Random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            boxes = self._flip_boxes(boxes)

        # 2. Random rotation
        angle = random.uniform(*self.rotation_range)
        if abs(angle) > 0.1:
            image, boxes = self._rotate_image_boxes(image, boxes, angle)

        # 3. Random scaling
        scale = random.uniform(*self.scale_range)
        if abs(scale - 1.0) > 0.01:
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = TF.resize(image, (new_h, new_w))
            # Boxes remain normalized, so no adjustment needed

        # 4. Color augmentation (doesn't affect boxes)
        image = self.color_jitter(image)

        # 5. Resize to target size (letterbox to maintain aspect ratio)
        image, boxes = self._letterbox_resize(image, boxes, self.image_size)

        # 6. Convert to tensor and normalize
        image = self._normalize_image(image)

        # Filter out invalid boxes
        boxes = self._filter_boxes(boxes)

        return image, boxes

    def _flip_boxes(self, boxes):
        """Flip bounding boxes horizontally"""
        flipped = []
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            # Flip x_center: new_x = 1 - x
            flipped.append([class_id, 1.0 - x_center, y_center, width, height])
        return flipped

    def _rotate_image_boxes(self, image, boxes, angle):
        """Rotate image and adjust bounding boxes"""
        # Rotate image
        image = TF.rotate(image, angle, fill=0)

        w, h = image.size
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        rotated_boxes = []
        for box in boxes:
            class_id, x_center, y_center, width, height = box

            # Convert to absolute coordinates
            x_abs = x_center * w
            y_abs = y_center * h
            w_abs = width * w
            h_abs = height * h

            # Get corner points
            corners = [
                [x_abs - w_abs / 2, y_abs - h_abs / 2],
                [x_abs + w_abs / 2, y_abs - h_abs / 2],
                [x_abs + w_abs / 2, y_abs + h_abs / 2],
                [x_abs - w_abs / 2, y_abs + h_abs / 2],
            ]

            # Rotate corners around center
            cx, cy = w / 2, h / 2
            rotated_corners = []
            for x, y in corners:
                # Translate to origin
                x_t = x - cx
                y_t = y - cy
                # Rotate
                x_r = x_t * cos_a - y_t * sin_a
                y_r = x_t * sin_a + y_t * cos_a
                # Translate back
                rotated_corners.append([x_r + cx, y_r + cy])

            # Get new bounding box from rotated corners
            xs = [c[0] for c in rotated_corners]
            ys = [c[1] for c in rotated_corners]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            new_x_center = (x_min + x_max) / 2 / w
            new_y_center = (y_min + y_max) / 2 / h
            new_width = (x_max - x_min) / w
            new_height = (y_max - y_min) / h

            rotated_boxes.append(
                [class_id, new_x_center, new_y_center, new_width, new_height]
            )

        return image, rotated_boxes

    def _filter_boxes(self, boxes, min_area=0.001):
        """Remove boxes that are too small or out of bounds"""
        filtered = []
        for box in boxes:
            class_id, x_center, y_center, width, height = box

            # Check if box is within bounds and has reasonable size
            if (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and width * height > min_area
                and width < 1.0
                and height < 1.0
            ):
                filtered.append(box)

        return filtered


class YOLOValidationTransform(YOLOBaseTransform):
    def __init__(self, image_size=640):
        """
        Validation transforms for YOLO (no augmentation, only resize)

        Args:
            image_size: Target image size
        """
        self.image_size = image_size

    def __call__(self, image, boxes):
        """
        Apply validation transforms to image and bounding boxes

        Args:
            image: PIL Image
            boxes: List of [class_id, x_center, y_center, width, height] in YOLO format

        Returns:
            transformed_image: torch.Tensor
            boxes: Adjusted boxes (if letterboxing is applied)
        """
        # Convert boxes to list if numpy array
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()

        # Letterbox resize (same as training for consistency)
        image, boxes = self._letterbox_resize(image, boxes, self.image_size)

        # Convert to tensor and normalize
        image = self._normalize_image(image)

        return image, boxes


def get_train_transforms(image_size=640):
    """Returns augmentation transforms for YOLO training"""
    return YOLOAugmentation(image_size=image_size)


def get_val_transforms(image_size=640):
    """Returns transforms for YOLO validation (no augmentation)"""
    return YOLOValidationTransform(image_size=image_size)


# Example usage:
if __name__ == "__main__":

    # Initialize transforms
    train_transform = get_train_transforms(image_size=640)
    val_transform = get_val_transforms(image_size=640)

    # Load image and annotations
    # image = Image.open("tree.jpg")
    # boxes = [
    #     [0, 0.5, 0.5, 0.3, 0.4],  # class_id, x_center, y_center, width, height
    #     [1, 0.3, 0.3, 0.2, 0.2],
    # ]

    # Apply training augmentation
    # train_image, train_boxes = train_transform(image, boxes)

    # Apply validation transform
    # val_image, val_boxes = val_transform(image, boxes)
