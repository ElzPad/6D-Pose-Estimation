from PIL import Image, ImageFilter
import torch
import random
import numpy as np
from torchvision import transforms


class LineModRotationPredictionTransform(object):
    def __init__(self, image_size=224, padding_factor=0.1, is_train=True):
        """
        Custom Transform class that handles:
        1. Cropping the object using the bounding box.
        2. Applying Safe Augmentations (Photometric only) during training.
        3. Resizing and Normalizing for ResNet.

        Args:
            image_size (int): Input size for the ResNet (default 224).
            padding_factor (float): Context around the object (default 10%).
            is_train (bool): If True, applies random augmentations.
        """
        self.image_size = image_size
        self.padding_factor = padding_factor
        self.is_train = is_train

        # We use photometric transforms that do NOT change the 3D orientation.
        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.25,  # Randomly adjust brightness
                contrast=0.25,  # Randomly adjust contrast
                saturation=0.25,  # Randomly adjust saturation
                hue=0.05,  # Slight hue shift
            )

        # 2. Resizing and Normalization (Common to both)
        # Standard ImageNet statistics
        self.normalize = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _crop(self, image, bbox):
        """
        Helper to crop the object from the full image with padding.
        """
        w_img, h_img = image.size
        xmin, ymin, xmax, ymax = bbox

        # Calculate padding
        w_box = xmax - xmin
        h_box = ymax - ymin
        pad_x = int(w_box * self.padding_factor)
        pad_y = int(h_box * self.padding_factor)

        # Apply padding respecting image bounds
        x1 = max(0, xmin - pad_x)
        y1 = max(0, ymin - pad_y)
        x2 = min(w_img, xmax + pad_x)
        y2 = min(h_img, ymax + pad_y)

        # Crop returns a copy
        return image.crop((x1, y1, x2, y2))

    def _add_noise(self, image):
        """
        Adds random Gaussian noise to the PIL image.
        """
        img_arr = np.array(image)
        # Random noise sigma between 0 and 10
        sigma = np.random.uniform(0, 10.0)
        noise = np.random.normal(0, sigma, img_arr.shape)
        noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def __call__(self, image, bbox):
        """
        Args:
            image (PIL.Image or numpy array): The full input image.
            bbox (list): [xmin, ymin, xmax, ymax].
        Returns:
            torch.Tensor: The final processed tensor.
        """
        # Ensure input is PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 1. Crop Object
        # Note: We rely on the provided bbox. We do not perturb the bbox here
        # because significant shifts would effectively change the camera's center
        # of projection relative to the object, slightly invalidating the pose.
        crop = self._crop(image, bbox)

        # 2. Augment (Train only)
        if self.is_train:
            # A. Color Jitter (80% chance)
            if random.random() < 0.8:
                crop = self.color_jitter(crop)

            # B. Gaussian Blur (50% chance) - Simulates out-of-focus camera
            if random.random() < 0.5:
                # Random radius between 0 and 2.0
                radius = random.uniform(0.1, 2.0)
                crop = crop.filter(ImageFilter.GaussianBlur(radius=radius))

            # C. Gaussian Noise (30% chance) - Simulates sensor noise
            if random.random() < 0.3:
                crop = self._add_noise(crop)

        # 3. Normalize & Resize
        crop = self.normalize(crop)

        return crop


# Helper functions for clean import in dataset.py
def get_train_transforms(image_size=224):
    return LineModRotationPredictionTransform(image_size=image_size, is_train=True)


def get_val_transforms(image_size=224):
    return LineModRotationPredictionTransform(image_size=image_size, is_train=False)
