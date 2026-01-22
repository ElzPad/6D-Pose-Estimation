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
        # REDUCED strength to preserve edge/texture details crucial for rotation estimation
        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.1,  # Reduced from 0.25 - preserve shading gradients
                contrast=0.1,  # Reduced from 0.25 - preserve edge visibility
                saturation=0.15,  # Slightly reduced
                hue=0.03,  # Reduced from 0.05
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

    # In augmentations/transforms.py
    def _crop(self, image, bbox):
        w_img, h_img = image.size
        
        # Correct unpacking for [x_center, y_center, w_box, h_box]
        x_center, y_center, w_box, h_box = bbox
        x1, y1 = x_center - w_box/2, y_center - h_box/2
        x2, y2 = x_center + w_box/2, y_center + h_box/2
        
        # --- The rest of your logic remains the same, but using correct w_box ---
        pad_x = int(w_box * self.padding_factor)
        pad_y = int(h_box * self.padding_factor)
        
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(w_img, int(x2 + pad_x))
        crop_y2 = min(h_img, int(y2 + pad_y))
        
        return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def _add_noise(self, image):
        """
        Adds random Gaussian noise to the PIL image.
        Reduced sigma to preserve fine texture details.
        """
        img_arr = np.array(image)

        sigma = np.random.uniform(0, 5.0)  # Reduced from 10.0
        noise = np.random.normal(0, sigma, img_arr.shape)
        noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def __call__(self, image, bbox):
        """
        Args:
            image (PIL.Image or numpy array): The full input image.
            bbox (list): [x_center, y_center, w_box, h_box].
        Returns:
            torch.Tensor: The final processed tensor.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # crop
        crop = self._crop(image, bbox)

        # augment (Train only)
        # REDUCED augmentation to preserve orientation cues (edges, textures)
        if self.is_train:
            # Color Jitter (60% chance) - helps with lighting robustness
            if random.random() < 0.6:
                crop = self.color_jitter(crop)

            # Gaussian Blur (30% chance) - mild blur for regularization
            # Max radius 0.8 (reduced from 2.0) to preserve edge details
            if random.random() < 0.3:
                radius = random.uniform(0.1, 0.8)
                crop = crop.filter(ImageFilter.GaussianBlur(radius=radius))

            # Gaussian Noise (20% chance, reduced from 30%)
            if random.random() < 0.2:
                crop = self._add_noise(crop)

        # Normalize & Resize
        crop = self.normalize(crop)

        return crop

class LineModTranslationPredictionTransform(object):
    def __init__(self, image_size=224, padding_factor=0.1, is_train=True):
        """
    Full-image transform (NO cropping).
    Use this for translation training/inference when you want global context.
    """
    def __init__(self, image_size=224, is_train=True):
        self.image_size = image_size
        self.is_train = is_train

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.15,
                hue=0.03,
            )

        self.normalize = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _add_noise(self, image):
        img_arr = np.array(image)
        sigma = np.random.uniform(0, 5.0)
        noise = np.random.normal(0, sigma, img_arr.shape)
        noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if self.is_train:
            if random.random() < 0.6:
                image = self.color_jitter(image)

            if random.random() < 0.3:
                radius = random.uniform(0.1, 0.8)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))

            if random.random() < 0.2:
                image = self._add_noise(image)

        return self.normalize(image)

def get_train_translation_transforms(image_size=224):
    return LineModTranslationPredictionTransform(image_size=image_size, is_train=True)


def get_val_translation_transforms(image_size=224):
    return LineModTranslationPredictionTransform(image_size=image_size, is_train=False)


def get_train_transforms(image_size=224):
    return LineModRotationPredictionTransform(image_size=image_size, is_train=True)


def get_val_transforms(image_size=224):
    return LineModRotationPredictionTransform(image_size=image_size, is_train=False)
