import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from models.rgbd.depthCNN.model import DepthFeatureExtractor

LINEMOD_OBJECT_IDS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_OBJECTS = len(LINEMOD_OBJECT_IDS)

# Create mapping from object_id to one-hot index
OBJECT_ID_TO_INDEX = {obj_id: idx for idx, obj_id in enumerate(LINEMOD_OBJECT_IDS)}

class ResNetTranslationRGBD(nn.Module):
    """
    ResNet-50 backbone (ImageNet) for RGB + DepthCNN for depth + object identity conditioning + bbox + diameter for translation regression.
    Accepts RGBD input (4 channels: RGB + Depth), but splits RGB and D for separate feature extraction.
    Concatenates visual features (RGB + depth), normalized bbox, one-hot class, and normalized diameter, then predicts translation (x, y, z).
    """
    def __init__(self, num_objects: int = NUM_OBJECTS, freeze_backbone: bool = False, depth_init_method: str = "avg", depth_feat_dim: int = 256):
        super().__init__()
        self.num_objects = num_objects
        self.depth_feat_dim = depth_feat_dim

        # Load pretrained ResNet50 for RGB only (3 channels)
        self.rgb_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Only accept 3 channels for RGB
        self.rgb_backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.rgb_backbone.conv1.out_channels,
            kernel_size=self.rgb_backbone.conv1.kernel_size,
            stride=self.rgb_backbone.conv1.stride,
            padding=self.rgb_backbone.conv1.padding,
            bias=self.rgb_backbone.conv1.bias is not None
        )
        with torch.no_grad():
            self.rgb_backbone.conv1.weight.copy_(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).conv1.weight)
            if self.rgb_backbone.conv1.bias is not None:
                self.rgb_backbone.conv1.bias.copy_(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).conv1.bias)
        self.feature_dim = self.rgb_backbone.fc.in_features  # 2048
        self.rgb_backbone.fc = nn.Identity()

        # Depth feature extractor
        self.depth_cnn = DepthFeatureExtractor(out_dim=depth_feat_dim)

        # Small FC layers for bbox (4), diameter (1)
        self.bbox_fc = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
        )
        self.diameter_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Final head: features + one-hot + bbox + diameter
        self.trans_head = nn.Sequential(
            nn.Linear(self.feature_dim + depth_feat_dim + num_objects + 256 + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # Output: (x, y, z)
        )

        if freeze_backbone:
            for name, p in self.rgb_backbone.named_parameters():
                p.requires_grad = False
            # NOTE: We do NOT freeze depth_cnn because it's randomly initialized (not pretrained).
            # Freezing it would result in random, useless depth features.

    def unfreeze_backbone(self):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = True
        print("[INFO] RGB backbone unfrozen - all parameters now trainable")

    def forward(self, x, object_ids, bbox, diameter):
        """
        x: (B, 4, H, W) RGBD image (RGB + Depth channel)
        object_ids: (B,) tensor/list of object ids
        bbox: (B, 4) tensor (x_center, y_center, width, height) in pixels or normalized [0,1]
        diameter: (B, 1) tensor or (B,) (should be normalized)
        """
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :]
        rgb_feat = self.rgb_backbone(rgb)
        depth_feat = self.depth_cnn(depth)
        device = x.device
        indices = torch.tensor([OBJECT_ID_TO_INDEX[int(oid)] for oid in object_ids], device=device)
        one_hot = F.one_hot(indices, num_classes=self.num_objects).float()
        bbox_emb = self.bbox_fc(bbox)
        diam = diameter.view(-1, 1)
        diam_emb = self.diameter_embed(diam)
        combined = torch.cat([rgb_feat, depth_feat, one_hot, bbox_emb, diam_emb], dim=1)
        translation = self.trans_head(combined)
        return translation