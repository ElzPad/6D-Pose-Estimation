import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Depth import DepthFeatureExtractor
from torchvision.models import resnet50, ResNet50_Weights

LINEMOD_OBJECT_IDS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_OBJECTS = len(LINEMOD_OBJECT_IDS)

# Create mapping from object_id to one-hot index
OBJECT_ID_TO_INDEX = {obj_id: idx for idx, obj_id in enumerate(LINEMOD_OBJECT_IDS)}

class ResNetRotation(nn.Module):
    """
    ResNet-50 backbone (ImageNet) + quaternion head (4 dims).
    Outputs raw quaternion; we normalize in forward().
    """

    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.quat_head = nn.Linear(in_features, 4)
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False

    def forward(self, x, object_ids=None):
        feats = self.backbone(x)  # (B,2048)
        q = self.quat_head(feats)
        q = F.normalize(q, p=2, dim=1)
        return q

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[INFO] Backbone unfrozen - all parameters now trainable")

class ResNetRotationWithObjectID(nn.Module):
    """
    ResNet-50 backbone (ImageNet) + object identity conditioning + quaternion head.
    The model concatenates a one-hot encoded object ID to the visual features
    before the final FC layer. This allows the model to learn object-specific
    rotation patterns when training on multiple objects simultaneously.
    """
    def __init__(self, num_objects: int = NUM_OBJECTS, freeze_backbone: bool = False):
        super().__init__()
        self.num_objects = num_objects
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        self.quat_head = nn.Sequential(
            nn.Linear(self.feature_dim + num_objects, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        )

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[INFO] Backbone unfrozen - all parameters now trainable")

    def forward(self, x, object_ids):
        features = self.backbone(x)
        device = x.device
        indices = torch.tensor([OBJECT_ID_TO_INDEX[int(oid)] for oid in object_ids], device=device)
        one_hot = F.one_hot(indices, num_classes=self.num_objects).float()
        combined = torch.cat([features, one_hot], dim=1)

        q = self.quat_head(combined)
        q = F.normalize(q, p=2, dim=1)
        return q
    


class ResNetRotationRGBD(nn.Module):
    """RGB + Depth feature fusion for quaternion prediction.

    - RGB: ResNet-50 (ImageNet) -> 2048-dim feature
    - Depth: DepthFeatureExtractor -> depth_feat_dim
    - Fusion: concat -> MLP head -> quaternion
    """

    def __init__(
        self,
        depth_feat_dim: int = 256,
        num_objects: int = NUM_OBJECTS,
        use_object_id: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.use_object_id = use_object_id
        self.num_objects = num_objects

        self.rgb_backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.rgb_feat_dim = self.rgb_backbone.fc.in_features  # 2048
        self.rgb_backbone.fc = nn.Identity()

        self.depth_net = DepthFeatureExtractor(out_dim=depth_feat_dim)

        head_in = self.rgb_feat_dim + depth_feat_dim + (num_objects if use_object_id else 0)
        self.quat_head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 4),
        )

        if freeze_backbone:
            for p in self.rgb_backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.rgb_backbone.parameters():
            p.requires_grad = True
        print("[INFO] RGB backbone unfrozen - all parameters now trainable")

    def forward(self, rgb: torch.Tensor, depth_mask: torch.Tensor, object_ids=None) -> torch.Tensor:
        rgb_feats = self.rgb_backbone(rgb)          # (B,2048)
        depth_feats = self.depth_net(depth_mask)    # (B,depth_feat_dim)
        feats = torch.cat([rgb_feats, depth_feats], dim=1)

        if self.use_object_id:
            if object_ids is None:
                raise ValueError("object_ids must be provided when use_object_id=True")
            device = rgb.device
            indices = torch.tensor([OBJECT_ID_TO_INDEX[int(oid)] for oid in object_ids], device=device)
            one_hot = F.one_hot(indices, num_classes=self.num_objects).float()
            feats = torch.cat([feats, one_hot], dim=1)

        q = self.quat_head(feats)
        q = F.normalize(q, p=2, dim=1)
        return q
