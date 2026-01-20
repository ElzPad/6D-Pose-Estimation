import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights

LINEMOD_OBJECT_IDS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_OBJECTS = len(LINEMOD_OBJECT_IDS)

# Create mapping from object_id to one-hot index
OBJECT_ID_TO_INDEX = {obj_id: idx for idx, obj_id in enumerate(LINEMOD_OBJECT_IDS)}

class ResNetTranslationWithObjectID(nn.Module):
    """
    ResNet-50 backbone (ImageNet) + object identity conditioning + bbox + diameter for translation regression.
    Concatenates visual features, normalized bbox, one-hot class, and normalized diameter, then predicts translation (x, y, z).
    """
    def __init__(self, num_objects: int = NUM_OBJECTS, freeze_backbone: bool = False):
        super().__init__()
        self.num_objects = num_objects
        
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_dim = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

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
            nn.Linear(self.feature_dim + num_objects + 256 + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # Output: (x, y, z)
        )

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("[INFO] Backbone unfrozen - all parameters now trainable")

    def forward(self, x, object_ids, bbox, diameter):
        """
        x: (B, 3, H, W) image
        object_ids: (B,) tensor/list of object ids
        bbox: (B, 4) tensor (x_center, y_center, width, height) in pixels or normalized [0,1]
        diameter: (B, 1) tensor or (B,) (should be normalized)
        """
        features = self.backbone(x)
        device = x.device
        indices = torch.tensor([OBJECT_ID_TO_INDEX[int(oid)] for oid in object_ids], device=device)
        one_hot = F.one_hot(indices, num_classes=self.num_objects).float()

        bbox_emb = self.bbox_fc(bbox)
        
        diam = diameter.view(-1, 1)                  # (B,1)
        diam_emb = self.diameter_embed(diam)         # (B,32)

        combined = torch.cat([features, one_hot, bbox_emb, diam_emb], dim=1)
        translation = self.trans_head(combined)
        return translation