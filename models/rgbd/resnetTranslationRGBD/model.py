import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights

LINEMOD_OBJECT_IDS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_OBJECTS = len(LINEMOD_OBJECT_IDS)

# Create mapping from object_id to one-hot index
OBJECT_ID_TO_INDEX = {obj_id: idx for idx, obj_id in enumerate(LINEMOD_OBJECT_IDS)}

def adapt_conv1_for_rgbd(conv1: nn.Conv2d, init_method: str = "avg") -> nn.Conv2d:
    """
    Adapt a 3-channel conv1 layer to accept 4 channels (RGBD).
    
    Args:
        conv1: Original conv1 layer with 3 input channels
        init_method: How to initialize the depth channel weights:
            - "avg": Average of RGB weights (recommended - depth often correlates with luminance)
            - "zero": Zero initialization (depth starts with no contribution)
            - "green": Copy green channel weights (green ≈ luminance in human vision)
            - "scaled_avg": Average of RGB weights scaled by 0.5 (conservative start)
    
    Returns:
        New conv1 layer with 4 input channels
    """
    # Create new conv layer with 4 input channels
    new_conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None
    )
    
    with torch.no_grad():
        # Copy RGB weights (out_channels, 3, kH, kW) -> (out_channels, 3, kH, kW)
        new_conv1.weight[:, :3, :, :] = conv1.weight.clone()
        
        # Initialize depth channel weights based on method
        if init_method == "avg":
            # Average of RGB channels - good default since depth correlates with scene structure
            depth_weights = conv1.weight.mean(dim=1, keepdim=True)
        elif init_method == "zero":
            # Zero init - depth has no initial contribution, learned from scratch
            depth_weights = torch.zeros_like(conv1.weight[:, :1, :, :])
        elif init_method == "green":
            # Copy green channel - often captures luminance-like information
            depth_weights = conv1.weight[:, 1:2, :, :].clone()
        elif init_method == "scaled_avg":
            # Scaled average - conservative initialization
            depth_weights = conv1.weight.mean(dim=1, keepdim=True) * 0.5
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        new_conv1.weight[:, 3:4, :, :] = depth_weights
        
        # Copy bias if present
        if conv1.bias is not None:
            new_conv1.bias = nn.Parameter(conv1.bias.clone())
    
    return new_conv1

class ResNetTranslationRGBD(nn.Module):
    """
    ResNet-50 backbone (ImageNet) + object identity conditioning + bbox + diameter for translation regression.
    Accepts RGBD input (4 channels: RGB + Depth).
    Concatenates visual features, normalized bbox, one-hot class, and normalized diameter, then predicts translation (x, y, z).
    """
    def __init__(self, num_objects: int = NUM_OBJECTS, freeze_backbone: bool = False, depth_init_method: str = "avg"):
        super().__init__()
        self.num_objects = num_objects
        
        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Adapt conv1 to accept 4 channels (RGBD) instead of 3 (RGB)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = adapt_conv1_for_rgbd(original_conv1, init_method=depth_init_method)
        print(f"[INFO] Adapted conv1 for RGBD input (4 channels), depth init: '{depth_init_method}'")
        
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
        x: (B, 4, H, W) RGBD image (RGB + Depth channel)
        object_ids: (B,) tensor/list of object ids
        bbox: (B, 4) tensor (x_center, y_center, width, height) in pixels or normalized [0,1]
        diameter: (B, 1) tensor or (B,) (should be normalized)
        
        Note: Depth channel should be normalized to similar range as RGB (e.g., [0, 1] or ImageNet stats)
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