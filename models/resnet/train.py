import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import resnet50, ResNet50_Weights

from dataset.linemod_rotation_dataset import LineModRotationDataset

def parse_args():
    p = argparse.ArgumentParser("Fine-tune ResNet50 for LINEMOD rotation (quaternion)")
    p.add_argument("--dataset_path", type=str, default="data/linemod/Linemod_preprocessed/data",
                   help="Path to LINEMOD preprocessed root (contains data/01, data/02 ... OR object folders).")
    p.add_argument("--object_id", type=int, default=0, help="LINEMOD object id (e.g., 0=all, 1=ape, 5=can, etc.)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--split_percentage", type=float, default=0.8,
                   help="Fraction used for train in your dataset implementation.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="runs_resnet")
    p.add_argument("--run_name", type=str, default="resnet50_rot")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="If set, freeze all ResNet layers except the final FC head.")
    p.add_argument("--use_object_id", action="store_true",
                   help="If set, use object identity conditioning (one-hot) for multi-object training.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from. Loads model weights only (not optimizer).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# LINEMOD object IDs (13 objects total)
# These are the valid IDs in the dataset: 1,2,4,5,6,8,9,10,11,12,13,14,15
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
        self.backbone.fc = nn.Linear(in_features, 4)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                # keep fc trainable
                if not name.startswith("fc."):
                    p.requires_grad = False

    def forward(self, x, object_ids=None):
        q = self.backbone(x)                    # (B,4)
        q = F.normalize(q, p=2, dim=1)          # unit quaternion
        return q


class ResNetRotationWithObjectID(nn.Module):
    """
    ResNet-50 backbone (ImageNet) + object identity conditioning + quaternion head.
    
    The model concatenates a one-hot encoded object ID to the visual features
    before the final FC layer. This allows the model to learn object-specific
    rotation patterns when training on multiple objects simultaneously.
    
    Architecture:
        Image -> ResNet50 (up to avgpool) -> 2048-dim features
        Object ID -> One-hot encoding -> 13-dim
        Concatenate -> 2061-dim -> FC -> 4-dim quaternion
    """
    def __init__(self, num_objects: int = NUM_OBJECTS, freeze_backbone: bool = False):
        super().__init__()
        self.num_objects = num_objects
        
        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_dim = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Remove the original FC layer - we'll extract features manually
        self.backbone.fc = nn.Identity()
        
        # New head: features (2048) + one-hot object ID (13) -> quaternion (4)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim + num_objects, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 4)
        )
        
        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = False

    def forward(self, x, object_ids):
        """
        Args:
            x: (B, 3, 224, 224) input images
            object_ids: (B,) tensor of object IDs (actual LINEMOD IDs like 1,2,4,5,...)
        
        Returns:
            (B, 4) normalized quaternions
        """
        # Extract visual features
        features = self.backbone(x)  # (B, 2048)
        
        # Convert object IDs to one-hot encoding
        # First map actual IDs to indices (0-12)
        device = x.device
        
        indices = torch.tensor([OBJECT_ID_TO_INDEX[int(oid)] for oid in object_ids], 
                               device=device)
        one_hot = F.one_hot(indices, num_classes=self.num_objects).float()  # (B, 13)
        
        # Concatenate features and object identity
        combined = torch.cat([features, one_hot], dim=1)  # (B, 2061)
        
        # Predict quaternion
        q = self.head(combined)  # (B, 4)
        q = F.normalize(q, p=2, dim=1)  # unit quaternion
        
        return q

def quat_geodesic_loss(q_pred, q_gt):
    """
    Geodesic quaternion loss - true angular distance on SO(3).
    
    This loss computes the actual rotation angle between predicted and ground truth
    quaternions, providing gradients that are aligned with rotation improvement.
    
    The absolute value of the dot product handles the sign ambiguity naturally
    (q and -q represent the same rotation).
    
    Args:
        q_pred: (B, 4) predicted quaternions [w, x, y, z], should be normalized
        q_gt: (B, 4) ground truth quaternions [w, x, y, z], should be normalized
    
    Returns:
        Mean angular distance in radians
    """
    # Ensure normalization for numerical stability
    q_pred = F.normalize(q_pred, p=2, dim=1)
    q_gt = F.normalize(q_gt, p=2, dim=1)
    
    # Dot product: cos(angle/2) for unit quaternions
    # Absolute value handles sign ambiguity (q ≡ -q)
    dot = torch.abs((q_pred * q_gt).sum(dim=1))
    
    # Clamp to avoid numerical issues with acos at boundaries
    dot = dot.clamp(0, 1 - 1e-7)
    
    # Angular distance: angle = 2 * arccos(|dot|)
    angle = 2 * torch.acos(dot)
    
    return angle.mean()

@torch.no_grad()
def quat_angle_error_deg(q_pred, q_gt):
    """
    Computes angular distance between quaternions in degrees (sign invariant).
    """
    # Ensure unit
    q_pred = F.normalize(q_pred, p=2, dim=1)
    q_gt = F.normalize(q_gt, p=2, dim=1)

    # |dot| in [0,1]
    dot = torch.abs(torch.sum(q_pred * q_gt, dim=1)).clamp(0.0, 1.0)
    # angle = 2*acos(dot)
    ang = 2.0 * torch.acos(dot)
    return (ang * 180.0 / torch.pi).mean().item()


def train_one_epoch(model, loader, optimizer, device, use_object_id=False):
    model.train()
    total_loss = 0.0
    total_ang = 0.0
    n = 0

    for batch in loader:
        if use_object_id:
            imgs, q_gt, object_ids = batch
        else:
            imgs, q_gt, _ = batch  # Ignore object_id
            object_ids = None
            
        imgs = imgs.to(device, non_blocking=True).float()
        q_gt = q_gt.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        q_pred = model(imgs, object_ids) if use_object_id else model(imgs, None)
        loss = quat_geodesic_loss(q_pred, q_gt)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_ang += quat_angle_error_deg(q_pred, q_gt) * bs
        n += bs

    return total_loss / max(n, 1), total_ang / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, use_object_id=False):
    model.eval()
    total_loss = 0.0
    total_ang = 0.0
    n = 0

    for batch in loader:
        if use_object_id:
            imgs, q_gt, object_ids = batch
        else:
            imgs, q_gt, _ = batch  # Ignore object_id
            object_ids = None
            
        imgs = imgs.to(device, non_blocking=True).float()
        q_gt = q_gt.to(device, non_blocking=True).float()

        q_pred = model(imgs, object_ids) if use_object_id else model(imgs, None)
        loss = quat_geodesic_loss(q_pred, q_gt)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_ang += quat_angle_error_deg(q_pred, q_gt) * bs
        n += bs

    return total_loss / max(n, 1), total_ang / max(n, 1)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("Device:", device)

    train_ds = LineModRotationDataset(
        root_dir=args.dataset_path,
        object_id=args.object_id if args.object_id!=0 else None,
        split="train",
        split_percentage=args.split_percentage
    )
    val_ds = LineModRotationDataset(
        root_dir=args.dataset_path,
        object_id=args.object_id if args.object_id!=0 else None,
        split="test",
        split_percentage=args.split_percentage
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False
    )

    # Choose model based on --use_object_id flag
    use_object_id = args.use_object_id
    if use_object_id:
        print("Using ResNetRotationWithObjectID (object identity conditioning enabled)")
        model = ResNetRotationWithObjectID(
            num_objects=NUM_OBJECTS,
            freeze_backbone=args.freeze_backbone
        ).to(device)
    else:
        print("Using ResNetRotation (single object or no conditioning)")
        model = ResNetRotation(freeze_backbone=args.freeze_backbone).to(device)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
            resumed_epoch = checkpoint.get('epoch', 0)
            resumed_ang = checkpoint.get('best_val_ang_deg', None)
            print(f"  Loaded weights from epoch {resumed_epoch}" + 
                  (f" (best val ang: {resumed_ang:.2f}°)" if resumed_ang else ""))
        else:
            state_dict = checkpoint
            print("  Loaded raw state dict")
        
        # Load weights (strict=False to handle potential architecture differences)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Backbone frozen: {args.freeze_backbone}")

    # Only optimize trainable params (important if freezing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    save_dir = Path(args.save_dir) / args.run_name / f"obj_{args.object_id:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_resnet50_quat.pth"
    last_path = save_dir / "last_resnet50_quat.pth"

    best_val_ang = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ang = train_one_epoch(model, train_loader, optimizer, device, use_object_id)
        va_loss, va_ang = eval_one_epoch(model, val_loader, device, use_object_id)

        scheduler.step(va_ang)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch:03d}/{args.epochs}] lr={current_lr:.1e} | "
              f"train loss={tr_loss:.4f} ang={tr_ang:.2f}° | "
              f"val loss={va_loss:.4f} ang={va_ang:.2f}°")

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "object_id": args.object_id,
            "use_object_id": use_object_id,
            "args": vars(args),
        }, last_path)

        # Save best (by val angular error)
        if va_ang < best_val_ang:
            best_val_ang = va_ang
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_ang_deg": best_val_ang,
                "object_id": args.object_id,
                "use_object_id": use_object_id,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ New best saved: {best_path} (val ang {best_val_ang:.2f}°)")

        # Save every 10 epochs
        if epoch % 10 == 0:
            epoch_path = save_dir / f"epoch_{epoch:02d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ang_deg": va_ang,
                "object_id": args.object_id,
                "use_object_id": use_object_id,
                "args": vars(args),
            }, epoch_path)
            print(f"  ✓ Checkpoint saved: {epoch_path}")

    print("Done.")

if __name__ == "__main__":
    main()
